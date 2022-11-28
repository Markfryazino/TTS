import torch
import torch.nn as nn
import os
import wandb

from functools import partial
from torch.optim.lr_scheduler  import OneCycleLR
from tqdm import tqdm

import src.xcmyz_utils.utils as utils

from src.data_processing import get_data_to_buffer, BufferDataset, collate_fn_tensor
from src.configs import TrainConfig, FastSpeech2Config, MelSpectrogramConfig
from src.model import FastSpeech2
from src.training.loss import FastSpeechLoss
from src.training.synthesis import synthesis, prepare_texts_for_synthesis


def log_audio(model, wave_glow, prepared_texts, step, device="cuda:0", prefix=""):
    model.eval()
    for length_alpha in [0.8, 1.0, 1.2]:
        for pitch_alpha in [0.8, 1.0, 1.2]:
            for energy_alpha in [0.8, 1.0, 1.2]:
                settings = dict(
                    length_alpha=length_alpha,
                    pitch_alpha=pitch_alpha,
                    energy_alpha=energy_alpha
                )
                for i, (sequence, src_pos, raw_text) in enumerate(prepared_texts):
                    audio, sr = synthesis(model, wave_glow, sequence, src_pos, "test.wav", device=device, **settings)
                    wandb.log({
                        f"{prefix}step-{step}-text-{i}-settings-{settings}": wandb.Audio(
                            audio, sr, 
                            caption=f"Step: {step}. Settings: {settings}. Text: {raw_text}"
                        )
                    },
                    step=step)

    model.train()


def prepare(model_config: FastSpeech2Config, train_config: TrainConfig, mel_config: MelSpectrogramConfig):
    model = FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)

    os.makedirs(train_config.checkpoint_path, exist_ok=True)

    fastspeech_loss = FastSpeechLoss(model_config=model_config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    buffer = get_data_to_buffer(TrainConfig())
    dataset = BufferDataset(buffer)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn_tensor, batch_expand_size=train_config.batch_expand_size),
        drop_last=True,
        num_workers=0
    )

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(train_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })

    wave_glow = utils.get_WaveGlow()
    for path in os.listdir("."):
        if path.endswith(".patch"):
            os.remove(path)
    wave_glow = wave_glow.to(train_config.device)

    return model, train_loader, fastspeech_loss, optimizer, scheduler, wave_glow

def train(train_config: TrainConfig, model: FastSpeech2, train_loader: torch.utils.data.DataLoader, 
          fastspeech_loss: FastSpeechLoss, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler,
          wave_glow: torch.nn.Module):

    current_step = 0
    tqdm_bar = tqdm(total=train_config.epochs * len(train_loader) * train_config.batch_expand_size)

    prepared_texts = prepare_texts_for_synthesis(train_config.test_audio, train_config.text_cleaners, device=train_config.device)
    final_texts = prepare_texts_for_synthesis(train_config.test_audio + train_config.additional_test_audio, 
                                              train_config.text_cleaners, device=train_config.device)

    wandb.init(
       project=train_config.wandb_project,
       entity=train_config.wandb_entity
    )

    t_l = m_l = d_l = p_l = e_l = 0.

    try:
        for epoch in range(train_config.epochs):
            for i, batchs in enumerate(train_loader):
                # real batch start here
                for j, db in enumerate(batchs):
                    current_step += 1
                    tqdm_bar.update(1)

                    # Get Data
                    character = db["text"].long().to(train_config.device)
                    mel_target = db["mel_target"].float().to(train_config.device)
                    duration = db["duration"].int().to(train_config.device)
                    pitch = db["pitch"].to(train_config.device)
                    energy = db["energy"].to(train_config.device)
                    mel_pos = db["mel_pos"].long().to(train_config.device)
                    src_pos = db["src_pos"].long().to(train_config.device)
                    max_mel_len = db["mel_max_len"]

                    # Forward
                    mel_output, \
                    duration_predictor_output, \
                    pitch_predictor_output, \
                    energy_predictor_output = model(character,
                                                    src_pos,
                                                    mel_pos=mel_pos,
                                                    mel_max_length=max_mel_len,
                                                    length_target=duration,
                                                    pitch_target=pitch,
                                                    energy_target=energy)

                    # Calc Loss
                    mel_loss, duration_loss, pitch_loss, energy_loss = \
                        fastspeech_loss(mel_output,
                                        duration_predictor_output,
                                        pitch_predictor_output,
                                        energy_predictor_output,
                                        mel_target,
                                        duration,
                                        pitch,
                                        energy)

                    total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                    # Logger
                    t_l += total_loss.detach().cpu().numpy()
                    m_l += mel_loss.detach().cpu().numpy()
                    d_l += duration_loss.detach().cpu().numpy()
                    p_l += pitch_loss.detach().cpu().numpy()
                    e_l += energy_loss.detach().cpu().numpy()

                    if current_step % train_config.log_step == 0:
                        wandb.log({
                            "duration_loss": d_l / train_config.log_step,
                            "mel_loss": m_l / train_config.log_step,
                            "pitch_loss": p_l / train_config.log_step,
                            "energy_loss": e_l / train_config.log_step,
                            "total_loss": t_l / train_config.log_step,
                        },
                        step=current_step)
                        t_l = m_l = d_l = p_l = e_l = 0.

                    # Backward
                    total_loss.backward()

                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(
                        model.parameters(), train_config.grad_clip_thresh)
                    
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                    if current_step % train_config.eval_step == 0:
                        log_audio(model, wave_glow, prepared_texts, current_step, train_config.device)

                    if current_step % train_config.save_step == 0:
                        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                        )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                        wandb.save(os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                        print("save model at step %d ..." % current_step)
    except KeyboardInterrupt:
        print("Finish training!")

    log_audio(model, wave_glow, final_texts, current_step, train_config.device, prefix="final/")

    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
    )}, os.path.join(train_config.checkpoint_path, 'final_checkpoint.pth.tar' % current_step))
    wandb.save(os.path.join(train_config.checkpoint_path, 'checkpoint_final.pth.tar' % current_step))
    print("save model at step %d ..." % current_step)

    return model
