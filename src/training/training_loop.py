import torch
import torch.nn as nn
import os
import wandb

from functools import partial
from torch.optim.lr_scheduler  import OneCycleLR
from tqdm import tqdm

from src.data_processing import get_data_to_buffer, BufferDataset, collate_fn_tensor
from src.configs import TrainConfig, FastSpeech2Config, MelSpectrogramConfig
from src.model import FastSpeech2
from src.training.loss import FastSpeechLoss


def prepare(model_config: FastSpeech2Config, train_config: TrainConfig, mel_config: MelSpectrogramConfig):
    model = FastSpeech2(model_config, mel_config)
    model = model.to(train_config.device)

    fastspeech_loss = FastSpeechLoss()

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

    return model, train_loader, fastspeech_loss, optimizer, scheduler

def train(train_config: TrainConfig, model: FastSpeech2, train_loader: torch.utils.data.DataLoader, 
          fastspeech_loss: FastSpeechLoss, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler._LRScheduler):

    current_step = 0
    tqdm_bar = tqdm(total=train_config.epochs * len(train_loader) * train_config.batch_expand_size)

    wandb.init(
       project=train_config.wandb_project,
       entity=train_config.wandb_entity
    )

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
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, duration_predictor_output = model(character,
                                                            src_pos,
                                                            mel_pos=mel_pos,
                                                            mel_max_length=max_mel_len,
                                                            length_target=duration)

                # Calc Loss
                mel_loss, duration_loss = fastspeech_loss(mel_output,
                                                        duration_predictor_output,
                                                        mel_target,
                                                        duration)
                total_loss = mel_loss + duration_loss

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()

                wandb.log({
                   "duration_loss": d_l,
                   "mel_loss": m_l,
                   "total_loss": t_l
                })

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

    return model
