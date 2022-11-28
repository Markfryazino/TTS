import argparse
import os
import torch
import json
import wandb

from scipy.io.wavfile import write, read
from tqdm import tqdm

from src.model import FastSpeech2
from src.configs import MelSpectrogramConfig, FastSpeech2Config, TrainConfig
from src.training.synthesis import prepare_texts_for_synthesis, synthesis

import src.xcmyz_utils.utils as utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to FastSpeech2 checkpoint")
    parser.add_argument("--texts", type=str, required=True, help="Path to json file with texts")
    parser.add_argument("--durations", nargs="+", help="Values for duration alpha", default=[0.8, 1.0, 1.2])
    parser.add_argument("--pitches", nargs="+", help="Values for pitch alpha", default=[0.8, 1.0, 1.2])
    parser.add_argument("--energies", nargs="+", help="Values for energy alpha", default=[0.8, 1.0, 1.2])
    parser.add_argument("--use-wandb", action="store_true", help="Whether to upload results to wandb")
    parser.add_argument("--save-path", default="./test_wavs/", help="Where to save WAV files")
    parser.add_argument("--device", default="cuda:0", help="Device for synthesis")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    model = FastSpeech2(FastSpeech2Config, MelSpectrogramConfig)
    model.load_state_dict(torch.load(args.model)["model"])
    model = model.to(args.device)


    wave_glow = utils.get_WaveGlow(args.device)
    for path in os.listdir("."):
        if path.endswith(".patch"):
            os.remove(path)
    wave_glow = wave_glow.to(args.device)

    with open(args.texts) as f:
        texts = json.load(f)

    prepared_texts = prepare_texts_for_synthesis(texts, TrainConfig.text_cleaners, args.device)

    if args.use_wandb:
        wandb.init(project="TTS")

    model.eval()

    pbar = tqdm(total=len(args.durations) * len(args.pitches) * len(args.energies) * len(texts))

    for length_alpha in args.durations:
        for pitch_alpha in args.pitches:
            for energy_alpha in args.energies:
                settings = dict(
                    length_alpha=length_alpha,
                    pitch_alpha=pitch_alpha,
                    energy_alpha=energy_alpha
                )
                for i, (sequence, src_pos, raw_text) in enumerate(prepared_texts):
                    name = f"text-{i}-settings-{settings}.wav"
                    audio, sr = synthesis(model, wave_glow, sequence, src_pos, "test.wav", device=args.device, **settings)
                    write(os.path.join(args.save_path, f"{name}.wav"), sr, audio.astype("int16"))
                    sr, audio = read(os.path.join(args.save_path, f"{name}.wav"))

                    if args.use_wandb:
                        wandb.log({
                            name: wandb.Audio(
                                audio, sr,
                                caption=raw_text
                            )
                        })
                    pbar.update(1)


if __name__ == "__main__":
    main()
