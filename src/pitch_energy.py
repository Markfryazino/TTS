import src.xcmyz_utils.audio.hparams_audio as audio_params

import pyworld
import os
import torch
import numpy as np
import torchaudio

from tqdm import trange


def wav2pitch(wav):
    pitch, t = pyworld.dio(wav, audio_params.sampling_rate, frame_period=audio_params.hop_length * 1000 / audio_params.sampling_rate)
    pitch = pyworld.stonemask(wav, pitch, t, audio_params.sampling_rate)
    return pitch


def wav2energy(wav):
    stfts = torch.stft(
        torch.tensor(wav),
        audio_params.filter_length,
        audio_params.hop_length,
        audio_params.win_length,
        return_complex=False,
    )
    
    return ((stfts ** 2).sum((0, 2)) ** 0.5).numpy()


def process_wav(train_config, idx, wav_files):
    wav_data = torchaudio.load(os.path.join(train_config.wav_path, wav_files[idx]))[0].squeeze(0)
    wav = wav_data.numpy().astype(np.float64)
    pitch = wav2pitch(wav)
    energy = wav2energy(wav)

    np.save(os.path.join(train_config.pitch_path, f"{idx}.npy"), pitch)
    np.save(os.path.join(train_config.energy_path, f"{idx}.npy"), energy)

    return pitch, energy


def process_all_wavs(train_config):
    wav_files = sorted(os.listdir(train_config.wav_path))

    pitch_min = energy_min = np.inf
    pitch_max = energy_max = -np.inf

    os.makedirs(train_config.pitch_path, exist_ok=True)
    os.makedirs(train_config.energy_path, exist_ok=True)
    
    for idx in trange(len(wav_files)):
        pitch, energy = process_wav(train_config, idx, wav_files)
        pitch_min = min(pitch_min, pitch.min())
        pitch_max = max(pitch_max, pitch.max())
        energy_min = min(energy_min, energy.min())
        energy_max = max(energy_max, energy.max())

    np.save(os.path.join(train_config.pitch_path, "boundaries.npy"), np.array([pitch_min, pitch_max]))
    np.save(os.path.join(train_config.energy_path, "boundaries.npy"), np.array([energy_min, energy_max]))
