import src.xcmyz_utils.waveglow as waveglow
from src.xcmyz_utils.text import text_to_sequence
import src.xcmyz_utils.audio as audio
import src.xcmyz_utils.utils as utils

import numpy as np
import torch


def prepare_texts_for_synthesis(texts, text_cleaners, device="cuda:0"):
    result = []
    for raw_text in texts:
        text = np.array(text_to_sequence(raw_text, text_cleaners))
        text = np.stack([text])
        src_pos = np.array([i+1 for i in range(text.shape[1])])
        src_pos = np.stack([src_pos])
        sequence = torch.from_numpy(text).long().to(device)
        src_pos = torch.from_numpy(src_pos).long().to(device)
        result.append((sequence, src_pos, raw_text))
    return result


def synthesis(fs_model, wv_model, sequence, src_pos, out_path, length_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0, device="cuda:0"):
    with torch.no_grad():
        mel = fs_model.forward(sequence, src_pos, length_alpha=length_alpha, pitch_alpha=pitch_alpha, energy_alpha=energy_alpha)
    mel = mel.contiguous().transpose(1, 2).to(device)
    return waveglow.inference.inference(mel, wv_model, out_path)
