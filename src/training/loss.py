import torch
import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel, duration_predicted, pitch_predicted, energy_predicted, mel_target, duration_predictor_target,
                pitch_predictor_target, energy_predictor_target):
        mel_loss = self.mse_loss(mel, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predicted,
                                                torch.log1p(duration_predictor_target.float()))

        pitch_predictor_loss = self.mse_loss(pitch_predicted,
                                             torch.log1p(pitch_predictor_target.float()))

        energy_predictor_loss = self.mse_loss(energy_predicted, energy_predictor_target.float())

        return mel_loss, duration_predictor_loss, pitch_predictor_loss, energy_predictor_loss
