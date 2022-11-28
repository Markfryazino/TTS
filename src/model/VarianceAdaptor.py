import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configs import FastSpeech2Config


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class BaseFeaturePredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeech2Config):
        super(BaseFeaturePredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.feature_predictor_filter_size
        self.kernel = model_config.feature_predictor_kernel_size
        self.conv_output_size = model_config.feature_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class PitchRegulator(nn.Module):
    def __init__(self, model_config):
        super(PitchRegulator, self).__init__()
        self.pitch_predictor = BaseFeaturePredictor(model_config)
        self.max_log_pitch = torch.log1p(torch.tensor(model_config.max_pitch))

        self.boundaries = torch.linspace(0, 1, model_config.pitch_vocab)
        self.embedding = nn.Embedding(model_config.pitch_vocab, model_config.encoder_dim)

    def forward(self, x, alpha=1.0, target=None):
        pitch_predictor_output = self.pitch_predictor(x)
        if target is not None:
            log_pitch = torch.log1p(target) / self.max_log_pitch
        else:
            log_pitch = torch.log1p((torch.exp(pitch_predictor_output * self.max_log_pitch) - 1) * alpha) / self.max_log_pitch
        log_pitch = torch.clamp(log_pitch, 0, 1)
        emb_idxs = torch.bucketize(log_pitch, self.boundaries.to(log_pitch.device))
        embeddings = self.embedding(emb_idxs)
        return x + embeddings, pitch_predictor_output


class EnergyRegulator(nn.Module):
    def __init__(self, model_config):
        super(EnergyRegulator, self).__init__()
        self.energy_predictor = BaseFeaturePredictor(model_config)
        self.max_energy = model_config.max_energy

        self.boundaries = torch.linspace(0, 1, model_config.energy_vocab)
        self.embedding = nn.Embedding(model_config.energy_vocab, model_config.encoder_dim)

    def forward(self, x, alpha=1.0, target=None):
        energy_predictor_output = self.energy_predictor(x)
        if target is not None:
            energy = target / self.max_energy
        else:
            energy = energy_predictor_output * alpha
        energy = torch.clamp(energy, 0, 1)
        emb_idxs = torch.bucketize(energy, self.boundaries.to(energy.device))
        embeddings = self.embedding(emb_idxs)
        return x + embeddings, energy_predictor_output

    
class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = BaseFeaturePredictor(model_config)
        self.max_log_duration = torch.log1p(torch.tensor(model_config.max_duration))

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = ((torch.exp(duration_predictor_output * self.max_log_duration) - 1) * alpha + 0.5).int()
            output = self.LR(x, duration_predictor_output, mel_max_length)
            mel_pos = torch.stack([
                torch.Tensor([i + 1 for i in range(output.size(1))])
            ]).long().to(x.device)
        return output, mel_pos


class VarianceAdaptor(nn.Module):
    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()

        self.length_regulator = LengthRegulator(model_config)
        self.pitch_regulator = PitchRegulator(model_config)
        self.energy_regulator = EnergyRegulator(model_config)

    def forward(self, x, length_target, pitch_target, energy_target, length_alpha=1.0, 
                pitch_alpha=1.0, energy_alpha=1.0, mel_max_length=None):
        x, duration_predictor_output = self.length_regulator(x, length_alpha, length_target, mel_max_length)
        x, pitch_prediction_output = self.pitch_regulator(x, pitch_alpha, pitch_target)
        x, energy_prediction_output = self.energy_regulator(x, energy_alpha, energy_target)
        return x, duration_predictor_output, pitch_prediction_output, energy_prediction_output
