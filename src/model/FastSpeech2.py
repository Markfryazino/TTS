import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.FFT import FFTBlock
from src.model.VarianceAdaptor import VarianceAdaptor


def get_non_pad_mask(seq, pad_token):
    assert seq.dim() == 2
    return seq.ne(pad_token).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q, pad_token):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(pad_token)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        self.pad_token = model_config.PAD

        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            dropout=model_config.dropout,
            conv_kernel_sizes=model_config.fft_conv1d_kernels
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, pad_token=self.pad_token)
        non_pad_mask = get_non_pad_mask(src_seq, pad_token=self.pad_token)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        self.pad_token = model_config.PAD

        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            dropout_prob=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, pad_token=self.pad_token)
        non_pad_mask = get_non_pad_mask(enc_pos, pad_token=self.pad_token)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        return dec_output


class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, mel_config):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(model_config)
        self.variance_adapter = VarianceAdaptor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, 
                length_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
        x = self.encoder(src_seq, src_pos)

        if self.training:
            output, duration_predictor_output = self.variance_adapter(
                x, length_target, length_alpha, 
                pitch_alpha, energy_alpha, mel_max_length
            )
            output = self.decoder(output, mel_pos)
            output = self.mask_tensor(output, mel_pos, mel_max_length)
            return output, duration_predictor_output
        else:
            output, mel_pos = self.variance_adapter(
                x, length_target, length_alpha, 
                pitch_alpha, energy_alpha, mel_max_length
            )
            output = self.decoder(output, mel_pos)
            return output
