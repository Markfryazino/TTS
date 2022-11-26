import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask):
        q = self._reshape_to_heads(self.query_linear(q))
        k = self._reshape_to_heads(self.key_linear(k))
        v = self._reshape_to_heads(self.value_linear(v))

        attn_mask = self._reshape_mask(mask)
        output = self._attention_forward(q, k, v, attn_mask)
        output = self.output_linear(self._reshape_from_heads(output))
        return output

    def _attention_forward(self, q, k, v, mask):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, v)

    def _reshape_to_heads(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)\
                .transpose(1, 2)\
                .reshape(batch_size * self.num_heads, seq_len, self.head_dim)

    def _reshape_mask(self, mask):
        seq_len = mask.size(1)

        reshaped_mask = mask.repeat([self.num_heads, 1])
        scorelike_mask_d1 = reshaped_mask.repeat_interleave(seq_len, 1).reshape(-1, seq_len, seq_len)
        return scorelike_mask_d1 * scorelike_mask_d1.transpose(1, 2)

    def _reshape_from_heads(self, x):
        batch_size, seq_len, head_dim = x.size()
        batch_size //= self.num_heads

        return x.reshape(batch_size, self.num_heads, seq_len, head_dim)\
                .transpose(1, 2)\
                .reshape(batch_size, seq_len, self.hidden_dim)


class FFTBlock(nn.Module):
    def __init__(self, hidden_size=64, ff_dim=64, attn_num_heads=8, dropout_prob=0.0, conv_kernel_sizes=(9, 1)):
        super(FFTBlock, self).__init__()

        self.attention = SelfAttention(
            hidden_dim=hidden_size, num_heads=attn_num_heads
        )
        self.feedforward = nn.Sequential(
            nn.Conv1d(hidden_size, ff_dim, kernel_size=conv_kernel_sizes[0], padding=conv_kernel_sizes[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Conv1d(ff_dim, hidden_size, kernel_size=conv_kernel_sizes[1], padding=conv_kernel_sizes[1] // 2),
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, X, mask=None):
        if mask is None:
            mask = torch.ones((X.size(0), X.size(1)))

        X_normed = self.layer_norm(X)
        X_attn = self.attention(q=X_normed, k=X_normed, v=X_normed, mask=mask) + X
        X_ff = self.feedforward(X_attn.transpose(2, 1))
        return X_ff.transpose(2, 1) + X_attn
