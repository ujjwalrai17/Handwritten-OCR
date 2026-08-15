"""
CRNN + CTC Model with Additive Attention
Architecture:
  CNN backbone  → feature maps (B, C, H', W')
  Collapse H'   → sequence (B, W', C*H')
  BiLSTM stack  → context vectors (B, W', 2*hidden)
  Attention     → weighted context (B, W', 2*hidden)   [optional]
  Linear        → logits (B, W', vocab_size)
  CTC decode    → text

Designed for cursive/connected script where attention over the full
sequence helps disambiguate overlapping strokes that look identical
in isolation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config.settings import cfg

C = cfg.crnn


# ── CNN Backbone ──────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: tuple | None = None):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(pool))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CNNBackbone(nn.Module):
    """
    VGG-style CNN that maps (B,1,H,W) → (B, C_last, H/16, W/4).
    Asymmetric pooling preserves horizontal resolution for sequence modeling.
    """
    def __init__(self, channels: list[int]):
        super().__init__()
        # channels = [1, 64, 128, 256, 256, 512, 512]
        self.layers = nn.ModuleList()
        pool_schedule = [
            (2, 2),   # after ch[1]: H/2,  W/2
            (2, 2),   # after ch[2]: H/4,  W/4
            (2, 1),   # after ch[3]: H/8,  W/4
            None,     # ch[4]: no pool
            (2, 1),   # after ch[5]: H/16, W/4
            None,     # ch[6]: no pool
        ]
        for i in range(len(channels) - 1):
            self.layers.append(ConvBlock(channels[i], channels[i + 1], pool_schedule[i]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x  # (B, C_last, H/16, W/4)


# ── Additive Attention ────────────────────────────────────────────────────────

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over the RNN output sequence.
    Allows the model to focus on relevant stroke regions when decoding
    ambiguous overlapping characters.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        # rnn_out: (B, T, H)
        weights = torch.softmax(self.score(rnn_out), dim=1)  # (B, T, 1)
        context = (weights * rnn_out).sum(dim=1, keepdim=True)  # (B, 1, H)
        # Broadcast context and add residual
        enhanced = rnn_out + context.expand_as(rnn_out)
        return enhanced  # (B, T, H)


# ── Full CRNN ─────────────────────────────────────────────────────────────────

class CRNN(nn.Module):
    """
    CRNN + CTC with optional additive attention.

    Input:  (B, 1, H, W)  — grayscale line image, H fixed to cfg.crnn.input_height
    Output: (T, B, vocab_size) — CTC logits (time-first for nn.CTCLoss)
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.cnn = CNNBackbone(C.cnn_channels)

        # After CNN: feature height = input_height / 16
        cnn_out_h = C.input_height // 16
        cnn_out_c = C.cnn_channels[-1]
        rnn_input_size = cnn_out_c * cnn_out_h

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=C.rnn_hidden,
            num_layers=C.rnn_layers,
            bidirectional=C.rnn_bidirectional,
            batch_first=True,
            dropout=C.dropout if C.rnn_layers > 1 else 0.0,
        )
        rnn_out_dim = C.rnn_hidden * (2 if C.rnn_bidirectional else 1)

        self.attention = AdditiveAttention(rnn_out_dim) if C.use_attention else None
        self.dropout = nn.Dropout(C.dropout)
        self.fc = nn.Linear(rnn_out_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        feat = self.cnn(x)                          # (B, C, H', W')
        B, C_dim, H_prime, W_prime = feat.shape
        # Collapse spatial height into channel dim, treat width as time
        feat = feat.permute(0, 3, 1, 2)             # (B, W', C, H')
        feat = feat.reshape(B, W_prime, C_dim * H_prime)  # (B, T, features)

        rnn_out, _ = self.rnn(feat)                 # (B, T, rnn_out_dim)

        if self.attention is not None:
            rnn_out = self.attention(rnn_out)

        rnn_out = self.dropout(rnn_out)
        logits = self.fc(rnn_out)                   # (B, T, vocab_size)
        return logits.permute(1, 0, 2)              # (T, B, vocab_size) for CTCLoss


# ── Charset / Vocabulary ──────────────────────────────────────────────────────

CHARSET = (
    " !\"#$%&'()*+,-./"
    "0123456789:;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~"
)
# Index 0 reserved for CTC blank
CHAR2IDX = {c: i + 1 for i, c in enumerate(CHARSET)}
IDX2CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
VOCAB_SIZE = len(CHARSET) + 1  # +1 for blank


def encode_text(text: str) -> list[int]:
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX]


def decode_ctc(indices: list[int], blank_idx: int = 0) -> str:
    """Greedy CTC decode: collapse repeats, remove blanks."""
    result = []
    prev = None
    for idx in indices:
        if idx != blank_idx and idx != prev:
            result.append(IDX2CHAR.get(idx, ""))
        prev = idx
    return "".join(result)
