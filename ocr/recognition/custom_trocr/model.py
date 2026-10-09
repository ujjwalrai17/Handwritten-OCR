"""
Custom TrOCR — Built From Scratch
===================================
Full handwritten text recognition model using only PyTorch primitives.

Architecture:
  PatchEmbedding       — splits grayscale line image into 16x16 patches
  VisionTransformerEncoder — 4-layer ViT encoder (self-attention + FFN)
  TransformerDecoder   — 4-layer decoder (masked SA + cross-attention + FFN)
  CharacterTokenizer   — character-level vocab (100 chars)

Input:  (B, 1, 128, 1024) grayscale line image
Output: (B, T, vocab_size) logits
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PATCH EMBEDDING
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Splits a grayscale line image into non-overlapping 16x16 patches
    and projects each patch to embed_dim using Conv2d.

    Input:  (B, 1, 128, 1024)
    Output: (B, N, 256)   N = (128/16)*(1024/16) = 8*64 = 512 patches
    """
    def __init__(self, image_height=128, image_width=1024,
                 patch_size=16, in_channels=1, embed_dim=256, dropout=0.1):
        super().__init__()
        assert image_height % patch_size == 0
        assert image_width  % patch_size == 0

        self.num_patches = (image_height // patch_size) * (image_width // patch_size)

        # Conv2d with kernel=stride=patch_size = patch extraction + linear projection
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        # Learnable positional embedding — one per patch
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        x = self.projection(x)          # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1,2) # (B, N, embed_dim)
        x = self.norm(x + self.pos_embedding)
        return self.dropout(x)


# =============================================================================
# VISION TRANSFORMER ENCODER
# =============================================================================

class EncoderBlock(nn.Module):
    """
    Single ViT encoder layer:
      x = x + SelfAttention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))
    """
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                            dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class VisionTransformerEncoder(nn.Module):
    """
    Stacks num_layers EncoderBlocks.
    Input:  (B, N, embed_dim)  — patch embeddings
    Output: (B, N, embed_dim)  — enriched visual features
    """
    def __init__(self, embed_dim=256, num_heads=8,
                 num_layers=4, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# =============================================================================
# TRANSFORMER DECODER
# =============================================================================

class DecoderBlock(nn.Module):
    """
    Single decoder layer:
      1. Masked Self-Attention  — attend to previously generated tokens only
      2. Cross-Attention        — attend to visual encoder features
      3. Feed-Forward Network
    """
    def __init__(self, embed_dim=256, num_heads=8, ff_dim=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads,
                                                 dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # 1. Masked self-attention
        n = self.norm1(tgt)
        sa, _ = self.self_attn(n, n, n, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.drop(sa)
        # 2. Cross-attention with visual features
        n = self.norm2(tgt)
        ca, _ = self.cross_attn(n, memory, memory)
        tgt = tgt + self.drop(ca)
        # 3. Feed-forward
        tgt = tgt + self.drop(self.ff(self.norm3(tgt)))
        return tgt


class TransformerDecoder(nn.Module):
    """
    Full autoregressive text decoder.
    Input:  token_ids (B, T) + memory (B, N, embed_dim)
    Output: logits (B, T, vocab_size)
    """
    def __init__(self, vocab_size, embed_dim=256, num_heads=8,
                 num_layers=4, ff_dim=1024, max_len=128, dropout=0.1):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed   = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm       = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.drop        = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    @staticmethod
    def _causal_mask(T, device):
        return torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()

    def forward(self, tgt_ids, memory, tgt_key_padding_mask=None):
        T = tgt_ids.size(1)
        causal = self._causal_mask(T, tgt_ids.device)
        x = self.drop(
            self.token_embed(tgt_ids) * math.sqrt(self.token_embed.embedding_dim)
            + self.pos_embed[:, :T, :]
        )
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal,
                      tgt_key_padding_mask=tgt_key_padding_mask)
        return self.output_proj(self.norm(x))   # (B, T, vocab_size)


# =============================================================================
# FULL CUSTOM TrOCR MODEL
# =============================================================================

class CustomTrOCR(nn.Module):
    """
    Full Custom TrOCR — handwritten text recognition from scratch.

    Pipeline:
      image (B,1,128,1024)
        → PatchEmbedding       → (B, 512, 256)
        → VisionTransformerEncoder → (B, 512, 256)
        → TransformerDecoder   → (B, T, vocab_size)

    Parameters: ~10M (much lighter than Microsoft's 334M)
    Tokenizer:  character-level (100 chars, no BPE needed)
    """
    def __init__(self, vocab_size: int, cfg=None):
        super().__init__()
        if cfg is None:
            from ocr.recognition.custom_trocr.config import cfg as _cfg
            cfg = _cfg.model

        self.patch_embed = PatchEmbedding(
            image_height=cfg.image_height,
            image_width=cfg.image_width,
            patch_size=cfg.patch_size,
            in_channels=1,
            embed_dim=cfg.embed_dim,
            dropout=cfg.dropout,
        )
        self.encoder = VisionTransformerEncoder(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_encoder_layers,
            ff_dim=cfg.ff_dim,
            dropout=cfg.dropout,
        )
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_decoder_layers,
            ff_dim=cfg.ff_dim,
            max_len=cfg.max_text_length,
            dropout=cfg.dropout,
        )

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.patch_embed(images))

    def forward(self, images, decoder_input_ids, tgt_key_padding_mask=None):
        memory = self.encode(images)
        return self.decoder(decoder_input_ids, memory, tgt_key_padding_mask)

    def count_parameters(self):
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n
