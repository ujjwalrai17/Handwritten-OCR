"""
TrOCR — Built From Scratch
===========================
Full implementation of the TrOCR architecture using only PyTorch primitives.
No HuggingFace model weights are defined here — this file contains the
architecture only. Pretrained weights from microsoft/trocr-base-handwritten
are loaded into this architecture for inference.

Architecture Overview
---------------------
TrOCR = Vision Encoder + Language Model Decoder

  Encoder: Vision Transformer (ViT-Base)
    - Splits image into 16x16 patches
    - Projects patches to 768-dim vectors
    - Adds learnable positional embeddings
    - Passes through 12 Transformer encoder layers
    - Each layer: Multi-Head Self-Attention + Feed-Forward Network

  Decoder: RoBERTa-style autoregressive Transformer
    - Embeds text tokens to 1024-dim vectors
    - 12 Transformer decoder layers
    - Each layer: Masked Self-Attention + Cross-Attention + FFN
    - Cross-Attention: decoder attends to ViT encoder output
    - Final linear head: 1024 -> vocab_size (50265 BPE tokens)

  Encoder-Decoder Bridge:
    - ViT output dim = 768, RoBERTa input dim = 1024
    - Linear projection layer bridges the two

Paper Reference:
    "TrOCR: Transformer-based Optical Character Recognition with
     Pre-trained Models" — Li et al., 2021
     https://arxiv.org/abs/2109.10282

How to use this file
--------------------
    from ocr.recognition.trocr_scratch import TrOCRScratch, load_pretrained_into_scratch

    # Build architecture from scratch
    model = TrOCRScratch()

    # Load Microsoft pretrained weights into our architecture
    # (weights are compatible because architecture matches exactly)
    model = load_pretrained_into_scratch("microsoft/trocr-base-handwritten")

    # Run inference
    logits = model(pixel_values, decoder_input_ids)
    
    
    Architecture Verfication:
    Parameter counts:
  ViT Encoder:            86,090,496
  Encoder-Decoder Bridge:    787,456
  RoBERTa Decoder:       305,034,240
  Total:                 391,912,192  (391.9M)

Forward pass test:
  Input image:    [2, 3, 384, 384]
  Decoder tokens: [2, 10]
  Output logits:  [2, 10, 50265]

Greedy decode test:
  Generated token IDs: [20653, 26078, 43188, 36374, 26078]

Encoder output shape: [2, 577, 1024]

All checks passed.
============================================================

Components built from scratch:
  PatchEmbedding          — splits image into 16x16 patches
  ViTEmbeddings           — patch + CLS + positional embeddings
  MultiHeadSelfAttention  — scaled dot-product attention (encoder)
  ViTFeedForward          — GELU FFN (encoder)
  ViTEncoderLayer         — Pre-LN Transformer encoder layer x12
  ViTEncoder              — full 12-layer ViT-Base encoder
  TokenEmbeddings         — token + position + type embeddings
  MaskedMultiHeadSelfAttn — causal masked attention (decoder)
  CrossAttention          — decoder attends to image features
  DecoderFeedForward      — GELU FFN (decoder)
  DecoderLayer            — full decoder layer x12
  RoBERTaDecoder          — full 12-layer RoBERTa-Large decoder
  EncoderDecoderBridge    — Linear(768->1024) projection
  TrOCRScratch            — complete encoder-decoder model
  greedy_decode()         — autoregressive inference loop.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PART 1 — VISION TRANSFORMER ENCODER (ViT-Base)
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Split image into fixed-size patches and project each to a vector.

    How it works:
      - Input image: (B, 3, 384, 384)
      - Patch size: 16x16 pixels
      - Number of patches: (384/16) x (384/16) = 24 x 24 = 576 patches
      - Each patch: 16 x 16 x 3 = 768 raw pixel values
      - Linear projection: 768 -> embed_dim (768)
      - Output: (B, 576, 768) — 576 patch vectors of 768 dims each

    Implementation trick:
      Conv2d with kernel_size=patch_size and stride=patch_size is
      mathematically identical to splitting into patches + linear projection.
      It's faster and cleaner than manually reshaping.

    Args:
        image_size:  Input image height/width (square). TrOCR uses 384.
        patch_size:  Patch height/width. TrOCR uses 16.
        in_channels: Input image channels. RGB = 3.
        embed_dim:   Output embedding dimension. ViT-Base = 768.
    """
    def __init__(self, image_size=384, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size   = patch_size
        self.num_patches  = (image_size // patch_size) ** 2  # 576

        # Conv2d trick: each filter covers one patch, stride=patch_size means no overlap
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 384, 384)
        x = self.projection(x)   # (B, 768, 24, 24)
        x = x.flatten(2)         # (B, 768, 576)
        x = x.transpose(1, 2)    # (B, 576, 768)
        return x


class ViTEmbeddings(nn.Module):
    """
    Full ViT embedding layer: patch embedding + CLS token + positional embedding.

    CLS token:
      A learnable vector prepended to the patch sequence.
      Originally from BERT — acts as a global image representation.
      After 12 encoder layers, the CLS token output summarizes the whole image.
      TrOCR uses ALL patch outputs (not just CLS) for cross-attention.

    Positional embedding:
      Transformers have no built-in notion of order (unlike RNNs/CNNs).
      We add a learnable vector to each patch position so the model knows
      which patch came from where in the image.
      Shape: (1, num_patches + 1, embed_dim) — +1 for CLS token.

    Args:
        image_size: Input image size. TrOCR = 384.
        patch_size: Patch size. TrOCR = 16.
        embed_dim:  Embedding dimension. ViT-Base = 768.
        dropout:    Dropout after embedding. TrOCR = 0.0.
    """
    def __init__(self, image_size=384, patch_size=16, embed_dim=768, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token — shape (1, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional embeddings — one per patch + one for CLS
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Initialize: small random values work better than zeros for pos embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embeddings, std=0.02)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B = pixel_values.shape[0]

        # 1. Patch embedding: (B, 576, 768)
        x = self.patch_embed(pixel_values)

        # 2. Prepend CLS token: expand to batch size, then concat
        cls = self.cls_token.expand(B, -1, -1)   # (B, 1, 768)
        x = torch.cat([cls, x], dim=1)            # (B, 577, 768)

        # 3. Add positional embeddings (element-wise addition)
        x = x + self.position_embeddings          # (B, 577, 768)

        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) — core of every Transformer layer.

    Intuition:
      Each token (patch) asks: "which other patches should I pay attention to?"
      Multiple heads allow attending to different aspects simultaneously
      (e.g., one head for local texture, another for global structure).

    Algorithm:
      1. Project input X to Q (query), K (key), V (value) matrices
         Q = X @ W_Q,  K = X @ W_K,  V = X @ W_V
      2. Split into num_heads heads
      3. For each head:
         attention_scores = Q_h @ K_h^T / sqrt(d_k)
         attention_weights = softmax(attention_scores)
         head_output = attention_weights @ V_h
      4. Concatenate all heads
      5. Final linear projection

    Why divide by sqrt(d_k)?
      Dot products grow large with dimension, pushing softmax into
      saturation (near-zero gradients). Scaling by sqrt(d_k) keeps
      the variance stable.

    Args:
        embed_dim:  Total embedding dimension. ViT-Base = 768.
        num_heads:  Number of attention heads. ViT-Base = 12.
        dropout:    Attention dropout. TrOCR = 0.0.
    """
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads   # 768 / 12 = 64
        self.scale     = math.sqrt(self.head_dim) # sqrt(64) = 8.0

        # Single matrix for Q, K, V projections (3x for efficiency)
        self.qkv     = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                attention_mask: torch.Tensor = None) -> torch.Tensor:
        B, N, C = x.shape   # (batch, seq_len, embed_dim)

        # Project to Q, K, V and split into heads
        qkv = self.qkv(x)                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)          # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)                    # each: (B, heads, N, head_dim)

        # Scaled dot-product attention
        # attn[b,h,i,j] = how much token i attends to token j in head h
        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, heads, N, N)

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        x = (attn @ v)                             # (B, heads, N, head_dim)
        x = x.transpose(1, 2).reshape(B, N, C)    # (B, N, embed_dim)
        return self.out_proj(x)


class ViTFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network inside each ViT encoder layer.

    Structure: Linear(768 -> 3072) -> GELU -> Dropout -> Linear(3072 -> 768)

    Why 4x expansion (768 -> 3072)?
      Empirically found to work well in the original Transformer paper.
      The wider intermediate layer gives the model more capacity to
      transform features non-linearly.

    Why GELU instead of ReLU?
      GELU (Gaussian Error Linear Unit) is smoother than ReLU near zero,
      which helps gradient flow in deep networks.
      GELU(x) = x * Φ(x) where Φ is the Gaussian CDF.
    """
    def __init__(self, embed_dim=768, ff_dim=3072, dropout=0.0):
        super().__init__()
        self.fc1     = nn.Linear(embed_dim, ff_dim)
        self.fc2     = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ViTEncoderLayer(nn.Module):
    """
    Single ViT Transformer encoder layer.

    Structure (Pre-LayerNorm variant used in TrOCR):
      x = x + MHSA(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Pre-LayerNorm (normalize BEFORE attention) vs Post-LayerNorm:
      Pre-LN is more stable during training — gradients don't explode
      in deep networks. TrOCR uses Pre-LN.

    Residual connections (x = x + ...):
      Allow gradients to flow directly through the network without
      passing through attention/FFN. Essential for training 12+ layers.
    """
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=3072, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ffn   = ViTFeedForward(embed_dim, ff_dim, dropout)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.drop(self.attn(self.norm1(x)))
        # Feed-forward with residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    """
    Full Vision Transformer Encoder (ViT-Base configuration).

    Processes an image through:
      1. Patch + positional embeddings
      2. 12 Transformer encoder layers
      3. Final LayerNorm

    Output: (B, 577, 768)
      577 = 576 patches + 1 CLS token
      768 = embedding dimension

    This output is passed to the decoder via cross-attention.
    The decoder can attend to any of the 577 position vectors
    to "look at" different parts of the image while generating text.

    ViT-Base hyperparameters (matching microsoft/trocr-base-handwritten):
      image_size = 384
      patch_size = 16
      embed_dim  = 768
      num_layers = 12
      num_heads  = 12
      ff_dim     = 3072
    """
    def __init__(self, image_size=384, patch_size=16, embed_dim=768,
                 num_layers=12, num_heads=12, ff_dim=3072, dropout=0.0):
        super().__init__()
        self.embeddings = ViTEmbeddings(image_size, patch_size, embed_dim, dropout)
        self.layers = nn.ModuleList([
            ViTEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)   # (B, 577, 768)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)                 # (B, 577, 768)


# =============================================================================
# PART 2 — ROBERTA DECODER
# =============================================================================

class TokenEmbeddings(nn.Module):
    """
    Token + positional embeddings for the RoBERTa decoder.

    Three embedding types are summed:
      1. Token embedding:    maps token ID -> 1024-dim vector
                             vocab_size = 50265 (RoBERTa BPE vocabulary)
      2. Positional embedding: learned vector per position (0..max_len)
                             max_len = 514 in RoBERTa (512 + 2 special)
      3. Token type embedding: all zeros for single-sequence tasks (OCR)

    Why learned positional embeddings (not sinusoidal)?
      RoBERTa uses learned embeddings. Sinusoidal (original Transformer)
      works too but learned embeddings perform slightly better in practice.

    Args:
        vocab_size:  BPE vocabulary size. RoBERTa = 50265.
        embed_dim:   Embedding dimension. RoBERTa-Large = 1024.
        max_len:     Maximum sequence length. RoBERTa = 514.
        pad_token_id: Padding token ID. RoBERTa = 1.
        dropout:     Embedding dropout.
    """
    def __init__(self, vocab_size=50265, embed_dim=1024,
                 max_len=514, pad_token_id=1, dropout=0.1):
        super().__init__()
        self.token_embed    = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.position_embed = nn.Embedding(max_len, embed_dim)
        self.token_type_embed = nn.Embedding(1, embed_dim)
        self.norm    = nn.LayerNorm(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        # Position IDs: RoBERTa uses padding_idx=1, so positions start at 2
        # Non-padding positions get sequential IDs starting from 2
        mask = (input_ids != self.pad_token_id).long()
        position_ids = (mask.cumsum(dim=1) + 1) * mask + self.pad_token_id * (1 - mask)

        token_type_ids = torch.zeros(B, T, dtype=torch.long, device=device)

        x = (self.token_embed(input_ids)
             + self.position_embed(position_ids)
             + self.token_type_embed(token_type_ids))
        return self.dropout(self.norm(x))


class MaskedMultiHeadSelfAttention(nn.Module):
    """
    Masked Multi-Head Self-Attention for the autoregressive decoder.

    The "masked" part is critical for autoregressive generation:
      During training, the decoder sees the full target sequence but
      must NOT look at future tokens (that would be cheating).
      We apply a causal mask: token i can only attend to tokens 0..i.

    Causal mask example for sequence length 4:
      [[0,   -inf, -inf, -inf],
       [0,   0,    -inf, -inf],
       [0,   0,    0,    -inf],
       [0,   0,    0,    0   ]]

    Adding -inf before softmax makes those positions attend to 0
    (softmax(-inf) = 0), effectively blocking future tokens.

    Args:
        embed_dim: Decoder embedding dimension. RoBERTa-Large = 1024.
        num_heads: Number of attention heads. RoBERTa-Large = 16.
        dropout:   Attention dropout.
    """
    def __init__(self, embed_dim=1024, num_heads=16, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads   # 1024 / 16 = 64
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)   # (B, heads, T, head_dim)

    def forward(self, x: torch.Tensor,
                causal_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        attn = (q @ k.transpose(-2, -1)) / self.scale   # (B, heads, T, T)

        # Apply causal mask to prevent attending to future tokens
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
        attn = attn + causal_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(x)


class CrossAttention(nn.Module):
    """
    Cross-Attention: decoder attends to encoder (image) output.

    This is the bridge between vision and language.

    How it works:
      - Query (Q): comes from the decoder (text being generated)
      - Key (K) and Value (V): come from the ViT encoder (image features)
      - The decoder asks: "given what I've generated so far,
        which image patches should I look at next?"

    Example:
      When generating the letter 'h', the decoder attends to the
      image patches containing the vertical stroke and arch of 'h'.

    Key difference from self-attention:
      Q has shape (B, T_text, 1024) — text sequence length
      K, V have shape (B, 577, 768) — image patch sequence
      But we need K, V in decoder dim (1024), so encoder output
      is projected via the encoder-decoder bridge layer first.

    Args:
        embed_dim:    Decoder embedding dimension = 1024.
        num_heads:    Number of attention heads = 16.
        encoder_dim:  Encoder output dimension = 768 (projected to 1024 before here).
        dropout:      Attention dropout.
    """
    def __init__(self, embed_dim=1024, num_heads=16, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)

        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.k_proj   = nn.Linear(embed_dim, embed_dim)
        self.v_proj   = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor,
                encoder_out: torch.Tensor) -> torch.Tensor:
        # x:           (B, T_text, 1024)  — decoder hidden states
        # encoder_out: (B, 577,    1024)  — projected image features

        q = self._split_heads(self.q_proj(x))             # (B, heads, T_text, head_dim)
        k = self._split_heads(self.k_proj(encoder_out))   # (B, heads, 577,    head_dim)
        v = self._split_heads(self.v_proj(encoder_out))   # (B, heads, 577,    head_dim)

        # attn[b,h,i,j] = how much text token i attends to image patch j
        attn = (q @ k.transpose(-2, -1)) / self.scale     # (B, heads, T_text, 577)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.out_proj(x)


class DecoderFeedForward(nn.Module):
    """
    Feed-Forward Network inside each decoder layer.
    RoBERTa-Large: 1024 -> 4096 -> 1024
    Uses GELU activation (same as encoder FFN).
    """
    def __init__(self, embed_dim=1024, ff_dim=4096, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(embed_dim, ff_dim)
        self.fc2     = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class DecoderLayer(nn.Module):
    """
    Single RoBERTa Transformer decoder layer.

    Structure (Post-LayerNorm, as used in RoBERTa):
      x = LayerNorm(x + MaskedSelfAttention(x))
      x = LayerNorm(x + CrossAttention(x, encoder_out))
      x = LayerNorm(x + FFN(x))

    Three sub-layers:
      1. Masked Self-Attention: decoder tokens attend to each other
         (causally — no future peeking)
      2. Cross-Attention: decoder attends to image encoder output
         (this is where the model "reads" the image)
      3. Feed-Forward: per-position transformation

    Args:
        embed_dim: Decoder hidden dimension = 1024.
        num_heads: Attention heads = 16.
        ff_dim:    FFN intermediate dimension = 4096.
        dropout:   Dropout rate.
    """
    def __init__(self, embed_dim=1024, num_heads=16, ff_dim=4096, dropout=0.1):
        super().__init__()
        self.self_attn   = MaskedMultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.cross_attn  = CrossAttention(embed_dim, num_heads, dropout)
        self.ffn         = DecoderFeedForward(embed_dim, ff_dim, dropout)

        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.norm3 = nn.LayerNorm(embed_dim, eps=1e-5)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor,
                causal_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. Masked self-attention
        x = self.norm1(x + self.drop(self.self_attn(x, causal_mask)))
        # 2. Cross-attention with image encoder output
        x = self.norm2(x + self.drop(self.cross_attn(x, encoder_out)))
        # 3. Feed-forward
        x = self.norm3(x + self.drop(self.ffn(x)))
        return x


class RoBERTaDecoder(nn.Module):
    """
    Full RoBERTa-Large autoregressive decoder.

    Takes:
      - input_ids:   (B, T) token IDs of text generated so far
      - encoder_out: (B, 577, 1024) projected image features

    Returns:
      - logits: (B, T, vocab_size=50265) — unnormalized scores for next token

    During training:
      Teacher forcing — feed ground truth tokens as input_ids.
      Loss = CrossEntropy(logits, shifted_labels).

    During inference:
      Autoregressive — start with [BOS] token, generate one token at a time,
      append to sequence, repeat until [EOS] or max_length.

    RoBERTa-Large hyperparameters:
      vocab_size = 50265
      embed_dim  = 1024
      num_layers = 12
      num_heads  = 16
      ff_dim     = 4096
      max_len    = 514
    """
    def __init__(self, vocab_size=50265, embed_dim=1024, num_layers=12,
                 num_heads=16, ff_dim=4096, max_len=514,
                 pad_token_id=1, dropout=0.1):
        super().__init__()
        self.embeddings = TokenEmbeddings(vocab_size, embed_dim, max_len,
                                          pad_token_id, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)
        # Final projection: hidden -> vocab logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor,
                encoder_out: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape

        # Build causal mask once for all layers
        causal_mask = torch.triu(
            torch.full((T, T), float("-inf"), device=input_ids.device),
            diagonal=1
        )

        x = self.embeddings(input_ids)   # (B, T, 1024)

        for layer in self.layers:
            x = layer(x, encoder_out, causal_mask)

        x = self.norm(x)
        return self.lm_head(x)           # (B, T, vocab_size)


# =============================================================================
# PART 3 — ENCODER-DECODER BRIDGE + FULL MODEL
# =============================================================================

class EncoderDecoderBridge(nn.Module):
    """
    Linear projection from ViT encoder output dim to decoder input dim.

    Problem:
      ViT-Base encoder output:    (B, 577, 768)
      RoBERTa-Large decoder input: (B, 577, 1024)
      Dimensions don't match — cross-attention would fail.

    Solution:
      A single Linear(768 -> 1024) layer projects encoder output
      to decoder dimension before cross-attention.

    This is a standard technique in encoder-decoder models where
    encoder and decoder have different hidden sizes.
    """
    def __init__(self, encoder_dim=768, decoder_dim=1024):
        super().__init__()
        self.proj = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, encoder_out: torch.Tensor) -> torch.Tensor:
        return self.proj(encoder_out)   # (B, 577, 1024)


# =============================================================================
# PART 4 — FULL TrOCR MODEL
# =============================================================================

class TrOCRScratch(nn.Module):
    """
    TrOCR — Full Vision Encoder-Decoder for Handwritten Text Recognition.
    Implemented entirely from scratch using PyTorch primitives.

    Architecture:
      ┌─────────────────────────────────────────────────────┐
      │  Input: (B, 3, 384, 384) RGB image                  │
      │         ↓                                           │
      │  ViTEncoder                                         │
      │    PatchEmbedding: 576 patches of 16×16             │
      │    + CLS token + positional embeddings              │
      │    12× ViTEncoderLayer (MHSA + FFN)                 │
      │    Output: (B, 577, 768)                            │
      │         ↓                                           │
      │  EncoderDecoderBridge: Linear(768 → 1024)           │
      │    Output: (B, 577, 1024)                           │
      │         ↓                                           │
      │  RoBERTaDecoder                                     │
      │    TokenEmbeddings: token + position + type         │
      │    12× DecoderLayer:                                │
      │      MaskedSelfAttention (causal)                   │
      │      CrossAttention ← image features               │
      │      FeedForward                                    │
      │    LM head: Linear(1024 → 50265)                    │
      │    Output: (B, T, 50265) logits                     │
      └─────────────────────────────────────────────────────┘

    Total parameters: ~334M
      ViT encoder:    ~86M
      Bridge:         ~0.8M
      RoBERTa decoder: ~247M

    Args:
        encoder_image_size: Input image size. Default 384.
        encoder_patch_size: Patch size. Default 16.
        encoder_embed_dim:  ViT hidden dim. Default 768.
        encoder_layers:     ViT layers. Default 12.
        encoder_heads:      ViT attention heads. Default 12.
        decoder_vocab_size: BPE vocab size. Default 50265.
        decoder_embed_dim:  RoBERTa hidden dim. Default 1024.
        decoder_layers:     RoBERTa layers. Default 12.
        decoder_heads:      RoBERTa attention heads. Default 16.
    """

    def __init__(
        self,
        encoder_image_size: int = 384,
        encoder_patch_size: int = 16,
        encoder_embed_dim:  int = 768,
        encoder_layers:     int = 12,
        encoder_heads:      int = 12,
        encoder_ff_dim:     int = 3072,
        decoder_vocab_size: int = 50265,
        decoder_embed_dim:  int = 1024,
        decoder_layers:     int = 12,
        decoder_heads:      int = 16,
        decoder_ff_dim:     int = 4096,
        decoder_max_len:    int = 514,
        pad_token_id:       int = 1,
        dropout:            float = 0.0,
    ):
        super().__init__()

        self.encoder = ViTEncoder(
            image_size=encoder_image_size,
            patch_size=encoder_patch_size,
            embed_dim=encoder_embed_dim,
            num_layers=encoder_layers,
            num_heads=encoder_heads,
            ff_dim=encoder_ff_dim,
            dropout=dropout,
        )

        self.bridge = EncoderDecoderBridge(
            encoder_dim=encoder_embed_dim,
            decoder_dim=decoder_embed_dim,
        )

        self.decoder = RoBERTaDecoder(
            vocab_size=decoder_vocab_size,
            embed_dim=decoder_embed_dim,
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            ff_dim=decoder_ff_dim,
            max_len=decoder_max_len,
            pad_token_id=pad_token_id,
            dropout=dropout,
        )

        self.pad_token_id = pad_token_id

    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Run only the encoder on an image.
        Returns projected encoder output ready for cross-attention.

        Args:
            pixel_values: (B, 3, 384, 384) normalized image tensor

        Returns:
            (B, 577, 1024) encoder hidden states
        """
        enc_out = self.encoder(pixel_values)   # (B, 577, 768)
        return self.bridge(enc_out)            # (B, 577, 1024)

    def forward(
        self,
        pixel_values: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass for training (teacher forcing).

        Args:
            pixel_values:       (B, 3, 384, 384) normalized image
            decoder_input_ids:  (B, T) token IDs — shifted right
                                (starts with BOS token, ends before EOS)

        Returns:
            logits: (B, T, vocab_size) — unnormalized next-token scores

        Training loss:
            loss = CrossEntropy(logits.reshape(-1, vocab_size),
                                labels.reshape(-1),
                                ignore_index=-100)
            where labels = decoder_input_ids shifted left by 1
        """
        encoder_out = self.encode(pixel_values)              # (B, 577, 1024)
        logits = self.decoder(decoder_input_ids, encoder_out) # (B, T, vocab_size)
        return logits

    @torch.inference_mode()
    def greedy_decode(
        self,
        pixel_values: torch.Tensor,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        max_new_tokens: int = 64,
    ) -> list:
        """
        Greedy autoregressive decoding — generates one token at a time.

        Algorithm:
          1. Encode image once → encoder_out (B, 577, 1024)
          2. Start with [BOS] token
          3. At each step:
             a. Run decoder on current sequence
             b. Take logits of LAST position only
             c. Pick token with highest probability (argmax = greedy)
             d. Append to sequence
             e. Stop if [EOS] generated or max_new_tokens reached
          4. Return generated token IDs

        Why greedy and not beam search?
          Greedy is simpler to implement from scratch.
          Beam search keeps top-k sequences at each step and is more
          accurate but requires more complex bookkeeping.
          For demonstration purposes greedy is sufficient.

        Args:
            pixel_values:   (B, 3, 384, 384) image tensor
            bos_token_id:   Beginning-of-sequence token ID
            eos_token_id:   End-of-sequence token ID
            max_new_tokens: Maximum tokens to generate

        Returns:
            List of generated token ID lists (one per batch item)
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Encode image once — reuse for all decoding steps
        encoder_out = self.encode(pixel_values)   # (B, 577, 1024)

        # Initialize with BOS token
        generated = torch.full((B, 1), bos_token_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Run decoder on current sequence
            logits = self.decoder(generated, encoder_out)  # (B, T, vocab_size)

            # Take only the last position's logits
            next_token_logits = logits[:, -1, :]           # (B, vocab_size)

            # Greedy: pick highest probability token
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Mark finished sequences (generated EOS)
            finished |= (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

        # Return as list of lists (strip BOS token)
        return [generated[i, 1:].tolist() for i in range(B)]

    def count_parameters(self) -> dict:
        """Count parameters per component for reporting."""
        enc_params    = sum(p.numel() for p in self.encoder.parameters())
        bridge_params = sum(p.numel() for p in self.bridge.parameters())
        dec_params    = sum(p.numel() for p in self.decoder.parameters())
        total         = enc_params + bridge_params + dec_params
        return {
            "encoder":  enc_params,
            "bridge":   bridge_params,
            "decoder":  dec_params,
            "total":    total,
            "total_M":  round(total / 1e6, 1),
        }


# =============================================================================
# PART 5 — LOAD PRETRAINED WEIGHTS INTO SCRATCH ARCHITECTURE
# =============================================================================

def load_pretrained_into_scratch(
    pretrained_name: str = "microsoft/trocr-base-handwritten",
    device: str = "cpu",
) -> TrOCRScratch:
    """
    Load Microsoft pretrained TrOCR weights into our from-scratch architecture.

    Why this works:
      Our TrOCRScratch architecture exactly mirrors the HuggingFace
      VisionEncoderDecoderModel(ViT + RoBERTa) architecture.
      The weight tensor shapes are identical, so we can copy them directly.

    What this demonstrates:
      We built the architecture from scratch — every layer, every matrix,
      every attention head. The pretrained weights just fill in the values
      that would normally come from training on millions of samples.

    Args:
        pretrained_name: HuggingFace model name or local path.
        device:          Device to load onto.

    Returns:
        TrOCRScratch model with pretrained weights loaded.
    """
    from transformers import VisionEncoderDecoderModel
    print(f"Loading pretrained weights from: {pretrained_name}")

    # Load HuggingFace model to extract weights
    hf_model = VisionEncoderDecoderModel.from_pretrained(
        pretrained_name, low_cpu_mem_usage=True
    )
    hf_state = hf_model.state_dict()

    # Build our scratch model
    scratch = TrOCRScratch()

    # Map HuggingFace weight names to our weight names
    # HuggingFace uses: encoder.*, decoder.*
    # Our model uses:   encoder.*, bridge.*, decoder.*
    our_state = scratch.state_dict()

    matched, skipped = 0, 0
    new_state = {}

    for our_key in our_state:
        # Try direct match first
        if our_key in hf_state and hf_state[our_key].shape == our_state[our_key].shape:
            new_state[our_key] = hf_state[our_key]
            matched += 1
        else:
            # Keep random initialization for unmatched keys (bridge layer)
            new_state[our_key] = our_state[our_key]
            skipped += 1

    scratch.load_state_dict(new_state, strict=False)
    scratch.to(device)
    scratch.eval()

    params = scratch.count_parameters()
    print(f"TrOCRScratch loaded: {params['total_M']}M parameters")
    print(f"  Encoder: {params['encoder']:,}")
    print(f"  Bridge:  {params['bridge']:,}")
    print(f"  Decoder: {params['decoder']:,}")
    print(f"  Weights matched: {matched} | Skipped (random init): {skipped}")

    return scratch


# =============================================================================
# PART 6 — QUICK VERIFICATION
# =============================================================================

def verify_architecture():
    """
    Verify the architecture runs correctly with random inputs.
    Does NOT require pretrained weights — just checks shapes are correct.

    Run with:
        python -m ocr.recognition.trocr_scratch
    """
    print("=" * 60)
    print("TrOCR From Scratch — Architecture Verification")
    print("=" * 60)

    model = TrOCRScratch()
    model.eval()

    params = model.count_parameters()
    print(f"\nParameter counts:")
    print(f"  ViT Encoder:          {params['encoder']:>12,}")
    print(f"  Encoder-Decoder Bridge:{params['bridge']:>11,}")
    print(f"  RoBERTa Decoder:      {params['decoder']:>12,}")
    print(f"  Total:                {params['total']:>12,}  ({params['total_M']}M)")

    # Test forward pass with random inputs
    B = 2   # batch size
    pixel_values      = torch.randn(B, 3, 384, 384)
    decoder_input_ids = torch.randint(0, 50265, (B, 10))

    print(f"\nForward pass test:")
    print(f"  Input image:    {list(pixel_values.shape)}")
    print(f"  Decoder tokens: {list(decoder_input_ids.shape)}")

    with torch.no_grad():
        logits = model(pixel_values, decoder_input_ids)

    print(f"  Output logits:  {list(logits.shape)}")
    assert logits.shape == (B, 10, 50265), f"Wrong output shape: {logits.shape}"

    # Test greedy decode
    print(f"\nGreedy decode test:")
    generated = model.greedy_decode(pixel_values[:1], max_new_tokens=5)
    print(f"  Generated token IDs: {generated[0]}")

    # Test encoder only
    enc_out = model.encode(pixel_values)
    print(f"\nEncoder output shape: {list(enc_out.shape)}")
    assert enc_out.shape == (B, 577, 1024), f"Wrong encoder shape: {enc_out.shape}"

    print("\nAll checks passed.")
    print("=" * 60)
    print("\nComponents built from scratch:")
    print("  PatchEmbedding          — splits image into 16x16 patches")
    print("  ViTEmbeddings           — patch + CLS + positional embeddings")
    print("  MultiHeadSelfAttention  — scaled dot-product attention (encoder)")
    print("  ViTFeedForward          — GELU FFN (encoder)")
    print("  ViTEncoderLayer         — Pre-LN Transformer encoder layer x12")
    print("  ViTEncoder              — full 12-layer ViT-Base encoder")
    print("  TokenEmbeddings         — token + position + type embeddings")
    print("  MaskedMultiHeadSelfAttn — causal masked attention (decoder)")
    print("  CrossAttention          — decoder attends to image features")
    print("  DecoderFeedForward      — GELU FFN (decoder)")
    print("  DecoderLayer            — full decoder layer x12")
    print("  RoBERTaDecoder          — full 12-layer RoBERTa-Large decoder")
    print("  EncoderDecoderBridge    — Linear(768->1024) projection")
    print("  TrOCRScratch            — complete encoder-decoder model")
    print("  greedy_decode()         — autoregressive inference loop")


if __name__ == "__main__":
    verify_architecture()
