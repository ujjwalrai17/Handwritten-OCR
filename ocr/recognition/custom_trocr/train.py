"""
Training script for Custom TrOCR.

Uses IAM dataset from HuggingFace (Teklia/IAM-line).
Each sample: grayscale line image + ground truth text.

Loss: CrossEntropy with teacher forcing.
Saves best checkpoint to: ocr/recognition/custom_trocr/checkpoints/best.pt

Usage:
    python -m ocr.recognition.custom_trocr.train
    python -m ocr.recognition.custom_trocr.train --epochs 10 --batch 4
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np

from ocr.recognition.custom_trocr.config import cfg
from ocr.recognition.custom_trocr.model import CustomTrOCR
from ocr.recognition.custom_trocr.tokenizer import CharacterTokenizer, PAD_ID, BOS_ID, EOS_ID
from ocr.recognition.custom_trocr.image_processor import ImageProcessor
from ocr.utils.logger import get_logger

log = get_logger("custom_trocr.train")


# ── Dataset ───────────────────────────────────────────────────────────────────

class IAMLineDataset(Dataset):
    """Loads IAM line dataset from HuggingFace."""

    def __init__(self, split="train", tokenizer=None, processor=None, max_text_len=128):
        from datasets import load_dataset
        log.info("Loading IAM dataset (split=%s)...", split)
        self.ds        = load_dataset("Teklia/IAM-line", split=split, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_len   = max_text_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item  = self.ds[idx]
        image = item["image"]
        text  = item["text"]

        # Convert PIL to numpy BGR
        img_np = np.array(image.convert("RGB"))[:, :, ::-1]
        tensor = self.processor.process(img_np)   # (1, 128, 1024)

        # Encode text
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        ids = ids[:self.max_len]
        return tensor, ids, text


def collate_fn(batch):
    images, ids_list, texts = zip(*batch)
    images = torch.stack(images)   # (B, 1, H, W)

    max_len = max(len(ids) for ids in ids_list)
    padded  = torch.full((len(ids_list), max_len), PAD_ID, dtype=torch.long)
    pad_mask = torch.ones(len(ids_list), max_len, dtype=torch.bool)

    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        pad_mask[i, :len(ids)] = False   # False = not padding

    return images, padded, pad_mask, texts


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    tokenizer = CharacterTokenizer()
    processor = ImageProcessor(cfg.model.image_height, cfg.model.image_width)

    # Save tokenizer
    tok_path = cfg.paths.data_processed / "tokenizer.json"
    tok_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(tok_path)
    log.info("Tokenizer saved: %s (vocab=%d)", tok_path, tokenizer.vocab_size)

    # Dataset
    full_ds = IAMLineDataset("train", tokenizer, processor, cfg.model.max_text_length)
    n_val   = max(1, int(len(full_ds) * cfg.training.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    log.info("Train: %d | Val: %d", len(train_ds), len(val_ds))

    # Model
    model = CustomTrOCR(vocab_size=tokenizer.vocab_size).to(device)
    log.info("CustomTrOCR parameters: %d (%.1fM)",
             model.count_parameters(), model.count_parameters() / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=cfg.training.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    ckpt_dir = cfg.paths.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        t0 = time.perf_counter()

        for images, tgt, pad_mask, _ in train_loader:
            images   = images.to(device)
            tgt      = tgt.to(device)
            pad_mask = pad_mask.to(device)

            # Teacher forcing: input = tgt[:-1], label = tgt[1:]
            decoder_input = tgt[:, :-1]
            labels        = tgt[:, 1:]
            input_mask    = pad_mask[:, :-1]

            logits = model(images, decoder_input, input_mask)  # (B, T-1, V)
            loss   = criterion(logits.reshape(-1, tokenizer.vocab_size),
                               labels.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, tgt, pad_mask, _ in val_loader:
                images   = images.to(device)
                tgt      = tgt.to(device)
                pad_mask = pad_mask.to(device)
                decoder_input = tgt[:, :-1]
                labels        = tgt[:, 1:]
                input_mask    = pad_mask[:, :-1]
                logits = model(images, decoder_input, input_mask)
                val_losses.append(criterion(
                    logits.reshape(-1, tokenizer.vocab_size),
                    labels.reshape(-1)
                ).item())

        avg_train = sum(train_losses) / len(train_losses)
        avg_val   = sum(val_losses)   / len(val_losses)
        elapsed   = time.perf_counter() - t0

        log.info("Epoch %02d/%02d | train=%.4f | val=%.4f | %.1fs",
                 epoch, args.epochs, avg_train, avg_val, elapsed)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab_size": tokenizer.vocab_size,
                "val_loss": avg_val,
            }, ckpt_dir / "best.pt")
            log.info("  ✓ Best model saved (val_loss=%.4f)", avg_val)

    log.info("Training complete. Best val_loss=%.4f", best_val_loss)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int,   default=cfg.training.epochs)
    p.add_argument("--batch",  type=int,   default=cfg.training.batch_size)
    p.add_argument("--lr",     type=float, default=cfg.training.learning_rate)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
