"""
CRNN + CTC Training Script
Usage:
    python train_crnn.py --images data/raw/images --labels data/raw/labels
    python train_crnn.py --images data/raw/images --labels data/raw/labels --epochs 30 --lr 1e-3
    python train_crnn.py --images data/raw/images --labels data/raw/labels --overlap-val-dir data/overlap_samples

Overlap-aware validation:
    Pass --overlap-val-dir with a folder of specifically challenging
    overlapping/cursive samples to track CER on the hard subset separately.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from config.settings import cfg
from ocr.recognition.crnn.model import CRNN, VOCAB_SIZE, encode_text, decode_ctc
from ocr.augmentation.augmentor import augment_line
from ocr.evaluation.metrics import compute_cer, compute_wer
from ocr.utils.logger import get_logger

log = get_logger("train_crnn")
C = cfg.crnn
T = cfg.training
P = cfg.paths


# ── Dataset ───────────────────────────────────────────────────────────────────

class LineImageDataset(Dataset):
    """
    Loads (image, text) pairs from:
      images_dir/*.{jpg,png}  ↔  labels_dir/*.txt  (matching stems)
    Applies augmentation during training.
    """

    def __init__(self, images_dir: str, labels_dir: str, augment: bool = False):
        self._images = sorted(Path(images_dir).glob("*.jpg")) + \
                       sorted(Path(images_dir).glob("*.png"))
        self._labels_dir = Path(labels_dir)
        self._augment = augment
        log.info("LineImageDataset: %d samples (augment=%s)", len(self._images), augment)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img_path = self._images[idx]
        label_path = self._labels_dir / (img_path.stem + ".txt")
        text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.full((C.input_height, 128), 255, dtype=np.uint8)

        if self._augment:
            img = augment_line(img)

        # Resize to fixed height, keep aspect ratio
        h, w = img.shape
        scale = C.input_height / h
        new_w = max(1, int(w * scale))
        img = cv2.resize(img, (new_w, C.input_height), interpolation=cv2.INTER_LINEAR)

        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.unsqueeze(0)  # (1, H, W)

        label_indices = encode_text(text)
        return tensor, label_indices, text


def _collate(batch):
    """Pad images to same width; stack labels for CTCLoss."""
    images, labels, texts = zip(*batch)
    max_w = max(img.shape[2] for img in images)
    padded = torch.full((len(images), 1, C.input_height, max_w), 1.0)
    for i, img in enumerate(images):
        padded[i, :, :, :img.shape[2]] = img

    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    flat_labels = torch.tensor([idx for l in labels for idx in l], dtype=torch.long)
    return padded, flat_labels, label_lengths, texts


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def evaluate_crnn(model, loader, device) -> tuple[float, float]:
    model.eval()
    preds, refs = [], []
    for images, _, _, texts in loader:
        images = images.to(device)
        logits = model(images)                          # (T, B, vocab)
        best_idx = logits.argmax(dim=-1).permute(1, 0)  # (B, T)
        for i, indices in enumerate(best_idx):
            pred = decode_ctc(indices.cpu().tolist())
            preds.append(pred)
            refs.append(texts[i])
    return compute_cer(preds, refs), compute_wer(preds, refs)


# ── Training Loop ─────────────────────────────────────────────────────────────

def train():
    args = _parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # Datasets
    full_ds = LineImageDataset(args.images, args.labels, augment=True)
    val_size = max(1, int(len(full_ds) * 0.1))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    # Disable augmentation on val split
    val_ds.dataset._augment = False

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=_collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=_collate, num_workers=0)

    # Overlap-specific validation set (hard subset)
    overlap_loader = None
    if args.overlap_val_dir and Path(args.overlap_val_dir).exists():
        overlap_ds = LineImageDataset(args.overlap_val_dir, args.labels, augment=False)
        overlap_loader = DataLoader(overlap_ds, batch_size=args.batch,
                                    collate_fn=_collate, num_workers=0)
        log.info("Overlap validation set: %d samples", len(overlap_ds))

    model = CRNN(vocab_size=VOCAB_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=T.weight_decay)
    ctc_loss = nn.CTCLoss(blank=C.ctc_blank_idx, reduction="mean", zero_infinity=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    P.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_cer = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for step, (images, flat_labels, label_lengths, _) in enumerate(train_loader, 1):
            images = images.to(device)
            flat_labels = flat_labels.to(device)

            logits = model(images)                      # (T, B, vocab)
            T_len = logits.size(0)
            B = logits.size(1)
            input_lengths = torch.full((B,), T_len, dtype=torch.long)

            loss = ctc_loss(
                logits.log_softmax(dim=-1),
                flat_labels,
                input_lengths,
                label_lengths,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

            if step % 50 == 0:
                log.info("Epoch %d | Step %d/%d | Loss %.4f",
                         epoch, step, len(train_loader), loss.item())

        cer, wer = evaluate_crnn(model, val_loader, device)
        elapsed = time.perf_counter() - t0
        scheduler.step(cer)
        log.info("Epoch %d | Loss %.4f | CER %.4f | WER %.4f | %.1fs",
                 epoch, epoch_loss / len(train_loader), cer, wer, elapsed)

        if overlap_loader:
            o_cer, o_wer = evaluate_crnn(model, overlap_loader, device)
            log.info("  Overlap-subset CER %.4f | WER %.4f", o_cer, o_wer)

        # Checkpoint
        ckpt = P.checkpoints_dir / f"crnn_epoch{epoch:02d}_cer{cer:.4f}.pth"
        torch.save(model.state_dict(), ckpt)

        if cer < best_cer:
            best_cer = cer
            patience_counter = 0
            torch.save(model.state_dict(), P.checkpoints_dir / "crnn_best.pth")
            log.info("New best CRNN saved (CER=%.4f)", best_cer)
        else:
            patience_counter += 1
            if patience_counter >= T.early_stopping_patience:
                log.info("Early stopping at epoch %d", epoch)
                break

    log.info("CRNN training complete. Best CER: %.4f", best_cer)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train CRNN+CTC for cursive HTR")
    parser.add_argument("--images",          required=True)
    parser.add_argument("--labels",          required=True)
    parser.add_argument("--overlap-val-dir", default=None,
                        help="Folder of hard overlapping samples for separate CER tracking")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--batch",   type=int,   default=16)
    return parser.parse_args()


if __name__ == "__main__":
    train()
