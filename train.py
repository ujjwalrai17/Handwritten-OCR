"""
TrOCR Fine-Tuning Script
Usage:
    python train.py
    python train.py --epochs 5 --lr 3e-5 --batch 4
    python train.py --local --images data/raw/images --labels data/raw/labels
    python train.py --local --images data/raw/images --labels data/raw/labels \
        --hard-images data/hard_samples/images --hard-labels data/hard_samples/labels

Hard-sample strategy
--------------------
Pass --hard-images / --hard-labels pointing to a folder of 100-300 lines
that are specifically dense, overlapping, fast cursive. These are appended
to the training set cfg.training.hard_sample_oversample_ratio times so the
model sees them proportionally more often than easy IAM lines.

Focal loss
----------
FL(p_t) = -(1-p_t)^gamma * log(p_t)
gamma=0  -> standard cross-entropy
gamma=2  -> strong focus on uncertain tokens (recommended for overlapping strokes)
Disable with --no-focal to compare against baseline CE.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    RobertaTokenizer,
    ViTImageProcessor,
    get_linear_schedule_with_warmup,
)

from config.settings import cfg
from ocr.dataset.iam_dataset import IAMLineDataset, LocalImageDataset, build_combined_dataset
from ocr.evaluation.metrics import compute_cer, compute_wer
from ocr.utils.logger import get_logger

log = get_logger("train")
C = cfg.training
P = cfg.paths


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune TrOCR on handwriting data")
    parser.add_argument("--epochs",  type=int,   default=C.num_epochs)
    parser.add_argument("--lr",      type=float, default=C.learning_rate)
    parser.add_argument("--batch",   type=int,   default=C.train_batch_size)
    parser.add_argument("--local",   action="store_true",
                        help="Use local dataset instead of HuggingFace IAM")
    parser.add_argument("--rimes",   action="store_true",
                        help="Add RIMES dataset for ligature diversity")
    parser.add_argument("--images",  type=str,   default=str(P.raw_dir / "images"))
    parser.add_argument("--labels",  type=str,   default=str(P.raw_dir / "labels"))
    parser.add_argument("--hard-images", type=str, default=None,
                        help="Folder of hard (overlapping) sample images for oversampling")
    parser.add_argument("--hard-labels", type=str, default=None,
                        help="Folder of hard sample labels")
    parser.add_argument("--val-images", type=str, default=None)
    parser.add_argument("--val-labels", type=str, default=None)
    parser.add_argument("--resume",  type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--no-focal", action="store_true",
                        help="Disable focal loss (use standard cross-entropy)")
    return parser.parse_args()


# ── Focal loss ────────────────────────────────────────────────────────────────

def _focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Focal loss for seq2seq token prediction.
    logits: (B, T, vocab_size)
    labels: (B, T) with -100 for padding
    """
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = labels.reshape(-1)

    ce = F.cross_entropy(logits_flat, labels_flat,
                         ignore_index=ignore_index, reduction="none")
    valid_mask = (labels_flat != ignore_index).float()
    p_t = torch.exp(-ce)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce * valid_mask).sum() / valid_mask.sum().clamp(min=1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_batch(processor, label_ids):
    label_ids = label_ids.clone()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    return processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)


def evaluate_epoch(model, processor, loader, device) -> tuple[float, float]:
    model.eval()
    preds, refs = [], []
    with torch.inference_mode():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            generated = model.generate(
                pixel_values,
                num_beams=cfg.model.beam_size,
                max_new_tokens=cfg.model.max_new_tokens,
            )
            preds.extend(processor.tokenizer.batch_decode(generated, skip_special_tokens=True))
            refs.extend(_decode_batch(processor, batch["labels"]))
    return compute_cer(preds, refs), compute_wer(preds, refs)


# ── Training ──────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_focal = not args.no_focal
    log.info("Device: %s | Focal loss: %s (gamma=%.1f)",
             device, use_focal, C.focal_loss_gamma)

    # Load model
    model_name = args.resume or cfg.model.name
    log.info("Loading model: %s", model_name)
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model.name)
    image_processor = ViTImageProcessor.from_pretrained(cfg.model.name)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(device)

    # Dataset
    if args.local:
        val_images = args.val_images or args.images
        val_labels = args.val_labels or args.labels
        train_ds = LocalImageDataset(args.images, args.labels, processor)
        val_ds   = LocalImageDataset(val_images, val_labels, processor)
    else:
        train_ds = build_combined_dataset(
            processor, split="train",
            use_iam=True, use_rimes=args.rimes,
            local_images=args.images if Path(args.images).exists() else None,
            local_labels=args.labels if Path(args.labels).exists() else None,
        )
        val_ds = IAMLineDataset("validation", processor)

    # Hard-sample oversampling
    if args.hard_images and args.hard_labels:
        hard_ds = LocalImageDataset(args.hard_images, args.hard_labels, processor)
        ratio = C.hard_sample_oversample_ratio
        log.info("Oversampling %d hard samples x%d", len(hard_ds), ratio)
        train_ds = ConcatDataset([train_ds] + [hard_ds] * ratio)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=C.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=C.warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    best_cer = float("inf")
    patience_counter = 0
    P.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device)
            labels       = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)

            if use_focal:
                # outputs.logits is (B, T, vocab); apply focal loss instead of CE
                loss = _focal_loss(outputs.logits, labels, gamma=C.focal_loss_gamma)
            else:
                loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            if step % 50 == 0:
                log.info("Epoch %d | Step %d/%d | Loss %.4f",
                         epoch, step, len(train_loader), loss.item())

        avg_loss = epoch_loss / len(train_loader)
        cer, wer = evaluate_epoch(model, processor, val_loader, device)
        elapsed = time.perf_counter() - t0
        log.info("Epoch %d done | Loss %.4f | CER %.4f | WER %.4f | %.1fs",
                 epoch, avg_loss, cer, wer, elapsed)

        ckpt_path = P.checkpoints_dir / f"epoch_{epoch:02d}_cer{cer:.4f}"
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        log.info("Checkpoint saved: %s", ckpt_path)

        if cer < best_cer:
            best_cer = cer
            patience_counter = 0
            best_path = P.checkpoints_dir / "best_model"
            model.save_pretrained(best_path)
            processor.save_pretrained(best_path)
            log.info("New best model saved (CER=%.4f)", best_cer)
        else:
            patience_counter += 1
            log.info("No improvement. Patience %d/%d", patience_counter, C.early_stopping_patience)
            if patience_counter >= C.early_stopping_patience:
                log.info("Early stopping triggered at epoch %d", epoch)
                break

    log.info("Training complete. Best CER: %.4f", best_cer)


if __name__ == "__main__":
    train()
