"""
TrOCR Fine-Tuning Script
Usage:
    python train.py
    python train.py --epochs 5 --lr 3e-5 --batch 4
    python train.py --local --images data/raw/images --labels data/raw/labels

Training Pipeline:
    IAM Dataset → DataLoader → TrOCR Fine-tune → Checkpoint → CER/WER Eval
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    RobertaTokenizer,
    ViTImageProcessor,
    get_linear_schedule_with_warmup,
)

from config.settings import cfg
from ocr.dataset.iam_dataset import IAMLineDataset, LocalImageDataset
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
    parser.add_argument("--images",  type=str,   default=str(P.raw_dir / "images"))
    parser.add_argument("--labels",  type=str,   default=str(P.raw_dir / "labels"))
    parser.add_argument("--val-images", type=str, default=None,
                        help="Validation images dir (defaults to --images if not set)")
    parser.add_argument("--val-labels", type=str, default=None,
                        help="Validation labels dir (defaults to --labels if not set)")
    parser.add_argument("--resume",  type=str,   default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def _decode_batch(processor, label_ids):
    """Replace -100 padding with pad_token_id before decoding."""
    label_ids = label_ids.clone()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    return processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)


def evaluate_epoch(model, processor, loader, device) -> tuple[float, float]:
    """Run validation and return (CER, WER)."""
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
            decoded_preds = processor.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            decoded_refs = _decode_batch(processor, batch["labels"])
            preds.extend(decoded_preds)
            refs.extend(decoded_refs)

    cer = compute_cer(preds, refs)
    wer = compute_wer(preds, refs)
    return cer, wer


def train():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    # ── Load model ────────────────────────────────────────────────────────────
    model_name = args.resume or cfg.model.name
    log.info("Loading model: %s", model_name)
    tokenizer = RobertaTokenizer.from_pretrained(cfg.model.name)
    image_processor = ViTImageProcessor.from_pretrained(cfg.model.name)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    # Required config for seq2seq generation
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    if args.local:
        val_images = args.val_images or args.images
        val_labels = args.val_labels or args.labels
        train_ds = LocalImageDataset(args.images, args.labels, processor)
        val_ds   = LocalImageDataset(val_images, val_labels, processor)
    else:
        train_ds = IAMLineDataset("train", processor)
        val_ds   = IAMLineDataset("validation", processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=C.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=C.warmup_steps, num_training_steps=total_steps
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
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

        # ── Checkpoint ────────────────────────────────────────────────────────
        ckpt_path = P.checkpoints_dir / f"epoch_{epoch:02d}_cer{cer:.4f}"
        model.save_pretrained(ckpt_path)
        processor.save_pretrained(ckpt_path)
        log.info("Checkpoint saved: %s", ckpt_path)

        # ── Early Stopping ────────────────────────────────────────────────────
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
