"""
Inference for Custom TrOCR.
Supports greedy decoding and beam search.

Usage:
    python -m ocr.recognition.custom_trocr.predict --image data/samples/nishant.jpg
    python -m ocr.recognition.custom_trocr.predict --image data/samples/nishant.jpg --beam 5
"""

import argparse
import heapq
from pathlib import Path

import torch

from ocr.recognition.custom_trocr.config import cfg
from ocr.recognition.custom_trocr.model import CustomTrOCR
from ocr.recognition.custom_trocr.tokenizer import CharacterTokenizer, BOS_ID, EOS_ID
from ocr.recognition.custom_trocr.image_processor import ImageProcessor


def load_model(checkpoint: str, device: str):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    vocab_size = ckpt["vocab_size"]
    model = CustomTrOCR(vocab_size=vocab_size).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab_size


@torch.no_grad()
def greedy_decode(model, image_tensor, tokenizer, max_len, device):
    img = image_tensor.unsqueeze(0).to(device)   # (1, 1, H, W)
    memory = model.encode(img)                   # (1, N, D)

    generated = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits = model.decoder(generated, memory)
        next_token = logits[0, -1, :].argmax().item()
        generated = torch.cat([generated,
                                torch.tensor([[next_token]], device=device)], dim=1)
        if next_token == EOS_ID:
            break
    return tokenizer.decode(generated[0].cpu().tolist(), skip_special=True)


@torch.no_grad()
def beam_search_decode(model, image_tensor, tokenizer, max_len, device, beam_size=5):
    img = image_tensor.unsqueeze(0).to(device)
    memory = model.encode(img)

    beams = [(0.0, [BOS_ID])]
    completed = []

    for _ in range(max_len - 1):
        candidates = []
        for score, tokens in beams:
            if tokens[-1] == EOS_ID:
                completed.append((score, tokens))
                continue
            seq = torch.tensor([tokens], dtype=torch.long, device=device)
            logits = model.decoder(seq, memory)
            log_probs = torch.log_softmax(logits[0, -1, :], dim=-1)
            topk_probs, topk_ids = log_probs.topk(beam_size)
            for prob, tid in zip(topk_probs.tolist(), topk_ids.tolist()):
                candidates.append((score - prob, tokens + [tid]))
        beams = heapq.nsmallest(beam_size, candidates, key=lambda x: x[0])
        if not beams:
            break

    completed += beams
    _, best_tokens = min(completed, key=lambda x: x[0])
    return tokenizer.decode(best_tokens, skip_special=True)


def predict(image_path: str, checkpoint: str = None, beam_size: int = 1):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint is None:
        checkpoint = str(cfg.paths.checkpoint_dir / "best.pt")

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"No checkpoint found at {checkpoint}\n"
            f"Train first: python -m ocr.recognition.custom_trocr.train"
        )

    # Load tokenizer
    tok_path = cfg.paths.data_processed / "tokenizer.json"
    tokenizer = (CharacterTokenizer.load(tok_path)
                 if tok_path.exists() else CharacterTokenizer())

    model, _ = load_model(checkpoint, device)
    processor = ImageProcessor(cfg.model.image_height, cfg.model.image_width)
    image_tensor = processor.process(image_path)

    if beam_size <= 1:
        text = greedy_decode(model, image_tensor, tokenizer,
                             cfg.model.max_text_length, device)
        method = "greedy"
    else:
        text = beam_search_decode(model, image_tensor, tokenizer,
                                  cfg.model.max_text_length, device, beam_size)
        method = f"beam(k={beam_size})"

    print(f"[{method}] {text}")
    return text


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",      required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--beam",       type=int, default=1)
    args = p.parse_args()
    predict(args.image, args.checkpoint, args.beam)
