"""
IAM Handwriting Dataset Loader
Supports:
  - HuggingFace IAM (Teklia/IAM-line)          — primary cursive dataset
  - HuggingFace RIMES (via datasets)            — French cursive, good for ligatures
  - Local folder of image/label pairs
  - ConcatDataset for multi-source training

Public dataset recommendations for cursive HTR:
  IAM   (English, 13k lines) — best general cursive baseline
  RIMES (French, 12k lines)  — excellent for connected/ligature script
  CVL   (English/German)     — multi-writer, good for writer-independent models
  GNHK  (English)            — specifically messy/fast handwriting

Used by: train.py, train_crnn.py
"""

from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
from config.settings import cfg
from ocr.utils.logger import get_logger

log = get_logger(__name__)


class IAMLineDataset(Dataset):
    """
    Loads IAM line-level dataset from HuggingFace.
    Each sample: {"image": PIL.Image, "text": str}
    """

    def __init__(self, split: str = "train", processor=None):
        from datasets import load_dataset
        log.info("Loading IAM dataset split='%s'...", split)
        self._ds = load_dataset("Teklia/IAM-line", split=split, trust_remote_code=True)
        self._processor = processor
        log.info("IAM %s: %d samples", split, len(self._ds))

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]
        image = sample["image"].convert("RGB")
        text = sample["text"]

        if self._processor is None:
            return {"image": image, "text": text}

        encoding = self._processor(images=image, return_tensors="pt")
        labels = self._processor.tokenizer(
            text,
            padding="max_length",
            max_length=cfg.training.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == self._processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoding.pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
        }


class RIMESLineDataset(Dataset):
    """
    RIMES 2011 line-level dataset via HuggingFace.
    French cursive — excellent for training on connected/ligature strokes.
    Combine with IAM for a multilingual cursive model.
    """

    def __init__(self, split: str = "train", processor=None):
        from datasets import load_dataset
        log.info("Loading RIMES dataset split='%s'...", split)
        # Note: use trust_remote_code=True if required by the dataset card
        self._ds = load_dataset("Teklia/RIMES-2011-line", split=split, trust_remote_code=True)
        self._processor = processor
        log.info("RIMES %s: %d samples", split, len(self._ds))

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        sample = self._ds[idx]
        image = sample["image"].convert("RGB")
        text = sample["text"]
        if self._processor is None:
            return {"image": image, "text": text}
        encoding = self._processor(images=image, return_tensors="pt")
        labels = self._processor.tokenizer(
            text, padding="max_length",
            max_length=cfg.training.max_target_length,
            truncation=True, return_tensors="pt",
        ).input_ids
        labels[labels == self._processor.tokenizer.pad_token_id] = -100
        return {
            "pixel_values": encoding.pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
        }


def build_combined_dataset(
    processor,
    split: str = "train",
    use_iam: bool = True,
    use_rimes: bool = False,
    local_images: str = None,
    local_labels: str = None,
) -> Dataset:
    """
    Build a combined dataset from multiple sources for transfer learning.
    Recommended strategy:
      1. Pre-train on IAM (large, clean cursive)
      2. Fine-tune on IAM + your local messy samples
      3. Optionally add RIMES for ligature diversity
    """
    datasets = []
    if use_iam:
        datasets.append(IAMLineDataset(split, processor))
    if use_rimes:
        try:
            datasets.append(RIMESLineDataset(split, processor))
        except Exception as e:
            log.warning("RIMES dataset unavailable: %s", e)
    if local_images and local_labels:
        datasets.append(LocalImageDataset(local_images, local_labels, processor))

    if not datasets:
        raise ValueError("No datasets specified for build_combined_dataset")
    if len(datasets) == 1:
        return datasets[0]

    combined = ConcatDataset(datasets)
    log.info("Combined dataset: %d total samples from %d sources",
             len(combined), len(datasets))
    return combined


class LocalImageDataset(Dataset):
    """
    Loads image/label pairs from a local folder.
    Expects: images/*.jpg  and  labels/*.txt  (matching stems)
    """

    def __init__(self, images_dir: str, labels_dir: str, processor=None):
        self._images = sorted(Path(images_dir).glob("*.jpg")) + \
                       sorted(Path(images_dir).glob("*.png"))
        self._labels_dir = Path(labels_dir)
        self._processor = processor
        log.info("LocalDataset: %d images from %s", len(self._images), images_dir)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img_path = self._images[idx]
        label_path = self._labels_dir / (img_path.stem + ".txt")
        text = label_path.read_text(encoding="utf-8").strip() if label_path.exists() else ""
        image = Image.open(img_path).convert("RGB")

        if self._processor is None:
            return {"image": image, "text": text}

        encoding = self._processor(images=image, return_tensors="pt")
        labels = self._processor.tokenizer(
            text,
            padding="max_length",
            max_length=cfg.training.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids
        labels[labels == self._processor.tokenizer.pad_token_id] = -100

        return {
            "pixel_values": encoding.pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
        }
