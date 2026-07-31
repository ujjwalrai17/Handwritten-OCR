"""
IAM Handwriting Dataset Loader
Supports: HuggingFace datasets (IAM) + local folder of image/label pairs
Used by: train.py
"""

from pathlib import Path
from torch.utils.data import Dataset
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
