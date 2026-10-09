"""
Image preprocessor for Custom TrOCR.
Converts any BGR numpy crop → grayscale tensor (1, 128, 1024).
"""
import cv2
import numpy as np
import torch


class ImageProcessor:
    def __init__(self, target_height=128, target_width=1024):
        self.h = target_height
        self.w = target_width

    def process(self, image) -> torch.Tensor:
        """
        Accept BGR numpy array or file path.
        Returns (1, H, W) float32 tensor normalized to [0, 1].
        """
        if isinstance(image, str):
            image = cv2.imread(image)

        # Convert to grayscale
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Resize to fixed dimensions
        gray = cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        return tensor.unsqueeze(0)   # (1, H, W)
