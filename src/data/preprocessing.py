"""
Image preprocessing utilities for spine X-ray images
"""

from typing import Tuple, Optional, List
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import cv2


class SpineImagePreprocessor:
    """
    Preprocessing pipeline for spine X-ray images.
    Handles:
    - Resizing
    - Normalization
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - Bone enhancement
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 512),
        normalize_mean: List[float] = [0.485, 0.456, 0.406],
        normalize_std: List[float] = [0.229, 0.224, 0.225],
        use_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8)
    ):
        """
        Args:
            image_size: Target image size (height, width)
            normalize_mean: Mean values for normalization
            normalize_std: Std values for normalization
            use_clahe: Whether to apply CLAHE
            clahe_clip_limit: Clip limit for CLAHE
            clahe_tile_size: Tile grid size for CLAHE
        """
        self.image_size = image_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.use_clahe = use_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size

        # Initialize CLAHE
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to enhance contrast in X-ray images.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Enhanced image
        """
        if len(image.shape) == 3:
            # Convert to YCrCb and apply CLAHE to Y channel
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = self.clahe.apply(ycrcb[:, :, 0])
            enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        else:
            # Grayscale image
            enhanced = self.clahe.apply(image)

        return enhanced

    def enhance_bones(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance bone structures in X-ray images using morphological operations.

        Args:
            image: Input image (H, W, C) or (H, W)

        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Top-hat transform to enhance bright structures (bones)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Add to original
        enhanced = cv2.add(gray, tophat)

        # Convert back to RGB if needed
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        return enhanced

    def preprocess_numpy(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess a numpy array image.

        Args:
            image: Input image array

        Returns:
            Preprocessed image
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

        # Apply CLAHE
        if self.use_clahe:
            image = self.apply_clahe(image)

        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LANCZOS4)

        return image

    def get_transform(self, train: bool = True) -> transforms.Compose:
        """
        Get torchvision transform pipeline.

        Args:
            train: Whether this is for training (affects some transforms)

        Returns:
            Composed transforms
        """
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ]

        return transforms.Compose(transform_list)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Process PIL Image.

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor
        """
        # Convert to numpy
        image_np = np.array(image)

        # Preprocess
        image_np = self.preprocess_numpy(image_np)

        # Convert to PIL for torchvision transforms
        image_pil = Image.fromarray(image_np)

        # Apply transforms
        transform = self.get_transform(train=False)
        tensor = transform(image_pil)

        return tensor


class DICOMPreprocessor:
    """
    Specialized preprocessor for DICOM images.
    Handles window/level adjustments for bone visualization.
    """

    def __init__(
        self,
        bone_window_center: int = 400,
        bone_window_width: int = 1800,
        soft_tissue_window_center: int = 50,
        soft_tissue_window_width: int = 350
    ):
        """
        Args:
            bone_window_center: Window center for bone visualization
            bone_window_width: Window width for bone visualization
            soft_tissue_window_center: Window center for soft tissue
            soft_tissue_window_width: Window width for soft tissue
        """
        self.bone_window_center = bone_window_center
        self.bone_window_width = bone_window_width
        self.soft_tissue_window_center = soft_tissue_window_center
        self.soft_tissue_window_width = soft_tissue_window_width

    def apply_window(
        self,
        image: np.ndarray,
        window_center: int,
        window_width: int
    ) -> np.ndarray:
        """
        Apply window/level adjustment to DICOM image.

        Args:
            image: Input image array (Hounsfield units)
            window_center: Window center value
            window_width: Window width value

        Returns:
            Windowed image (0-255)
        """
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2

        # Apply window
        windowed = np.clip(image, img_min, img_max)

        # Normalize to 0-255
        windowed = ((windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        return windowed

    def preprocess_dicom(
        self,
        pixel_array: np.ndarray,
        window_type: str = "bone"
    ) -> np.ndarray:
        """
        Preprocess DICOM pixel array.

        Args:
            pixel_array: Raw pixel array from DICOM
            window_type: "bone" or "soft_tissue"

        Returns:
            Preprocessed image
        """
        if window_type == "bone":
            windowed = self.apply_window(
                pixel_array,
                self.bone_window_center,
                self.bone_window_width
            )
        elif window_type == "soft_tissue":
            windowed = self.apply_window(
                pixel_array,
                self.soft_tissue_window_center,
                self.soft_tissue_window_width
            )
        else:
            # Auto-scale
            windowed = ((pixel_array - pixel_array.min()) /
                       (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)

        return windowed


class MultiViewProcessor:
    """
    Process multiple views of the same spine region together.
    Useful for combining AP, lateral, and oblique views.
    """

    def __init__(self, num_views: int = 2):
        """
        Args:
            num_views: Number of views to process together
        """
        self.num_views = num_views

    def stack_views(self, images: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack multiple views into a single tensor.

        Args:
            images: List of image tensors (C, H, W)

        Returns:
            Stacked tensor (num_views * C, H, W)
        """
        # Ensure all images have same size
        assert all(img.shape == images[0].shape for img in images), \
            "All images must have the same shape"

        # Stack along channel dimension
        stacked = torch.cat(images, dim=0)

        return stacked

    def process_multi_view_batch(
        self,
        view_dict: dict
    ) -> torch.Tensor:
        """
        Process a dictionary of different views.

        Args:
            view_dict: Dictionary with keys like 'ap', 'lateral', 'oblique'

        Returns:
            Processed multi-view tensor
        """
        # Get available views in consistent order
        view_order = ['ap', 'lateral', 'oblique_left', 'oblique_right']
        available_views = [view_dict[k] for k in view_order if k in view_dict]

        # Pad if necessary
        while len(available_views) < self.num_views:
            # Duplicate last view or create zeros
            if available_views:
                available_views.append(available_views[-1].clone())
            else:
                raise ValueError("At least one view must be provided")

        # Truncate if too many
        available_views = available_views[:self.num_views]

        # Stack
        return self.stack_views(available_views)
