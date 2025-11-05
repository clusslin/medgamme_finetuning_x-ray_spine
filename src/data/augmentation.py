"""
Data augmentation for spine X-ray images using Albumentations
"""

from typing import Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class SpineAugmentation:
    """
    Data augmentation pipeline specifically designed for spine X-ray images.
    Conservative augmentations to preserve diagnostic quality.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Augmentation configuration dictionary
        """
        self.config = config

    def get_transform(self, train: bool = True) -> A.Compose:
        """
        Get augmentation pipeline.

        Args:
            train: If True, return training augmentations. Else, minimal/no augmentation.

        Returns:
            Albumentations Compose object
        """
        if not train:
            # No augmentation for validation/test
            return A.Compose([])

        # Training augmentations - conservative for medical images
        augmentations = [
            # Geometric transformations
            A.HorizontalFlip(
                p=self.config.get('horizontal_flip_prob', 0.5)
            ),

            # Small rotations (spine should remain mostly upright)
            A.Rotate(
                limit=self.config.get('rotation_limit', 10),
                border_mode=0,
                p=0.5
            ),

            # Slight shifts
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=5,
                border_mode=0,
                p=0.3
            ),

            # Brightness and contrast adjustments
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=self.config.get('brightness_contrast_prob', 0.3)
            ),

            # Gamma correction (simulates different X-ray exposure settings)
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),

            # Add slight noise (simulates sensor noise)
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=self.config.get('noise_prob', 0.2)
            ),

            # Elastic deformation (very subtle for spine)
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                border_mode=0,
                p=0.1
            ),

            # Grid distortion (simulates slight positioning variations)
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.1,
                border_mode=0,
                p=0.1
            ),

            # Blur (simulates motion or focus issues)
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=0.1),

            # CLAHE (enhance local contrast)
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),

            # Sharpen
            A.Sharpen(
                alpha=(0.1, 0.3),
                lightness=(0.5, 1.0),
                p=0.2
            ),
        ]

        return A.Compose(augmentations)


class PathologySpecificAugmentation:
    """
    Augmentations tailored for specific pathologies.
    Different pathologies may benefit from different augmentation strategies.
    """

    @staticmethod
    def get_fracture_augmentation() -> A.Compose:
        """
        Augmentation for fracture detection.
        Focus on preserving fine details and edges.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.3),  # Very small rotation
            A.RandomBrightnessContrast(p=0.3),
            A.Sharpen(alpha=(0.2, 0.5), p=0.5),  # Enhance edges
            A.CLAHE(p=0.3),
        ])

    @staticmethod
    def get_degeneration_augmentation() -> A.Compose:
        """
        Augmentation for degenerative changes.
        Can tolerate more variation.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomGamma(p=0.3),
            A.CLAHE(p=0.3),
        ])

    @staticmethod
    def get_implant_augmentation() -> A.Compose:
        """
        Augmentation for implant detection.
        Metal appears very bright, needs contrast adjustment.
        """
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.5
            ),
            A.CLAHE(clip_limit=4.0, p=0.4),  # Higher clip limit for metal
        ])


class TestTimeAugmentation:
    """
    Test-Time Augmentation (TTA) for improved inference.
    Apply multiple augmentations at test time and average predictions.
    """

    def __init__(self, num_augments: int = 5):
        """
        Args:
            num_augments: Number of augmented versions to create
        """
        self.num_augments = num_augments

    def get_tta_transforms(self) -> list:
        """
        Get list of TTA transforms.

        Returns:
            List of Albumentations Compose objects
        """
        tta_transforms = [
            # Original
            A.Compose([]),

            # Horizontal flip
            A.Compose([A.HorizontalFlip(p=1.0)]),

            # Small rotation
            A.Compose([A.Rotate(limit=5, p=1.0)]),

            # Brightness adjustment
            A.Compose([A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=1.0
            )]),

            # CLAHE
            A.Compose([A.CLAHE(p=1.0)]),
        ]

        return tta_transforms[:self.num_augments]

    def apply_tta(self, image: np.ndarray) -> list:
        """
        Apply TTA to an image.

        Args:
            image: Input image (H, W, C)

        Returns:
            List of augmented images
        """
        transforms = self.get_tta_transforms()
        augmented_images = []

        for transform in transforms:
            augmented = transform(image=image)['image']
            augmented_images.append(augmented)

        return augmented_images


class MixUp:
    """
    MixUp augmentation for images and labels.
    Helps with regularization and generalization.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray
    ) -> tuple:
        """
        Apply MixUp to two images and their labels.

        Args:
            image1: First image
            image2: Second image
            label1: First label
            label2: Second label

        Returns:
            Mixed image and label
        """
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix images
        mixed_image = lam * image1 + (1 - lam) * image2

        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image.astype(image1.dtype), mixed_label


class CutMix:
    """
    CutMix augmentation for images.
    Cuts and pastes patches between images.
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray
    ) -> tuple:
        """
        Apply CutMix to two images.

        Args:
            image1: First image (H, W, C)
            image2: Second image (H, W, C)
            label1: First label
            label2: Second label

        Returns:
            Mixed image and label
        """
        h, w = image1.shape[:2]

        # Sample lambda
        lam = np.random.beta(self.alpha, self.alpha)

        # Sample cut size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)

        # Sample cut position
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        # Get bounding box
        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)

        # Cut and paste
        mixed_image = image1.copy()
        mixed_image[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

        # Adjust lambda to actual area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))

        # Mix labels
        mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_image, mixed_label
