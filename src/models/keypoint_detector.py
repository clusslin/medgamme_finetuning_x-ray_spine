"""
Vertebra keypoint detection for spine alignment and measurement.

This module detects key anatomical landmarks on vertebrae:
- Superior endplate corners (anterior, posterior)
- Inferior endplate corners (anterior, posterior)
- Pedicles
- Spinous process
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class VertebraKeypointDetector(nn.Module):
    """
    Detect keypoints on vertebrae for geometric measurements.

    For each vertebra, detects:
    - 4 corners (superior anterior/posterior, inferior anterior/posterior)
    - 2 pedicle centers
    - 1 spinous process tip
    Total: 7 keypoints per vertebra
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_vertebrae: int = 24,  # C1-C7 (7), T1-T12 (12), L1-L5 (5)
        num_keypoints_per_vertebra: int = 7,
        heatmap_size: Tuple[int, int] = (128, 128)
    ):
        """
        Args:
            hidden_size: Size of input features
            num_vertebrae: Maximum number of vertebrae to detect
            num_keypoints_per_vertebra: Number of keypoints per vertebra
            heatmap_size: Size of output heatmap
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_vertebrae = num_vertebrae
        self.num_keypoints_per_vertebra = num_keypoints_per_vertebra
        self.heatmap_size = heatmap_size
        self.total_keypoints = num_vertebrae * num_keypoints_per_vertebra

        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Conv2d(hidden_size, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Upsampling to heatmap resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Heatmap prediction head
        self.heatmap_head = nn.Conv2d(
            32,
            self.total_keypoints,
            kernel_size=1
        )

        # Vertebra presence detection (which vertebrae are visible)
        self.presence_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_vertebrae)
        )

    def forward(
        self,
        features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to detect keypoints.

        Args:
            features: Input features (B, C, H, W)

        Returns:
            Dictionary containing:
                - heatmaps: Keypoint heatmaps (B, total_keypoints, H_out, W_out)
                - keypoints: Detected keypoint coordinates (B, total_keypoints, 2)
                - presence: Vertebra presence probabilities (B, num_vertebrae)
        """
        # Process features
        x = self.feature_processor(features)  # (B, 256, H, W)

        # Detect vertebra presence
        presence = torch.sigmoid(self.presence_head(x))  # (B, num_vertebrae)

        # Upsample
        x = self.upsample(x)  # (B, 32, H_out, W_out)

        # Generate heatmaps
        heatmaps = self.heatmap_head(x)  # (B, total_keypoints, H_out, W_out)
        heatmaps = torch.sigmoid(heatmaps)

        # Extract keypoint coordinates from heatmaps
        keypoints = self.extract_keypoints_from_heatmaps(heatmaps)

        return {
            'heatmaps': heatmaps,
            'keypoints': keypoints,
            'presence': presence
        }

    def extract_keypoints_from_heatmaps(
        self,
        heatmaps: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract keypoint coordinates from heatmaps using soft-argmax.

        Args:
            heatmaps: Heatmaps (B, K, H, W)

        Returns:
            Keypoint coordinates (B, K, 2) in normalized [0, 1] range
        """
        B, K, H, W = heatmaps.shape

        # Create coordinate grids
        y_coords = torch.linspace(0, 1, H, device=heatmaps.device)
        x_coords = torch.linspace(0, 1, W, device=heatmaps.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Reshape for broadcasting
        y_grid = y_grid.view(1, 1, H, W)
        x_grid = x_grid.view(1, 1, H, W)

        # Normalize heatmaps
        heatmap_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
        normalized_heatmaps = heatmaps / heatmap_sum

        # Soft-argmax
        x_coords = (normalized_heatmaps * x_grid).sum(dim=(2, 3))  # (B, K)
        y_coords = (normalized_heatmaps * y_grid).sum(dim=(2, 3))  # (B, K)

        # Stack coordinates
        keypoints = torch.stack([x_coords, y_coords], dim=-1)  # (B, K, 2)

        return keypoints

    def get_vertebra_keypoints(
        self,
        keypoints: torch.Tensor,
        vertebra_idx: int
    ) -> torch.Tensor:
        """
        Get keypoints for a specific vertebra.

        Args:
            keypoints: All keypoints (B, total_keypoints, 2)
            vertebra_idx: Vertebra index (0 to num_vertebrae-1)

        Returns:
            Keypoints for the vertebra (B, num_keypoints_per_vertebra, 2)
        """
        start_idx = vertebra_idx * self.num_keypoints_per_vertebra
        end_idx = start_idx + self.num_keypoints_per_vertebra
        return keypoints[:, start_idx:end_idx, :]

    def visualize_keypoints(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        presence: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Visualize detected keypoints on image.

        Args:
            image: Input image (H, W, 3)
            keypoints: Detected keypoints (total_keypoints, 2) in normalized coords
            presence: Vertebra presence scores (num_vertebrae,)
            threshold: Presence threshold

        Returns:
            Image with keypoints drawn
        """
        import cv2

        vis_image = image.copy()
        H, W = image.shape[:2]

        # Define keypoint colors (different color for each vertebra)
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]

        for vert_idx in range(self.num_vertebrae):
            # Check if vertebra is present
            if presence is not None and presence[vert_idx] < threshold:
                continue

            # Get keypoints for this vertebra
            start_idx = vert_idx * self.num_keypoints_per_vertebra
            end_idx = start_idx + self.num_keypoints_per_vertebra
            vert_keypoints = keypoints[start_idx:end_idx]

            # Choose color
            color = colors[vert_idx % len(colors)]

            # Draw keypoints
            for kp_idx, (x_norm, y_norm) in enumerate(vert_keypoints):
                x = int(x_norm * W)
                y = int(y_norm * H)

                # Different marker for different keypoint types
                if kp_idx < 4:  # Endplate corners
                    cv2.circle(vis_image, (x, y), 5, color, -1)
                elif kp_idx < 6:  # Pedicles
                    cv2.circle(vis_image, (x, y), 4, color, 2)
                else:  # Spinous process
                    cv2.drawMarker(vis_image, (x, y), color,
                                 cv2.MARKER_CROSS, 8, 2)

            # Draw vertebra box (using 4 corner points)
            if len(vert_keypoints) >= 4:
                corners = vert_keypoints[:4]
                pts = np.array([
                    [int(corners[0][0] * W), int(corners[0][1] * H)],  # Superior anterior
                    [int(corners[1][0] * W), int(corners[1][1] * H)],  # Superior posterior
                    [int(corners[3][0] * W), int(corners[3][1] * H)],  # Inferior posterior
                    [int(corners[2][0] * W), int(corners[2][1] * H)],  # Inferior anterior
                ], np.int32)
                cv2.polylines(vis_image, [pts], True, color, 2)

        return vis_image


class KeypointLoss(nn.Module):
    """
    Loss function for keypoint detection.
    Combines heatmap loss and coordinate loss.
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        presence_weight: float = 0.5
    ):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.presence_weight = presence_weight

    def forward(
        self,
        pred_heatmaps: torch.Tensor,
        pred_presence: torch.Tensor,
        target_heatmaps: torch.Tensor,
        target_presence: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate keypoint detection loss.

        Args:
            pred_heatmaps: Predicted heatmaps (B, K, H, W)
            pred_presence: Predicted presence (B, num_vertebrae)
            target_heatmaps: Target heatmaps (B, K, H, W)
            target_presence: Target presence (B, num_vertebrae)

        Returns:
            Dictionary of losses
        """
        # Heatmap loss (MSE)
        heatmap_loss = F.mse_loss(pred_heatmaps, target_heatmaps)

        # Presence loss (BCE)
        presence_loss = F.binary_cross_entropy(pred_presence, target_presence)

        # Total loss
        total_loss = (
            self.heatmap_weight * heatmap_loss +
            self.presence_weight * presence_loss
        )

        return {
            'total_loss': total_loss,
            'heatmap_loss': heatmap_loss,
            'presence_loss': presence_loss
        }

    @staticmethod
    def generate_gaussian_heatmap(
        keypoints: np.ndarray,
        heatmap_size: Tuple[int, int],
        sigma: float = 2.0
    ) -> np.ndarray:
        """
        Generate Gaussian heatmap from keypoint coordinates.

        Args:
            keypoints: Keypoint coordinates (K, 2) in normalized [0, 1] range
            heatmap_size: Output heatmap size (H, W)
            sigma: Gaussian sigma

        Returns:
            Heatmaps (K, H, W)
        """
        K = len(keypoints)
        H, W = heatmap_size
        heatmaps = np.zeros((K, H, W), dtype=np.float32)

        # Create coordinate grids
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

        for k, (x_norm, y_norm) in enumerate(keypoints):
            # Convert normalized coords to pixel coords
            x_center = x_norm * (W - 1)
            y_center = y_norm * (H - 1)

            # Generate Gaussian
            gaussian = np.exp(
                -((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma ** 2)
            )
            heatmaps[k] = gaussian

        return heatmaps
