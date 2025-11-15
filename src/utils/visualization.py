"""
Visualization tools for spine measurements and annotations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
from typing import List, Dict, Tuple, Optional
from .measurements import VertebraKeypoints, SpineMeasurements


class SpineVisualizer:
    """Visualize spine measurements and annotations."""

    def __init__(self, pixel_spacing: float = 1.0):
        """
        Args:
            pixel_spacing: mm per pixel
        """
        self.pixel_spacing = pixel_spacing
        self.measurements = SpineMeasurements(pixel_spacing)

    def draw_keypoints(
        self,
        image: np.ndarray,
        vertebrae: List[VertebraKeypoints],
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw vertebra keypoints on image.

        Args:
            image: Input image (H, W, 3)
            vertebrae: List of vertebra keypoints
            show_labels: Whether to show vertebra labels

        Returns:
            Annotated image
        """
        vis_image = image.copy()
        H, W = image.shape[:2]

        # Define colors for different vertebrae
        colors = [
            (255, 100, 100),  # Light red
            (100, 255, 100),  # Light green
            (100, 100, 255),  # Light blue
            (255, 255, 100),  # Light cyan
            (255, 100, 255),  # Light magenta
            (100, 255, 255),  # Light yellow
        ]

        for i, vertebra in enumerate(vertebrae):
            color = colors[i % len(colors)]

            # Get corners
            corners = vertebra.get_corners()

            # Draw vertebral body outline
            pts = []
            for corner in corners:
                x = int(corner[0] * W) if corner[0] <= 1 else int(corner[0])
                y = int(corner[1] * H) if corner[1] <= 1 else int(corner[1])
                pts.append([x, y])

            pts = np.array([
                pts[0],  # Superior anterior
                pts[1],  # Superior posterior
                pts[3],  # Inferior posterior
                pts[2],  # Inferior anterior
            ], np.int32)

            # Draw outline
            cv2.polylines(vis_image, [pts], True, color, 2)

            # Draw corner points
            for pt in pts:
                cv2.circle(vis_image, tuple(pt), 4, color, -1)

            # Draw endplate lines (thicker)
            cv2.line(vis_image, tuple(pts[0]), tuple(pts[1]), color, 3)  # Superior
            cv2.line(vis_image, tuple(pts[2]), tuple(pts[3]), color, 3)  # Inferior

            # Draw pedicles if available
            if vertebra.pedicle_left is not None:
                x = int(vertebra.pedicle_left[0] * W) if vertebra.pedicle_left[0] <= 1 else int(vertebra.pedicle_left[0])
                y = int(vertebra.pedicle_left[1] * H) if vertebra.pedicle_left[1] <= 1 else int(vertebra.pedicle_left[1])
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), 2)

            if vertebra.pedicle_right is not None:
                x = int(vertebra.pedicle_right[0] * W) if vertebra.pedicle_right[0] <= 1 else int(vertebra.pedicle_right[0])
                y = int(vertebra.pedicle_right[1] * H) if vertebra.pedicle_right[1] <= 1 else int(vertebra.pedicle_right[1])
                cv2.circle(vis_image, (x, y), 5, (0, 255, 0), 2)

            # Draw spinous process if available
            if vertebra.spinous_process is not None:
                x = int(vertebra.spinous_process[0] * W) if vertebra.spinous_process[0] <= 1 else int(vertebra.spinous_process[0])
                y = int(vertebra.spinous_process[1] * H) if vertebra.spinous_process[1] <= 1 else int(vertebra.spinous_process[1])
                cv2.drawMarker(vis_image, (x, y), (255, 0, 255),
                             cv2.MARKER_CROSS, 10, 2)

            # Label vertebra
            if show_labels:
                label_pos = (pts[0] + pts[1]) // 2
                cv2.putText(
                    vis_image,
                    f"V{i}",
                    tuple(label_pos - [0, 10]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        return vis_image

    def draw_alignment_lines(
        self,
        image: np.ndarray,
        vertebrae: List[VertebraKeypoints]
    ) -> np.ndarray:
        """
        Draw anterior and posterior vertebral lines.

        Args:
            image: Input image (H, W, 3)
            vertebrae: List of vertebra keypoints

        Returns:
            Annotated image
        """
        vis_image = image.copy()
        H, W = image.shape[:2]

        if len(vertebrae) < 2:
            return vis_image

        # Compute alignment lines
        avl_point, avl_direction = self.measurements.compute_anterior_vertebral_line(
            vertebrae
        )
        pvl_point, pvl_direction = self.measurements.compute_posterior_vertebral_line(
            vertebrae
        )

        # Draw anterior vertebral line (AVL) - green
        self._draw_infinite_line(
            vis_image,
            avl_point,
            avl_direction,
            (0, 255, 0),
            2,
            H,
            W,
            "AVL"
        )

        # Draw posterior vertebral line (PVL) - blue
        self._draw_infinite_line(
            vis_image,
            pvl_point,
            pvl_direction,
            (255, 0, 0),
            2,
            H,
            W,
            "PVL"
        )

        return vis_image

    def draw_measurements(
        self,
        image: np.ndarray,
        vertebrae: List[VertebraKeypoints],
        show_heights: bool = True,
        show_disc_spaces: bool = True,
        show_cobb_angle: bool = True
    ) -> np.ndarray:
        """
        Draw measurements on image.

        Args:
            image: Input image (H, W, 3)
            vertebrae: List of vertebra keypoints
            show_heights: Show vertebral body heights
            show_disc_spaces: Show disc space heights
            show_cobb_angle: Show Cobb angle

        Returns:
            Annotated image
        """
        vis_image = image.copy()
        H, W = image.shape[:2]

        # Draw vertebral body heights
        if show_heights:
            for i, vertebra in enumerate(vertebrae):
                # Compute anterior height
                height_mm = self.measurements.compute_vertebral_body_height(
                    vertebra, 'anterior'
                )

                # Get anterior edge midpoint for label
                p1 = np.array(vertebra.superior_anterior)
                p2 = np.array(vertebra.inferior_anterior)

                # Convert to pixels
                p1_px = (int(p1[0] * W) if p1[0] <= 1 else int(p1[0]),
                        int(p1[1] * H) if p1[1] <= 1 else int(p1[1]))
                p2_px = (int(p2[0] * W) if p2[0] <= 1 else int(p2[0]),
                        int(p2[1] * H) if p2[1] <= 1 else int(p2[1]))

                # Draw measurement line
                cv2.line(vis_image, p1_px, p2_px, (255, 255, 0), 2)

                # Add text label
                mid_point = ((p1_px[0] + p2_px[0]) // 2 - 50,
                           (p1_px[1] + p2_px[1]) // 2)
                cv2.putText(
                    vis_image,
                    f"{height_mm:.1f}mm",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

        # Draw disc space heights
        if show_disc_spaces and len(vertebrae) >= 2:
            for i in range(len(vertebrae) - 1):
                superior = vertebrae[i]
                inferior = vertebrae[i + 1]

                # Compute disc height
                disc_height = self.measurements.compute_disc_height(
                    superior, inferior, 'mid'
                )

                # Get midpoint between vertebrae
                p1 = np.array(superior.inferior_anterior)
                p2 = np.array(inferior.superior_anterior)

                p1_px = (int(p1[0] * W) if p1[0] <= 1 else int(p1[0]),
                        int(p1[1] * H) if p1[1] <= 1 else int(p1[1]))
                p2_px = (int(p2[0] * W) if p2[0] <= 1 else int(p2[0]),
                        int(p2[1] * H) if p2[1] <= 1 else int(p2[1]))

                # Draw line
                cv2.line(vis_image, p1_px, p2_px, (0, 255, 255), 2)

                # Add label
                mid_point = ((p1_px[0] + p2_px[0]) // 2 + 10,
                           (p1_px[1] + p2_px[1]) // 2)
                cv2.putText(
                    vis_image,
                    f"{disc_height:.1f}mm",
                    mid_point,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

        # Draw Cobb angle
        if show_cobb_angle and len(vertebrae) >= 2:
            cobb_angle = self.measurements.compute_cobb_angle(
                vertebrae[0],
                vertebrae[-1]
            )

            # Draw angle arc (simplified)
            # Get superior endplate of first vertebra
            sup_p1, sup_p2 = vertebrae[0].get_superior_line()
            center_px = (
                int((sup_p1[0] + sup_p2[0]) / 2 * W),
                int((sup_p1[1] + sup_p2[1]) / 2 * H)
            )

            # Add text
            cv2.putText(
                vis_image,
                f"Cobb: {cobb_angle:.1f}deg",
                (center_px[0] + 20, center_px[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 128, 0),
                2
            )

        return vis_image

    def create_measurement_figure(
        self,
        vertebrae: List[VertebraKeypoints],
        region: str = "spine"
    ) -> Figure:
        """
        Create matplotlib figure with measurement plots.

        Args:
            vertebrae: List of vertebra keypoints
            region: Spine region

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Spine Measurements - {region.upper()}", fontsize=16)

        # Vertebral body heights
        if len(vertebrae) > 0:
            ax = axes[0, 0]
            heights_ant = [
                self.measurements.compute_vertebral_body_height(v, 'anterior')
                for v in vertebrae
            ]
            heights_post = [
                self.measurements.compute_vertebral_body_height(v, 'posterior')
                for v in vertebrae
            ]
            x = range(len(vertebrae))
            ax.plot(x, heights_ant, 'o-', label='Anterior', linewidth=2)
            ax.plot(x, heights_post, 's-', label='Posterior', linewidth=2)
            ax.set_xlabel('Vertebra Index')
            ax.set_ylabel('Height (mm)')
            ax.set_title('Vertebral Body Heights')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Disc heights
        if len(vertebrae) >= 2:
            ax = axes[0, 1]
            disc_distances = self.measurements.compute_intervertebral_distances(
                vertebrae
            )
            disc_heights = [d['mid_mm'] for d in disc_distances]
            x = range(len(disc_heights))
            ax.plot(x, disc_heights, 'o-', linewidth=2, color='cyan')
            ax.set_xlabel('Disc Level')
            ax.set_ylabel('Height (mm)')
            ax.set_title('Intervertebral Disc Heights')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=np.mean(disc_heights), color='r', linestyle='--',
                      label=f'Mean: {np.mean(disc_heights):.1f}mm')
            ax.legend()

        # Alignment deviations
        if len(vertebrae) >= 3:
            ax = axes[1, 0]
            alignment = self.measurements.compute_alignment_deviation(vertebrae)
            x = range(len(alignment['anterior_deviations']))
            ax.plot(x, alignment['anterior_deviations'], 'o-',
                   label='Anterior', linewidth=2)
            ax.plot(x, alignment['posterior_deviations'], 's-',
                   label='Posterior', linewidth=2)
            ax.set_xlabel('Vertebra Index')
            ax.set_ylabel('Deviation (mm)')
            ax.set_title('Alignment Deviations from Reference Lines')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Disc wedging
        if len(vertebrae) >= 2:
            ax = axes[1, 1]
            disc_distances = self.measurements.compute_intervertebral_distances(
                vertebrae
            )
            wedging = [d['wedging_mm'] for d in disc_distances]
            x = range(len(wedging))
            bars = ax.bar(x, wedging, color='orange', alpha=0.7)

            # Color bars differently based on positive/negative
            for i, (bar, w) in enumerate(zip(bars, wedging)):
                if w < 0:
                    bar.set_color('red')

            ax.set_xlabel('Disc Level')
            ax.set_ylabel('Wedging (mm)')
            ax.set_title('Disc Wedging (Anterior - Posterior)')
            ax.grid(True, alpha=0.3, axis='y')
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)

        plt.tight_layout()
        return fig

    def _draw_infinite_line(
        self,
        image: np.ndarray,
        point: np.ndarray,
        direction: np.ndarray,
        color: Tuple[int, int, int],
        thickness: int,
        H: int,
        W: int,
        label: str = ""
    ):
        """Draw an infinite line on image."""
        # Normalize direction
        direction = direction / (np.linalg.norm(direction) + 1e-8)

        # Calculate line endpoints that extend beyond image
        t_max = max(H, W) * 2
        p1 = point - direction * t_max
        p2 = point + direction * t_max

        # Convert to pixel coordinates
        p1_px = (int(p1[0] * W) if p1[0] <= 1 else int(p1[0]),
                int(p1[1] * H) if p1[1] <= 1 else int(p1[1]))
        p2_px = (int(p2[0] * W) if p2[0] <= 1 else int(p2[0]),
                int(p2[1] * H) if p2[1] <= 1 else int(p2[1]))

        # Draw line
        cv2.line(image, p1_px, p2_px, color, thickness)

        # Add label
        if label:
            label_pos = (int(point[0] * W) if point[0] <= 1 else int(point[0]),
                        int(point[1] * H) if point[1] <= 1 else int(point[1]))
            cv2.putText(
                image,
                label,
                (label_pos[0] + 10, label_pos[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

    def create_comprehensive_visualization(
        self,
        image: np.ndarray,
        vertebrae: List[VertebraKeypoints],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create comprehensive visualization with all annotations.

        Args:
            image: Input image
            vertebrae: List of vertebra keypoints
            save_path: Optional path to save result

        Returns:
            Annotated image
        """
        # Start with keypoints
        vis_image = self.draw_keypoints(image, vertebrae, show_labels=True)

        # Add alignment lines
        vis_image = self.draw_alignment_lines(vis_image, vertebrae)

        # Add measurements
        vis_image = self.draw_measurements(
            vis_image,
            vertebrae,
            show_heights=True,
            show_disc_spaces=True,
            show_cobb_angle=True
        )

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, vis_image)

        return vis_image


def create_heatmap_overlay(
    image: np.ndarray,
    heatmaps: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create overlay of keypoint heatmaps on image.

    Args:
        image: Input image (H, W, 3)
        heatmaps: Keypoint heatmaps (K, H_heat, W_heat)
        alpha: Blending factor

    Returns:
        Overlaid image
    """
    # Sum all heatmaps
    combined_heatmap = heatmaps.sum(axis=0)

    # Normalize
    combined_heatmap = (combined_heatmap - combined_heatmap.min()) / (
        combined_heatmap.max() - combined_heatmap.min() + 1e-8
    )

    # Resize to match image
    H, W = image.shape[:2]
    heatmap_resized = cv2.resize(combined_heatmap, (W, H))

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # Blend with image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    return overlay
