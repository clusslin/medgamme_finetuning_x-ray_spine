"""
Geometric measurements for spine analysis.

Includes:
- Spinal alignment lines (anterior/posterior vertebral lines)
- Intervertebral spacing
- Cobb angle measurement
- Vertebral body height
- Disc height
- Spondylolisthesis measurement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import cv2


@dataclass
class VertebraKeypoints:
    """Container for vertebra keypoints."""
    superior_anterior: Tuple[float, float]
    superior_posterior: Tuple[float, float]
    inferior_anterior: Tuple[float, float]
    inferior_posterior: Tuple[float, float]
    pedicle_left: Optional[Tuple[float, float]] = None
    pedicle_right: Optional[Tuple[float, float]] = None
    spinous_process: Optional[Tuple[float, float]] = None

    def get_corners(self) -> np.ndarray:
        """Get 4 corner points as numpy array."""
        return np.array([
            self.superior_anterior,
            self.superior_posterior,
            self.inferior_anterior,
            self.inferior_posterior
        ])

    def get_superior_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get superior endplate line."""
        return (
            np.array(self.superior_anterior),
            np.array(self.superior_posterior)
        )

    def get_inferior_line(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get inferior endplate line."""
        return (
            np.array(self.inferior_anterior),
            np.array(self.inferior_posterior)
        )


class SpineMeasurements:
    """
    Compute geometric measurements on spine X-ray images.
    """

    def __init__(self, pixel_spacing: float = 1.0):
        """
        Args:
            pixel_spacing: mm per pixel (for converting pixels to mm)
        """
        self.pixel_spacing = pixel_spacing

    def compute_cobb_angle(
        self,
        superior_vertebra: VertebraKeypoints,
        inferior_vertebra: VertebraKeypoints
    ) -> float:
        """
        Compute Cobb angle between two vertebrae.
        Used for scoliosis assessment.

        Args:
            superior_vertebra: Superior vertebra keypoints
            inferior_vertebra: Inferior vertebra keypoints

        Returns:
            Cobb angle in degrees
        """
        # Get superior endplate of superior vertebra
        sup_p1, sup_p2 = superior_vertebra.get_superior_line()
        sup_vector = sup_p2 - sup_p1

        # Get inferior endplate of inferior vertebra
        inf_p1, inf_p2 = inferior_vertebra.get_inferior_line()
        inf_vector = inf_p2 - inf_p1

        # Calculate angle between lines
        angle = self._angle_between_vectors(sup_vector, inf_vector)

        # Cobb angle is the complement if > 90 degrees
        if angle > 90:
            angle = 180 - angle

        return angle

    def compute_vertebral_body_height(
        self,
        vertebra: VertebraKeypoints,
        position: str = 'anterior'
    ) -> float:
        """
        Compute vertebral body height.

        Args:
            vertebra: Vertebra keypoints
            position: 'anterior', 'posterior', or 'mid'

        Returns:
            Height in mm
        """
        if position == 'anterior':
            # Distance from superior anterior to inferior anterior
            height_px = np.linalg.norm(
                np.array(vertebra.superior_anterior) -
                np.array(vertebra.inferior_anterior)
            )
        elif position == 'posterior':
            # Distance from superior posterior to inferior posterior
            height_px = np.linalg.norm(
                np.array(vertebra.superior_posterior) -
                np.array(vertebra.inferior_posterior)
            )
        else:  # mid
            # Average of anterior and posterior
            ant_height = self.compute_vertebral_body_height(vertebra, 'anterior')
            post_height = self.compute_vertebral_body_height(vertebra, 'posterior')
            return (ant_height + post_height) / 2

        return height_px * self.pixel_spacing

    def compute_disc_height(
        self,
        superior_vertebra: VertebraKeypoints,
        inferior_vertebra: VertebraKeypoints,
        position: str = 'anterior'
    ) -> float:
        """
        Compute intervertebral disc height.

        Args:
            superior_vertebra: Superior vertebra keypoints
            inferior_vertebra: Inferior vertebra keypoints
            position: 'anterior', 'posterior', or 'mid'

        Returns:
            Disc height in mm
        """
        if position == 'anterior':
            # Distance between inferior anterior of superior vertebra
            # and superior anterior of inferior vertebra
            p1 = np.array(superior_vertebra.inferior_anterior)
            p2 = np.array(inferior_vertebra.superior_anterior)
        elif position == 'posterior':
            p1 = np.array(superior_vertebra.inferior_posterior)
            p2 = np.array(inferior_vertebra.superior_posterior)
        else:  # mid
            # Average of anterior and posterior
            ant_height = self.compute_disc_height(
                superior_vertebra, inferior_vertebra, 'anterior'
            )
            post_height = self.compute_disc_height(
                superior_vertebra, inferior_vertebra, 'posterior'
            )
            return (ant_height + post_height) / 2

        height_px = np.linalg.norm(p2 - p1)
        return height_px * self.pixel_spacing

    def compute_anterior_vertebral_line(
        self,
        vertebrae: List[VertebraKeypoints]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute anterior vertebral line (AVL).
        Fits a line through anterior superior corners.

        Args:
            vertebrae: List of vertebra keypoints

        Returns:
            (point_on_line, direction_vector)
        """
        # Collect anterior superior points
        points = np.array([
            v.superior_anterior for v in vertebrae
        ])

        # Fit line using least squares
        return self._fit_line_2d(points)

    def compute_posterior_vertebral_line(
        self,
        vertebrae: List[VertebraKeypoints]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute posterior vertebral line (PVL).
        Fits a line through posterior superior corners.

        Args:
            vertebrae: List of vertebra keypoints

        Returns:
            (point_on_line, direction_vector)
        """
        # Collect posterior superior points
        points = np.array([
            v.superior_posterior for v in vertebrae
        ])

        # Fit line using least squares
        return self._fit_line_2d(points)

    def compute_spondylolisthesis(
        self,
        superior_vertebra: VertebraKeypoints,
        inferior_vertebra: VertebraKeypoints
    ) -> Dict[str, float]:
        """
        Compute spondylolisthesis (vertebral slip).

        Args:
            superior_vertebra: Superior vertebra keypoints
            inferior_vertebra: Inferior vertebra keypoints

        Returns:
            Dictionary with:
                - translation_mm: Horizontal translation in mm
                - translation_percent: Translation as % of vertebral width
                - grade: Meyerding grade (1-4)
        """
        # Get inferior posterior corner of superior vertebra
        sup_post = np.array(superior_vertebra.inferior_posterior)

        # Get superior endplate line of inferior vertebra
        inf_p1, inf_p2 = inferior_vertebra.get_superior_line()

        # Project superior posterior point onto inferior endplate
        projection = self._project_point_onto_line(sup_post, inf_p1, inf_p2)

        # Calculate horizontal distance (anterior-posterior translation)
        translation_px = abs(projection[0] - sup_post[0])
        translation_mm = translation_px * self.pixel_spacing

        # Calculate vertebral width of inferior vertebra
        vertebral_width_px = np.linalg.norm(inf_p2 - inf_p1)
        vertebral_width_mm = vertebral_width_px * self.pixel_spacing

        # Calculate percentage
        translation_percent = (translation_mm / vertebral_width_mm) * 100

        # Determine Meyerding grade
        if translation_percent < 25:
            grade = 1
        elif translation_percent < 50:
            grade = 2
        elif translation_percent < 75:
            grade = 3
        else:
            grade = 4

        return {
            'translation_mm': translation_mm,
            'translation_percent': translation_percent,
            'grade': grade,
            'vertebral_width_mm': vertebral_width_mm
        }

    def compute_alignment_deviation(
        self,
        vertebrae: List[VertebraKeypoints]
    ) -> Dict[str, np.ndarray]:
        """
        Compute deviation of each vertebra from ideal alignment.

        Args:
            vertebrae: List of vertebra keypoints

        Returns:
            Dictionary with:
                - anterior_deviations: Deviations from AVL (mm)
                - posterior_deviations: Deviations from PVL (mm)
        """
        # Compute alignment lines
        avl_point, avl_direction = self.compute_anterior_vertebral_line(vertebrae)
        pvl_point, pvl_direction = self.compute_posterior_vertebral_line(vertebrae)

        anterior_deviations = []
        posterior_deviations = []

        for vertebra in vertebrae:
            # Distance from anterior superior point to AVL
            ant_point = np.array(vertebra.superior_anterior)
            ant_deviation_px = self._point_to_line_distance(
                ant_point, avl_point, avl_direction
            )
            anterior_deviations.append(ant_deviation_px * self.pixel_spacing)

            # Distance from posterior superior point to PVL
            post_point = np.array(vertebra.superior_posterior)
            post_deviation_px = self._point_to_line_distance(
                post_point, pvl_point, pvl_direction
            )
            posterior_deviations.append(post_deviation_px * self.pixel_spacing)

        return {
            'anterior_deviations': np.array(anterior_deviations),
            'posterior_deviations': np.array(posterior_deviations)
        }

    def compute_intervertebral_distances(
        self,
        vertebrae: List[VertebraKeypoints]
    ) -> List[Dict[str, float]]:
        """
        Compute distances between adjacent vertebrae.

        Args:
            vertebrae: List of vertebra keypoints

        Returns:
            List of distance measurements for each adjacent pair
        """
        distances = []

        for i in range(len(vertebrae) - 1):
            superior = vertebrae[i]
            inferior = vertebrae[i + 1]

            # Anterior disc space
            ant_distance = self.compute_disc_height(superior, inferior, 'anterior')

            # Posterior disc space
            post_distance = self.compute_disc_height(superior, inferior, 'posterior')

            # Mid disc space
            mid_distance = (ant_distance + post_distance) / 2

            # Disc wedging (difference between anterior and posterior)
            wedging = ant_distance - post_distance

            distances.append({
                'anterior_mm': ant_distance,
                'posterior_mm': post_distance,
                'mid_mm': mid_distance,
                'wedging_mm': wedging,
                'wedging_angle': np.degrees(np.arctan(
                    wedging / self._get_vertebra_depth(superior)
                ))
            })

        return distances

    def compute_lordosis_kyphosis_angle(
        self,
        vertebrae: List[VertebraKeypoints]
    ) -> float:
        """
        Compute overall lordosis/kyphosis angle of a spinal segment.
        Uses Cobb method with first and last vertebra.

        Args:
            vertebrae: List of vertebra keypoints

        Returns:
            Angle in degrees (positive = lordosis, negative = kyphosis)
        """
        if len(vertebrae) < 2:
            return 0.0

        # Use first and last vertebra
        angle = self.compute_cobb_angle(vertebrae[0], vertebrae[-1])

        # Determine if lordosis or kyphosis based on direction
        # This is a simplified version - actual determination may need
        # to consider the specific region (cervical/thoracic/lumbar)

        return angle

    def _angle_between_vectors(
        self,
        v1: np.ndarray,
        v2: np.ndarray
    ) -> float:
        """Calculate angle between two vectors in degrees."""
        v1_u = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-8)
        angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        return np.degrees(angle_rad)

    def _fit_line_2d(
        self,
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a line to 2D points using least squares.

        Args:
            points: Points (N, 2)

        Returns:
            (point_on_line, direction_vector)
        """
        # Center points
        centroid = points.mean(axis=0)
        points_centered = points - centroid

        # SVD to get principal direction
        _, _, vh = np.linalg.svd(points_centered)
        direction = vh[0]

        return centroid, direction

    def _point_to_line_distance(
        self,
        point: np.ndarray,
        line_point: np.ndarray,
        line_direction: np.ndarray
    ) -> float:
        """Calculate perpendicular distance from point to line."""
        # Vector from line point to target point
        v = point - line_point

        # Project onto line direction
        projection_length = np.dot(v, line_direction)
        projection = projection_length * line_direction

        # Perpendicular component
        perpendicular = v - projection

        return np.linalg.norm(perpendicular)

    def _project_point_onto_line(
        self,
        point: np.ndarray,
        line_p1: np.ndarray,
        line_p2: np.ndarray
    ) -> np.ndarray:
        """Project a point onto a line defined by two points."""
        line_vec = line_p2 - line_p1
        point_vec = point - line_p1

        # Project
        projection_length = np.dot(point_vec, line_vec) / (
            np.linalg.norm(line_vec) ** 2 + 1e-8
        )
        projection = line_p1 + projection_length * line_vec

        return projection

    def _get_vertebra_depth(self, vertebra: VertebraKeypoints) -> float:
        """Get vertebral body depth (anterior-posterior dimension)."""
        # Average depth from superior and inferior endplates
        sup_depth = np.linalg.norm(
            np.array(vertebra.superior_posterior) -
            np.array(vertebra.superior_anterior)
        )
        inf_depth = np.linalg.norm(
            np.array(vertebra.inferior_posterior) -
            np.array(vertebra.inferior_anterior)
        )
        return (sup_depth + inf_depth) / 2 * self.pixel_spacing


def parse_keypoints_to_vertebrae(
    keypoints: np.ndarray,
    num_keypoints_per_vertebra: int = 7,
    presence: Optional[np.ndarray] = None,
    threshold: float = 0.5
) -> List[VertebraKeypoints]:
    """
    Parse flat keypoint array into VertebraKeypoints objects.

    Args:
        keypoints: Keypoints array (total_keypoints, 2)
        num_keypoints_per_vertebra: Number of keypoints per vertebra
        presence: Vertebra presence scores
        threshold: Presence threshold

    Returns:
        List of VertebraKeypoints objects
    """
    vertebrae = []
    num_vertebrae = len(keypoints) // num_keypoints_per_vertebra

    for i in range(num_vertebrae):
        # Check presence
        if presence is not None and presence[i] < threshold:
            continue

        # Extract keypoints for this vertebra
        start_idx = i * num_keypoints_per_vertebra
        kps = keypoints[start_idx:start_idx + num_keypoints_per_vertebra]

        # Create VertebraKeypoints object
        vertebra = VertebraKeypoints(
            superior_anterior=tuple(kps[0]),
            superior_posterior=tuple(kps[1]),
            inferior_anterior=tuple(kps[2]),
            inferior_posterior=tuple(kps[3]),
            pedicle_left=tuple(kps[4]) if len(kps) > 4 else None,
            pedicle_right=tuple(kps[5]) if len(kps) > 5 else None,
            spinous_process=tuple(kps[6]) if len(kps) > 6 else None
        )
        vertebrae.append(vertebra)

    return vertebrae


def create_measurement_report(
    vertebrae: List[VertebraKeypoints],
    pixel_spacing: float = 1.0,
    region: str = "spine"
) -> Dict[str, any]:
    """
    Create comprehensive measurement report.

    Args:
        vertebrae: List of vertebra keypoints
        pixel_spacing: mm per pixel
        region: Spine region (cervical/thoracic/lumbar)

    Returns:
        Dictionary with all measurements
    """
    measurements = SpineMeasurements(pixel_spacing)

    report = {
        'region': region,
        'num_vertebrae': len(vertebrae),
        'vertebral_heights': [],
        'disc_heights': [],
        'alignment': {},
        'curvature': {},
        'spondylolisthesis': []
    }

    # Vertebral body heights
    for i, vertebra in enumerate(vertebrae):
        heights = {
            'vertebra_index': i,
            'anterior_mm': measurements.compute_vertebral_body_height(
                vertebra, 'anterior'
            ),
            'posterior_mm': measurements.compute_vertebral_body_height(
                vertebra, 'posterior'
            ),
            'mid_mm': measurements.compute_vertebral_body_height(
                vertebra, 'mid'
            )
        }
        # Wedging (compression)
        heights['wedging_mm'] = heights['anterior_mm'] - heights['posterior_mm']
        report['vertebral_heights'].append(heights)

    # Intervertebral disc heights
    if len(vertebrae) >= 2:
        report['disc_heights'] = measurements.compute_intervertebral_distances(
            vertebrae
        )

    # Alignment
    if len(vertebrae) >= 3:
        alignment = measurements.compute_alignment_deviation(vertebrae)
        report['alignment'] = {
            'anterior_deviations_mm': alignment['anterior_deviations'].tolist(),
            'posterior_deviations_mm': alignment['posterior_deviations'].tolist(),
            'max_anterior_deviation_mm': float(np.max(
                np.abs(alignment['anterior_deviations'])
            )),
            'max_posterior_deviation_mm': float(np.max(
                np.abs(alignment['posterior_deviations'])
            ))
        }

    # Curvature (lordosis/kyphosis)
    if len(vertebrae) >= 2:
        report['curvature']['cobb_angle'] = measurements.compute_lordosis_kyphosis_angle(
            vertebrae
        )

    # Spondylolisthesis check
    for i in range(len(vertebrae) - 1):
        slip = measurements.compute_spondylolisthesis(
            vertebrae[i],
            vertebrae[i + 1]
        )
        if slip['grade'] > 1:  # Only report if significant
            slip['level'] = f"Vertebra {i} on {i+1}"
            report['spondylolisthesis'].append(slip)

    return report
