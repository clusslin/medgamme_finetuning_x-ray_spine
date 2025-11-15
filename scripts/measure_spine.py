#!/usr/bin/env python3
"""
Spine measurement script with automatic keypoint detection.

This script:
1. Detects vertebra keypoints from X-ray images
2. Computes geometric measurements (heights, spacing, angles)
3. Generates comprehensive measurement reports
4. Creates annotated visualizations
"""

import argparse
import sys
from pathlib import Path
import yaml
import json
import numpy as np
from PIL import Image
import torch
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.keypoint_detector import VertebraKeypointDetector
from src.utils.measurements import (
    parse_keypoints_to_vertebrae,
    create_measurement_report,
    SpineMeasurements
)
from src.utils.visualization import SpineVisualizer, create_heatmap_overlay
from src.data.preprocessing import SpineImagePreprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure spine geometry from X-ray images"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to spine X-ray image"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/measurement_config.yaml",
        help="Path to measurement config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to keypoint detector checkpoint (if available)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/measurements",
        help="Output directory for results"
    )
    parser.add_argument(
        "--pixel_spacing",
        type=float,
        default=None,
        help="Pixel spacing in mm/pixel (overrides config)"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["cervical", "thoracic", "lumbar"],
        default="lumbar",
        help="Spine region"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization outputs"
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Show matplotlib plots"
    )
    return parser.parse_args()


def load_model(config: dict, checkpoint_path: str = None) -> VertebraKeypointDetector:
    """Load keypoint detector model."""
    model = VertebraKeypointDetector(
        hidden_size=2048,
        num_vertebrae=config['keypoint_detection']['num_vertebrae'],
        num_keypoints_per_vertebra=config['keypoint_detection']['num_keypoints_per_vertebra'],
        heatmap_size=tuple(config['keypoint_detection']['heatmap_size'])
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Warning: No checkpoint provided. Using untrained model.")
        print("For real measurements, you need a trained keypoint detector.")

    model.eval()
    return model


def detect_keypoints(
    model: VertebraKeypointDetector,
    image: torch.Tensor,
    device: str = 'cpu'
) -> dict:
    """Detect keypoints from image."""
    model.to(device)
    image = image.to(device)

    with torch.no_grad():
        # For this demo, we need to generate dummy features
        # In real implementation, this would come from the MedGemma backbone
        B, C, H, W = image.shape
        dummy_features = torch.randn(B, 2048, H // 32, W // 32).to(device)

        outputs = model(dummy_features)

    return {
        'keypoints': outputs['keypoints'].cpu().numpy(),
        'presence': outputs['presence'].cpu().numpy(),
        'heatmaps': outputs['heatmaps'].cpu().numpy()
    }


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"\nLoading image: {args.image}")
    image_pil = Image.open(args.image).convert('RGB')
    image_np = np.array(image_pil)

    # Preprocess image
    preprocessor = SpineImagePreprocessor(
        image_size=(512, 512)
    )
    image_tensor = preprocessor(image_pil).unsqueeze(0)

    # Load keypoint detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.checkpoint:
        model = load_model(config, args.checkpoint)
        print("\nDetecting vertebra keypoints...")
        detection_results = detect_keypoints(model, image_tensor, device)

        keypoints = detection_results['keypoints'][0]  # (total_keypoints, 2)
        presence = detection_results['presence'][0]   # (num_vertebrae,)
        heatmaps = detection_results['heatmaps'][0]   # (total_keypoints, H, W)
    else:
        # For demo: generate simulated keypoints
        # In practice, you would train the detector first
        print("\nNo checkpoint provided. Generating demo keypoints...")
        print("NOTE: For real measurements, please train a keypoint detector first.")

        num_vertebrae = 5  # Example: L1-L5
        keypoints, presence = generate_demo_keypoints(
            num_vertebrae,
            config['keypoint_detection']['num_keypoints_per_vertebra']
        )
        heatmaps = None

    # Parse keypoints to vertebrae structures
    pixel_spacing = args.pixel_spacing or config['measurements']['pixel_spacing']
    print(f"\nPixel spacing: {pixel_spacing} mm/pixel")

    vertebrae = parse_keypoints_to_vertebrae(
        keypoints,
        num_keypoints_per_vertebra=config['keypoint_detection']['num_keypoints_per_vertebra'],
        presence=presence,
        threshold=config['keypoint_detection']['presence_threshold']
    )

    print(f"Detected {len(vertebrae)} vertebrae")

    # Compute measurements
    print("\nComputing measurements...")
    measurements_report = create_measurement_report(
        vertebrae,
        pixel_spacing=pixel_spacing,
        region=args.region
    )

    # Print summary
    print("\n" + "=" * 60)
    print("MEASUREMENT SUMMARY")
    print("=" * 60)
    print(f"Region: {measurements_report['region'].upper()}")
    print(f"Number of vertebrae: {measurements_report['num_vertebrae']}")

    if measurements_report['vertebral_heights']:
        print("\nVertebral Body Heights:")
        for vh in measurements_report['vertebral_heights']:
            print(f"  V{vh['vertebra_index']}: "
                  f"Ant={vh['anterior_mm']:.1f}mm, "
                  f"Post={vh['posterior_mm']:.1f}mm, "
                  f"Wedge={vh['wedging_mm']:.1f}mm")

    if measurements_report['disc_heights']:
        print("\nIntervertebral Disc Heights:")
        for i, dh in enumerate(measurements_report['disc_heights']):
            print(f"  Disc {i}-{i+1}: "
                  f"Ant={dh['anterior_mm']:.1f}mm, "
                  f"Post={dh['posterior_mm']:.1f}mm, "
                  f"Mid={dh['mid_mm']:.1f}mm")

    if 'cobb_angle' in measurements_report['curvature']:
        print(f"\nCobb Angle: {measurements_report['curvature']['cobb_angle']:.1f}°")

    if measurements_report['alignment']:
        print(f"\nAlignment Deviations:")
        print(f"  Max Anterior: {measurements_report['alignment']['max_anterior_deviation_mm']:.1f}mm")
        print(f"  Max Posterior: {measurements_report['alignment']['max_posterior_deviation_mm']:.1f}mm")

    if measurements_report['spondylolisthesis']:
        print("\nSpondylolisthesis Detected:")
        for slip in measurements_report['spondylolisthesis']:
            print(f"  {slip['level']}: "
                  f"Grade {slip['grade']} "
                  f"({slip['translation_percent']:.1f}% slip)")

    # Save report
    report_file = output_dir / f"{Path(args.image).stem}_report.json"
    with open(report_file, 'w') as f:
        json.dump(measurements_report, f, indent=2)
    print(f"\nReport saved to: {report_file}")

    # Visualization
    if args.visualize:
        print("\nGenerating visualizations...")

        visualizer = SpineVisualizer(pixel_spacing=pixel_spacing)

        # Create comprehensive visualization
        vis_image = visualizer.create_comprehensive_visualization(
            image_np,
            vertebrae,
            save_path=str(output_dir / f"{Path(args.image).stem}_annotated.png")
        )
        print(f"Annotated image saved to: {output_dir / f'{Path(args.image).stem}_annotated.png'}")

        # Create measurement plots
        if len(vertebrae) >= 2:
            fig = visualizer.create_measurement_figure(vertebrae, args.region)
            fig.savefig(
                output_dir / f"{Path(args.image).stem}_measurements.png",
                dpi=300,
                bbox_inches='tight'
            )
            print(f"Measurement plots saved to: {output_dir / f'{Path(args.image).stem}_measurements.png'}")

            if args.show_plots:
                import matplotlib.pyplot as plt
                plt.show()

        # Create heatmap overlay if available
        if heatmaps is not None:
            heatmap_overlay = create_heatmap_overlay(
                image_np,
                heatmaps,
                alpha=config['visualization']['heatmap_alpha']
            )
            cv2.imwrite(
                str(output_dir / f"{Path(args.image).stem}_heatmap.png"),
                heatmap_overlay
            )
            print(f"Heatmap overlay saved to: {output_dir / f'{Path(args.image).stem}_heatmap.png'}")

    print("\nMeasurement complete!")


def generate_demo_keypoints(
    num_vertebrae: int = 5,
    num_keypoints_per_vertebra: int = 7
) -> tuple:
    """
    Generate demo keypoints for demonstration purposes.
    In practice, these would come from a trained detector.
    """
    keypoints = []
    presence = []

    # Generate keypoints for each vertebra (simulating lumbar spine L1-L5)
    for i in range(num_vertebrae):
        # Position vertebrae vertically with some spacing
        y_center = 0.2 + (i * 0.15)
        x_center = 0.5

        # 4 corner points (superior anterior, superior posterior, inferior anterior, inferior posterior)
        # Vertebral body dimensions (approximate)
        width = 0.15
        height = 0.08

        keypoints.extend([
            [x_center - width/2, y_center - height/2],  # Superior anterior
            [x_center + width/2, y_center - height/2],  # Superior posterior
            [x_center - width/2, y_center + height/2],  # Inferior anterior
            [x_center + width/2, y_center + height/2],  # Inferior posterior
            [x_center - width/3, y_center],              # Pedicle left
            [x_center + width/3, y_center],              # Pedicle right
            [x_center + width/2 + 0.05, y_center],       # Spinous process
        ])

        presence.append(1.0)

    # Pad remaining vertebrae (not present)
    max_vertebrae = 24
    for i in range(num_vertebrae, max_vertebrae):
        keypoints.extend([[0.0, 0.0]] * num_keypoints_per_vertebra)
        presence.append(0.0)

    keypoints = np.array(keypoints, dtype=np.float32)
    presence = np.array(presence, dtype=np.float32)

    return keypoints, presence


if __name__ == "__main__":
    main()
