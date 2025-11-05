#!/usr/bin/env python3
"""
Inference script for MedGemma spine X-ray model
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

import torch
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.medgemma_model import MedGemmaSpineModel
from src.data.preprocessing import SpineImagePreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with MedGemma spine X-ray model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input X-ray image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Analyze this spine X-ray image and provide a detailed radiology report.",
        help="Prompt for report generation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--generate_report",
        action="store_true",
        help="Generate full radiology report"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = MedGemmaSpineModel.from_pretrained(
        args.checkpoint,
        num_regions=len(config['data']['spine_regions']),
        num_poses=len(config['data']['pose_types']),
        num_pathologies=len(config['data']['pathologies']),
        num_implants=len(config['data']['implants'])
    )
    model.to(device)
    model.eval()

    # Setup preprocessor
    preprocessor = SpineImagePreprocessor(
        image_size=tuple(config['data']['image_size']),
        normalize_mean=config['data']['normalize_mean'],
        normalize_std=config['data']['normalize_std']
    )

    # Load and preprocess image
    print(f"\nLoading image: {args.image}")
    image = Image.open(args.image).convert('RGB')
    image_tensor = preprocessor(image).unsqueeze(0).to(device)

    # Prepare prompt
    tokenizer = model.tokenizer
    inputs = tokenizer(
        args.prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Get classification predictions
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_features=image_tensor
        )

        # Region prediction
        region_probs = torch.softmax(outputs['region_logits'], dim=1)[0]
        region_pred = torch.argmax(region_probs).item()
        region_label = config['data']['spine_regions'][region_pred]

        # Pose prediction
        pose_probs = torch.softmax(outputs['pose_logits'], dim=1)[0]
        pose_pred = torch.argmax(pose_probs).item()
        pose_label = config['data']['pose_types'][pose_pred]

        # Pathology predictions (multi-label)
        pathology_probs = torch.sigmoid(outputs['pathology_logits'])[0]
        pathology_preds = (pathology_probs > 0.5).cpu().numpy()
        detected_pathologies = [
            config['data']['pathologies'][i]
            for i, pred in enumerate(pathology_preds)
            if pred
        ]

        # Implant predictions (multi-label)
        implant_probs = torch.sigmoid(outputs['implant_logits'])[0]
        implant_preds = (implant_probs > 0.5).cpu().numpy()
        detected_implants = [
            config['data']['implants'][i]
            for i, pred in enumerate(implant_preds)
            if pred
        ]

    # Prepare results
    results = {
        'image_path': args.image,
        'spine_region': {
            'prediction': region_label,
            'confidence': float(region_probs[region_pred]),
            'all_probabilities': {
                label: float(prob)
                for label, prob in zip(config['data']['spine_regions'], region_probs.cpu().numpy())
            }
        },
        'pose': {
            'prediction': pose_label,
            'confidence': float(pose_probs[pose_pred]),
            'all_probabilities': {
                label: float(prob)
                for label, prob in zip(config['data']['pose_types'], pose_probs.cpu().numpy())
            }
        },
        'pathologies': {
            'detected': detected_pathologies,
            'probabilities': {
                label: float(prob)
                for label, prob in zip(config['data']['pathologies'], pathology_probs.cpu().numpy())
            }
        },
        'implants': {
            'detected': detected_implants,
            'probabilities': {
                label: float(prob)
                for label, prob in zip(config['data']['implants'], implant_probs.cpu().numpy())
            }
        }
    }

    # Generate report if requested
    if args.generate_report:
        print("\nGenerating radiology report...")
        report = model.generate_report(
            prompt=args.prompt,
            image_features=image_tensor,
            max_length=512
        )
        results['generated_report'] = report

    # Print results
    print("\n" + "=" * 50)
    print("ANALYSIS RESULTS")
    print("=" * 50)
    print(f"\nSpine Region: {region_label} (confidence: {region_probs[region_pred]:.3f})")
    print(f"Pose: {pose_label} (confidence: {pose_probs[pose_pred]:.3f})")

    if detected_pathologies:
        print(f"\nDetected Pathologies:")
        for path in detected_pathologies:
            idx = config['data']['pathologies'].index(path)
            print(f"  - {path} (probability: {pathology_probs[idx]:.3f})")
    else:
        print("\nNo pathologies detected")

    if detected_implants:
        print(f"\nDetected Implants:")
        for imp in detected_implants:
            idx = config['data']['implants'].index(imp)
            print(f"  - {imp} (probability: {implant_probs[idx]:.3f})")
    else:
        print("\nNo implants detected")

    if args.generate_report:
        print(f"\nGenerated Report:")
        print("-" * 50)
        print(results['generated_report'])
        print("-" * 50)

    # Save results if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {output_path}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
