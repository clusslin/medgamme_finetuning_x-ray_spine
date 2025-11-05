#!/usr/bin/env python3
"""
Evaluation script for MedGemma spine X-ray model
"""

import argparse
import sys
from pathlib import Path
import yaml
import json

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.medgemma_model import MedGemmaSpineModel
from src.data.dataset import SpineXrayDataModule
from src.utils.metrics import MultiTaskMetrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma spine X-ray model")
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
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, metrics_calculator, device):
    """Evaluate model on dataset"""
    model.eval()

    # Collect predictions and targets
    all_predictions = {
        'region': [],
        'pose': [],
        'pathology': [],
        'implant': []
    }
    all_targets = {
        'region': [],
        'pose': [],
        'pathology': [],
        'implant': []
    }
    all_scores = {
        'region': [],
        'pose': [],
        'pathology': [],
        'implant': []
    }

    total_loss = 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        outputs = model(
            input_ids=batch.get('report_tokens', {}).get('input_ids'),
            attention_mask=batch.get('report_tokens', {}).get('attention_mask'),
            image_features=batch.get('image'),
            labels={
                'region': batch['region_label'],
                'pose': batch['pose_label'],
                'pathology': batch['pathology_labels'],
                'implant': batch['implant_labels']
            }
        )

        total_loss += outputs['total_loss'].item()

        # Collect predictions - Region
        region_preds = torch.argmax(outputs['region_logits'], dim=1)
        all_predictions['region'].append(region_preds.cpu().numpy())
        all_targets['region'].append(batch['region_label'].cpu().numpy())
        all_scores['region'].append(torch.softmax(outputs['region_logits'], dim=1).cpu().numpy())

        # Pose
        pose_preds = torch.argmax(outputs['pose_logits'], dim=1)
        all_predictions['pose'].append(pose_preds.cpu().numpy())
        all_targets['pose'].append(batch['pose_label'].cpu().numpy())
        all_scores['pose'].append(torch.softmax(outputs['pose_logits'], dim=1).cpu().numpy())

        # Pathology (multi-label)
        pathology_preds = (torch.sigmoid(outputs['pathology_logits']) > 0.5).float()
        all_predictions['pathology'].append(pathology_preds.cpu().numpy())
        all_targets['pathology'].append(batch['pathology_labels'].cpu().numpy())
        all_scores['pathology'].append(torch.sigmoid(outputs['pathology_logits']).cpu().numpy())

        # Implant (multi-label)
        implant_preds = (torch.sigmoid(outputs['implant_logits']) > 0.5).float()
        all_predictions['implant'].append(implant_preds.cpu().numpy())
        all_targets['implant'].append(batch['implant_labels'].cpu().numpy())
        all_scores['implant'].append(torch.sigmoid(outputs['implant_logits']).cpu().numpy())

    # Concatenate all predictions
    for key in all_predictions:
        all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
        all_targets[key] = np.concatenate(all_targets[key], axis=0)
        all_scores[key] = np.concatenate(all_scores[key], axis=0)

    # Calculate metrics
    task_metrics = metrics_calculator.compute_all_metrics(
        all_predictions,
        all_targets,
        all_scores
    )

    # Add average loss
    task_metrics['overall'] = {
        'average_loss': total_loss / len(dataloader)
    }

    return task_metrics


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

    # Setup data
    print("\nLoading data...")
    tokenizer = model.tokenizer
    data_module = SpineXrayDataModule(config, tokenizer)
    data_module.setup()

    # Get dataloader for specified split
    if args.split == "train":
        dataloader = data_module.train_dataloader()
        dataset_size = len(data_module.train_dataset)
    elif args.split == "val":
        dataloader = data_module.val_dataloader()
        dataset_size = len(data_module.val_dataset)
    else:  # test
        dataloader = data_module.test_dataloader()
        dataset_size = len(data_module.test_dataset)

    print(f"Evaluating on {args.split} split: {dataset_size} samples")

    # Setup metrics
    metrics_calculator = MultiTaskMetrics(
        region_labels=config['data']['spine_regions'],
        pose_labels=config['data']['pose_types'],
        pathology_labels=config['data']['pathologies'],
        implant_labels=config['data']['implants']
    )

    # Evaluate
    print("\nRunning evaluation...")
    metrics = evaluate(model, dataloader, metrics_calculator, device)

    # Print metrics
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    metrics_calculator.print_metrics(metrics)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics as JSON
    metrics_json = {}
    for task_name, task_metrics in metrics.items():
        metrics_json[task_name] = {}
        for key, value in task_metrics.items():
            if isinstance(value, (int, float, str, bool)):
                metrics_json[task_name][key] = value
            elif isinstance(value, np.ndarray):
                metrics_json[task_name][key] = value.tolist()

    output_file = output_dir / f"metrics_{args.split}.json"
    with open(output_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Plot confusion matrices
    for task_name, task_metrics in metrics.items():
        if 'confusion_matrix' in task_metrics and task_name != 'overall':
            cm = task_metrics['confusion_matrix']

            if task_name in ['region', 'pose']:
                # Single-label classification
                labels = config['data'][f'spine_regions' if task_name == 'region' else 'pose_types']
                metrics_calculator.plot_confusion_matrix(
                    cm,
                    labels,
                    title=f"{task_name.capitalize()} Confusion Matrix",
                    save_path=output_dir / f"confusion_matrix_{task_name}_{args.split}.png"
                )
            else:
                # Multi-label - skip for now (would need different visualization)
                pass

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
