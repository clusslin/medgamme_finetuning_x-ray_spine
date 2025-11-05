#!/usr/bin/env python3
"""
Main training script for MedGemma spine X-ray finetuning
"""

import argparse
import sys
from pathlib import Path
import yaml

import torch
from accelerate import Accelerator

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.medgemma_model import MedGemmaSpineModel
from src.data.dataset import SpineXrayDataModule
from src.training.trainer import SpineXrayTrainer
from src.utils.embeddings import ReportEmbedder


def parse_args():
    parser = argparse.ArgumentParser(description="Train MedGemma for spine X-ray analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override output dir if specified
    if args.output_dir:
        config['output']['checkpoint_dir'] = args.output_dir

    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision='bf16' if config['training']['bf16'] else 'fp16' if config['training']['fp16'] else 'no',
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
    )

    accelerator.print("=" * 50)
    accelerator.print("MedGemma Spine X-ray Finetuning")
    accelerator.print("=" * 50)
    accelerator.print(f"Config: {args.config}")
    accelerator.print(f"Device: {accelerator.device}")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Num processes: {accelerator.num_processes}")

    # Initialize model
    accelerator.print("\nInitializing model...")
    model = MedGemmaSpineModel(
        model_name=config['model']['name'],
        num_regions=len(config['data']['spine_regions']),
        num_poses=len(config['data']['pose_types']),
        num_pathologies=len(config['data']['pathologies']),
        num_implants=len(config['data']['implants']),
        use_lora=config['model']['use_lora'],
        lora_config={
            'r': config['model']['lora_r'],
            'lora_alpha': config['model']['lora_alpha'],
            'lora_dropout': config['model']['lora_dropout'],
            'target_modules': config['model']['lora_target_modules']
        },
        load_in_4bit=config['model']['load_in_4bit'],
        load_in_8bit=config['model']['load_in_8bit']
    )

    # Setup tokenizer
    tokenizer = model.tokenizer

    # Setup data module
    accelerator.print("\nSetting up data...")
    data_module = SpineXrayDataModule(config, tokenizer)
    data_module.setup()

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    accelerator.print(f"Training samples: {len(data_module.train_dataset)}")
    accelerator.print(f"Validation samples: {len(data_module.val_dataset)}")

    # Initialize trainer
    accelerator.print("\nInitializing trainer...")
    trainer = SpineXrayTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        accelerator=accelerator
    )

    # Load checkpoint if specified
    if args.checkpoint:
        accelerator.print(f"\nLoading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Start training
    accelerator.print("\nStarting training...")
    trainer.train()

    accelerator.print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
