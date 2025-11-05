"""
Trainer for MedGemma spine X-ray multi-task learning
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb
from accelerate import Accelerator

from ..models.medgemma_model import MedGemmaSpineModel
from ..utils.metrics import MultiTaskMetrics
from ..utils.embeddings import ReportEmbedder


class SpineXrayTrainer:
    """
    Trainer for multi-task spine X-ray analysis.

    Handles:
    - Multi-task learning with weighted losses
    - LoRA finetuning
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Logging (wandb, tensorboard)
    """

    def __init__(
        self,
        model: MedGemmaSpineModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict[str, Any],
        accelerator: Optional[Accelerator] = None
    ):
        """
        Args:
            model: MedGemma model
            train_dataloader: Training dataloader
            val_dataloader: Validation dataloader
            config: Training configuration
            accelerator: HuggingFace Accelerator for distributed training
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config

        # Setup accelerator
        if accelerator is None:
            self.accelerator = Accelerator(
                mixed_precision='bf16' if config['training']['bf16'] else 'fp16' if config['training']['fp16'] else 'no',
                gradient_accumulation_steps=config['training']['gradient_accumulation_steps']
            )
        else:
            self.accelerator = accelerator

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.val_dataloader
            )

        # Task weights for multi-task learning
        self.task_weights = config['training']['task_weights']

        # Setup metrics
        self.metrics = MultiTaskMetrics(
            region_labels=config['data']['spine_regions'],
            pose_labels=config['data']['pose_types'],
            pathology_labels=config['data']['pathologies'],
            implant_labels=config['data']['implants']
        )

        # Tracking
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Setup logging
        self._setup_logging()

        # Output directories
        self.checkpoint_dir = Path(config['output']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        # Separate parameters for different learning rates if needed
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        return optimizer

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        num_training_steps = len(self.train_dataloader) * self.config['training']['num_epochs']
        warmup_steps = self.config['training']['warmup_steps']

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - warmup_steps,
            eta_min=1e-6
        )

        # Sequential scheduler (warmup then cosine)
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        return scheduler

    def _setup_logging(self):
        """Setup logging (wandb)"""
        if self.config['wandb']['enabled'] and self.accelerator.is_main_process:
            wandb.init(
                project=self.config['wandb']['project'],
                entity=self.config['wandb']['entity'],
                config=self.config,
                tags=self.config['wandb']['tags']
            )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        task_losses = {
            'region_loss': 0,
            'pose_loss': 0,
            'pathology_loss': 0,
            'implant_loss': 0
        }

        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.epoch}",
            disable=not self.accelerator.is_main_process
        )

        for batch_idx, batch in enumerate(pbar):
            with self.accelerator.accumulate(self.model):
                # Forward pass
                outputs = self.model(
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

                # Calculate weighted loss
                losses = outputs['losses']
                weighted_loss = sum(
                    losses[task] * self.task_weights.get(task.replace('_loss', '_classification'), 1.0)
                    for task in losses
                )

                # Backward pass
                self.accelerator.backward(weighted_loss)

                # Gradient clipping
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Update metrics
                total_loss += weighted_loss.item()
                for task, loss_value in losses.items():
                    if task in task_losses:
                        task_losses[task] += loss_value.item()

                # Update progress bar
                if self.accelerator.is_main_process:
                    pbar.set_postfix({
                        'loss': weighted_loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr']
                    })

                # Logging
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    self._log_metrics({
                        'train/loss': weighted_loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        **{f'train/{k}': v.item() for k, v in losses.items()}
                    })

                self.global_step += 1

                # Validation
                if self.global_step % self.config['training']['eval_steps'] == 0:
                    val_metrics = self.validate()
                    self.model.train()

                    # Save checkpoint if best
                    if val_metrics['val/total_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val/total_loss']
                        self.save_checkpoint(is_best=True)
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    # Early stopping
                    if self.patience_counter >= self.config['training']['early_stopping_patience']:
                        self.accelerator.print("Early stopping triggered")
                        return None

                # Regular checkpoint
                if self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint()

        # Average losses
        num_batches = len(self.train_dataloader)
        avg_metrics = {
            'train/epoch_loss': total_loss / num_batches,
            **{f'train/epoch_{k}': v / num_batches for k, v in task_losses.items()}
        }

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()

        total_loss = 0
        task_losses = {
            'region_loss': 0,
            'pose_loss': 0,
            'pathology_loss': 0,
            'implant_loss': 0
        }

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

        for batch in tqdm(
            self.val_dataloader,
            desc="Validation",
            disable=not self.accelerator.is_main_process
        ):
            # Forward pass
            outputs = self.model(
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

            # Calculate loss
            losses = outputs['losses']
            weighted_loss = sum(
                losses[task] * self.task_weights.get(task.replace('_loss', '_classification'), 1.0)
                for task in losses
            )

            total_loss += weighted_loss.item()
            for task, loss_value in losses.items():
                if task in task_losses:
                    task_losses[task] += loss_value.item()

            # Collect predictions
            # Region
            region_preds = torch.argmax(outputs['region_logits'], dim=1)
            all_predictions['region'].append(self.accelerator.gather(region_preds).cpu().numpy())
            all_targets['region'].append(self.accelerator.gather(batch['region_label']).cpu().numpy())
            all_scores['region'].append(
                self.accelerator.gather(torch.softmax(outputs['region_logits'], dim=1)).cpu().numpy()
            )

            # Pose
            pose_preds = torch.argmax(outputs['pose_logits'], dim=1)
            all_predictions['pose'].append(self.accelerator.gather(pose_preds).cpu().numpy())
            all_targets['pose'].append(self.accelerator.gather(batch['pose_label']).cpu().numpy())
            all_scores['pose'].append(
                self.accelerator.gather(torch.softmax(outputs['pose_logits'], dim=1)).cpu().numpy()
            )

            # Pathology (multi-label)
            pathology_preds = (torch.sigmoid(outputs['pathology_logits']) > 0.5).float()
            all_predictions['pathology'].append(self.accelerator.gather(pathology_preds).cpu().numpy())
            all_targets['pathology'].append(self.accelerator.gather(batch['pathology_labels']).cpu().numpy())
            all_scores['pathology'].append(
                self.accelerator.gather(torch.sigmoid(outputs['pathology_logits'])).cpu().numpy()
            )

            # Implant (multi-label)
            implant_preds = (torch.sigmoid(outputs['implant_logits']) > 0.5).float()
            all_predictions['implant'].append(self.accelerator.gather(implant_preds).cpu().numpy())
            all_targets['implant'].append(self.accelerator.gather(batch['implant_labels']).cpu().numpy())
            all_scores['implant'].append(
                self.accelerator.gather(torch.sigmoid(outputs['implant_logits'])).cpu().numpy()
            )

        # Concatenate all predictions
        import numpy as np
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key], axis=0)
            all_targets[key] = np.concatenate(all_targets[key], axis=0)
            all_scores[key] = np.concatenate(all_scores[key], axis=0)

        # Calculate metrics
        if self.accelerator.is_main_process:
            task_metrics = self.metrics.compute_all_metrics(
                all_predictions,
                all_targets,
                all_scores
            )

            # Print metrics
            self.metrics.print_metrics(task_metrics)

        # Average losses
        num_batches = len(self.val_dataloader)
        val_metrics = {
            'val/total_loss': total_loss / num_batches,
            **{f'val/{k}': v / num_batches for k, v in task_losses.items()}
        }

        # Add task metrics to logging
        if self.accelerator.is_main_process:
            for task_name, metrics in task_metrics.items():
                val_metrics[f'val/{task_name}_f1_macro'] = metrics.get('f1_macro', 0)
                val_metrics[f'val/{task_name}_accuracy'] = metrics.get('accuracy', metrics.get('subset_accuracy', 0))

        self._log_metrics(val_metrics)

        return val_metrics

    def train(self):
        """Main training loop"""
        self.accelerator.print(f"Starting training for {self.config['training']['num_epochs']} epochs")

        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch

            # Train epoch
            train_metrics = self.train_epoch()

            if train_metrics is None:  # Early stopping
                break

            # Log epoch metrics
            self._log_metrics(train_metrics)

            # Validate at end of epoch
            val_metrics = self.validate()

            # Save checkpoint
            if val_metrics['val/total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val/total_loss']
                self.save_checkpoint(is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping check
            if self.patience_counter >= self.config['training']['early_stopping_patience']:
                self.accelerator.print("Early stopping triggered")
                break

        self.accelerator.print("Training completed!")

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return

        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save checkpoint
        checkpoint_name = f"checkpoint_step_{self.global_step}"
        if is_best:
            checkpoint_name = "best_model"

        save_path = self.checkpoint_dir / checkpoint_name
        save_path.mkdir(exist_ok=True)

        # Save model
        unwrapped_model.save_pretrained(str(save_path))

        # Save training state
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, save_path / 'training_state.pt')

        self.accelerator.print(f"Checkpoint saved to {save_path}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the best and recent ones"""
        save_total_limit = self.config['training']['save_total_limit']

        # Get all checkpoints
        checkpoints = sorted(
            [d for d in self.checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith('checkpoint_')],
            key=lambda x: int(x.name.split('_')[-1])
        )

        # Remove old checkpoints
        if len(checkpoints) > save_total_limit:
            for checkpoint in checkpoints[:-save_total_limit]:
                import shutil
                shutil.rmtree(checkpoint)
                self.accelerator.print(f"Removed old checkpoint: {checkpoint}")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb/tensorboard"""
        if not self.accelerator.is_main_process:
            return

        if self.config['wandb']['enabled']:
            wandb.log(metrics, step=self.global_step)

    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        # Load training state
        training_state = torch.load(
            checkpoint_path / 'training_state.pt',
            map_location=self.accelerator.device
        )

        self.epoch = training_state['epoch']
        self.global_step = training_state['global_step']
        self.best_val_loss = training_state['best_val_loss']
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(training_state['scheduler_state_dict'])

        self.accelerator.print(f"Checkpoint loaded from {checkpoint_path}")
