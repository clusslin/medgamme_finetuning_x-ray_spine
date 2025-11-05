"""
Evaluation metrics for multi-task spine X-ray analysis
"""

from typing import Dict, List, Any, Optional
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    multilabel_confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


class MultiTaskMetrics:
    """
    Compute metrics for multiple tasks:
    - Region classification (cervical/thoracic/lumbar)
    - Pose classification (neutral/flexion/extension/oblique)
    - Pathology detection (multi-label)
    - Implant detection (multi-label)
    - Report generation (BLEU, ROUGE)
    """

    def __init__(
        self,
        region_labels: List[str],
        pose_labels: List[str],
        pathology_labels: List[str],
        implant_labels: List[str]
    ):
        """
        Args:
            region_labels: List of region label names
            pose_labels: List of pose label names
            pathology_labels: List of pathology label names
            implant_labels: List of implant label names
        """
        self.region_labels = region_labels
        self.pose_labels = pose_labels
        self.pathology_labels = pathology_labels
        self.implant_labels = implant_labels

    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        task_name: str = "classification"
    ) -> Dict[str, Any]:
        """
        Compute metrics for single-label classification.

        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_scores: Prediction scores (N, num_classes) for AUC
            task_name: Name of the task

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Precision, Recall, F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        metrics['precision_macro'] = precision
        metrics['recall_macro'] = recall
        metrics['f1_macro'] = f1

        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

        metrics['precision_per_class'] = precision_per_class
        metrics['recall_per_class'] = recall_per_class
        metrics['f1_per_class'] = f1_per_class
        metrics['support_per_class'] = support_per_class

        # Weighted metrics
        precision_weighted, recall_weighted, f1_weighted, _ = \
            precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

        metrics['precision_weighted'] = precision_weighted
        metrics['recall_weighted'] = recall_weighted
        metrics['f1_weighted'] = f1_weighted

        # AUC if scores provided
        if y_scores is not None:
            try:
                # One-vs-rest AUC
                metrics['auc_ovr'] = roc_auc_score(
                    y_true, y_scores, multi_class='ovr', average='macro'
                )
            except:
                metrics['auc_ovr'] = None

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

        return metrics

    def compute_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: Optional[np.ndarray] = None,
        task_name: str = "multilabel"
    ) -> Dict[str, Any]:
        """
        Compute metrics for multi-label classification.

        Args:
            y_true: True labels (N, num_classes)
            y_pred: Predicted labels (N, num_classes)
            y_scores: Prediction scores (N, num_classes)
            task_name: Name of the task

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Subset accuracy (exact match)
        metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)

        # Hamming loss (per-label accuracy)
        from sklearn.metrics import hamming_loss
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)

        # Per-label metrics
        precision_per_label, recall_per_label, f1_per_label, support_per_label = \
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )

        metrics['precision_per_label'] = precision_per_label
        metrics['recall_per_label'] = recall_per_label
        metrics['f1_per_label'] = f1_per_label
        metrics['support_per_label'] = support_per_label

        # Macro averages
        metrics['precision_macro'] = precision_per_label.mean()
        metrics['recall_macro'] = recall_per_label.mean()
        metrics['f1_macro'] = f1_per_label.mean()

        # Micro averages
        precision_micro, recall_micro, f1_micro, _ = \
            precision_recall_fscore_support(
                y_true.flatten(), y_pred.flatten(),
                average='binary', zero_division=0
            )

        metrics['precision_micro'] = precision_micro
        metrics['recall_micro'] = recall_micro
        metrics['f1_micro'] = f1_micro

        # AUC per label if scores provided
        if y_scores is not None:
            auc_per_label = []
            for i in range(y_true.shape[1]):
                try:
                    auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                    auc_per_label.append(auc)
                except:
                    auc_per_label.append(np.nan)

            metrics['auc_per_label'] = np.array(auc_per_label)
            metrics['auc_macro'] = np.nanmean(auc_per_label)

            # Average precision (PR-AUC)
            ap_per_label = []
            for i in range(y_true.shape[1]):
                try:
                    ap = average_precision_score(y_true[:, i], y_scores[:, i])
                    ap_per_label.append(ap)
                except:
                    ap_per_label.append(np.nan)

            metrics['ap_per_label'] = np.array(ap_per_label)
            metrics['map'] = np.nanmean(ap_per_label)  # Mean average precision

        # Multi-label confusion matrix
        metrics['confusion_matrix'] = multilabel_confusion_matrix(y_true, y_pred)

        return metrics

    def compute_all_metrics(
        self,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        scores: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute metrics for all tasks.

        Args:
            predictions: Dict with keys 'region', 'pose', 'pathology', 'implant'
            targets: Dict with keys 'region', 'pose', 'pathology', 'implant'
            scores: Dict with prediction scores (optional)

        Returns:
            Dictionary of metrics for each task
        """
        all_metrics = {}

        scores = scores or {}

        # Region classification
        if 'region' in predictions and 'region' in targets:
            all_metrics['region'] = self.compute_classification_metrics(
                targets['region'],
                predictions['region'],
                scores.get('region'),
                task_name='region'
            )

        # Pose classification
        if 'pose' in predictions and 'pose' in targets:
            all_metrics['pose'] = self.compute_classification_metrics(
                targets['pose'],
                predictions['pose'],
                scores.get('pose'),
                task_name='pose'
            )

        # Pathology detection (multi-label)
        if 'pathology' in predictions and 'pathology' in targets:
            all_metrics['pathology'] = self.compute_multilabel_metrics(
                targets['pathology'],
                predictions['pathology'],
                scores.get('pathology'),
                task_name='pathology'
            )

        # Implant detection (multi-label)
        if 'implant' in predictions and 'implant' in targets:
            all_metrics['implant'] = self.compute_multilabel_metrics(
                targets['implant'],
                predictions['implant'],
                scores.get('implant'),
                task_name='implant'
            )

        return all_metrics

    def print_metrics(self, metrics: Dict[str, Dict[str, Any]]):
        """Print metrics in a readable format"""
        for task_name, task_metrics in metrics.items():
            print(f"\n{'='*50}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*50}")

            # Print main metrics
            if 'accuracy' in task_metrics:
                print(f"Accuracy: {task_metrics['accuracy']:.4f}")

            if 'subset_accuracy' in task_metrics:
                print(f"Subset Accuracy (Exact Match): {task_metrics['subset_accuracy']:.4f}")

            if 'hamming_loss' in task_metrics:
                print(f"Hamming Loss: {task_metrics['hamming_loss']:.4f}")

            print(f"\nPrecision (Macro): {task_metrics.get('precision_macro', 0):.4f}")
            print(f"Recall (Macro): {task_metrics.get('recall_macro', 0):.4f}")
            print(f"F1 (Macro): {task_metrics.get('f1_macro', 0):.4f}")

            if 'auc_ovr' in task_metrics and task_metrics['auc_ovr'] is not None:
                print(f"AUC (OvR): {task_metrics['auc_ovr']:.4f}")

            if 'auc_macro' in task_metrics:
                print(f"AUC (Macro): {task_metrics['auc_macro']:.4f}")

            if 'map' in task_metrics:
                print(f"mAP: {task_metrics['map']:.4f}")

            # Print per-class/label metrics
            if 'f1_per_class' in task_metrics:
                print("\nPer-class F1 scores:")
                label_names = self._get_label_names(task_name)
                for i, (label, f1) in enumerate(zip(label_names, task_metrics['f1_per_class'])):
                    print(f"  {label}: {f1:.4f}")

            if 'f1_per_label' in task_metrics:
                print("\nPer-label F1 scores:")
                label_names = self._get_label_names(task_name)
                for i, (label, f1) in enumerate(zip(label_names, task_metrics['f1_per_label'])):
                    print(f"  {label}: {f1:.4f}")

    def _get_label_names(self, task_name: str) -> List[str]:
        """Get label names for a task"""
        if task_name == 'region':
            return self.region_labels
        elif task_name == 'pose':
            return self.pose_labels
        elif task_name == 'pathology':
            return self.pathology_labels
        elif task_name == 'implant':
            return self.implant_labels
        else:
            return []

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        labels: List[str],
        title: str = "Confusion Matrix",
        figsize: tuple = (10, 8),
        save_path: Optional[str] = None
    ):
        """
        Plot confusion matrix.

        Args:
            cm: Confusion matrix
            labels: Label names
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        """
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels
        )
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_multilabel_metrics(
        self,
        metrics: Dict[str, np.ndarray],
        labels: List[str],
        title: str = "Multi-label Metrics",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot metrics for multi-label classification.

        Args:
            metrics: Dictionary with 'precision', 'recall', 'f1', 'auc'
            labels: Label names
            title: Plot title
            figsize: Figure size
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(len(labels))
        width = 0.2

        # Plot bars
        if 'precision_per_label' in metrics:
            ax.bar(x - 1.5*width, metrics['precision_per_label'], width, label='Precision')

        if 'recall_per_label' in metrics:
            ax.bar(x - 0.5*width, metrics['recall_per_label'], width, label='Recall')

        if 'f1_per_label' in metrics:
            ax.bar(x + 0.5*width, metrics['f1_per_label'], width, label='F1')

        if 'auc_per_label' in metrics:
            ax.bar(x + 1.5*width, metrics['auc_per_label'], width, label='AUC')

        ax.set_xlabel('Labels')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")

        plt.close()


class TextGenerationMetrics:
    """
    Metrics for text generation (report generation).
    """

    @staticmethod
    def compute_bleu(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute BLEU scores.

        Args:
            references: List of reference texts
            hypotheses: List of generated texts

        Returns:
            Dictionary with BLEU-1, BLEU-2, BLEU-3, BLEU-4
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            import nltk
            nltk.download('punkt', quiet=True)
            from nltk.tokenize import word_tokenize

            # Tokenize
            refs = [[word_tokenize(ref.lower())] for ref in references]
            hyps = [word_tokenize(hyp.lower()) for hyp in hypotheses]

            # Compute BLEU with smoothing
            smoothie = SmoothingFunction().method4

            bleu1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoothie)
            bleu2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
            bleu3 = corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
            bleu4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

            return {
                'bleu-1': bleu1,
                'bleu-2': bleu2,
                'bleu-3': bleu3,
                'bleu-4': bleu4
            }
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            return {'bleu-1': 0, 'bleu-2': 0, 'bleu-3': 0, 'bleu-4': 0}

    @staticmethod
    def compute_rouge(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """
        Compute ROUGE scores.

        Args:
            references: List of reference texts
            hypotheses: List of generated texts

        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L
        """
        try:
            from rouge_score import rouge_scorer

            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []

            for ref, hyp in zip(references, hypotheses):
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)

            return {
                'rouge-1': np.mean(rouge1_scores),
                'rouge-2': np.mean(rouge2_scores),
                'rouge-L': np.mean(rougeL_scores)
            }
        except Exception as e:
            print(f"Error computing ROUGE: {e}")
            return {'rouge-1': 0, 'rouge-2': 0, 'rouge-L': 0}
