"""
Spine X-ray Dataset for MedGemma Finetuning
Handles multi-task learning for:
- Spine region classification (cervical, thoracic, lumbar)
- Pose classification (neutral, flexion, extension, oblique)
- Pathology detection
- Implant detection
- Report generation
"""

import os
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
import pydicom

from .preprocessing import SpineImagePreprocessor
from .augmentation import SpineAugmentation


class SpineXrayDataset(Dataset):
    """
    Dataset for spine X-ray images with multi-task annotations.

    Expected data structure:
    data_dir/
        images/
            patient_001_cervical_lateral.png
            patient_001_cervical_ap.png
            ...
        annotations.csv  # Contains labels for all tasks
        reports.json     # Contains radiology reports
        embeddings.pkl   # Pre-computed report embeddings (optional)
    """

    def __init__(
        self,
        data_dir: str,
        annotations_file: str = "annotations.csv",
        reports_file: str = "reports.json",
        embeddings_file: Optional[str] = None,
        transform: Optional[Any] = None,
        augmentation: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        max_text_length: int = 512,
        return_embeddings: bool = True,
        split: str = "train"
    ):
        """
        Args:
            data_dir: Root directory containing images and annotations
            annotations_file: CSV file with image-level annotations
            reports_file: JSON file with radiology reports
            embeddings_file: Pickle file with pre-computed embeddings
            transform: Image transformations
            augmentation: Data augmentation pipeline
            tokenizer: Tokenizer for text processing
            max_text_length: Maximum length for text sequences
            return_embeddings: Whether to return pre-computed embeddings
            split: Data split (train/val/test)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.augmentation = augmentation
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.return_embeddings = return_embeddings

        # Load annotations
        self.annotations = pd.read_csv(self.data_dir / annotations_file)

        # Load reports
        with open(self.data_dir / reports_file, 'r', encoding='utf-8') as f:
            self.reports = json.load(f)

        # Load pre-computed embeddings if available
        self.embeddings = None
        if embeddings_file and os.path.exists(self.data_dir / embeddings_file):
            with open(self.data_dir / embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

        # Define label mappings
        self.region_labels = ["cervical", "thoracic", "lumbar"]
        self.pose_labels = ["neutral", "flexion", "extension", "oblique"]
        self.pathology_labels = [
            "normal",
            "degenerative_disc_disease",
            "herniated_disc",
            "spinal_stenosis",
            "spondylolisthesis",
            "compression_fracture",
            "scoliosis",
            "ankylosing_spondylitis",
            "infection",
            "tumor"
        ]
        self.implant_labels = [
            "none",
            "screws",
            "rods",
            "cage",
            "artificial_disc",
            "bone_graft"
        ]

        # Create label to index mappings
        self.region2idx = {label: idx for idx, label in enumerate(self.region_labels)}
        self.pose2idx = {label: idx for idx, label in enumerate(self.pose_labels)}
        self.pathology2idx = {label: idx for idx, label in enumerate(self.pathology_labels)}
        self.implant2idx = {label: idx for idx, label in enumerate(self.implant_labels)}

        print(f"Loaded {len(self.annotations)} samples for {split} split")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
                - image: Preprocessed image tensor
                - region_label: Spine region label
                - pose_label: Pose type label
                - pathology_labels: Multi-label pathology tensor
                - implant_labels: Multi-label implant tensor
                - report_text: Original report text
                - report_tokens: Tokenized report
                - report_embedding: Pre-computed embedding (if available)
                - image_id: Unique image identifier
                - metadata: Additional metadata
        """
        # Get annotation row
        row = self.annotations.iloc[idx]

        # Load image
        image_path = self.data_dir / "images" / row['image_filename']
        image = self._load_image(image_path)

        # Apply transformations
        if self.augmentation and self.split == "train":
            image = self.augmentation(image=np.array(image))['image']

        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Get labels
        region_label = self.region2idx.get(row['spine_region'], 0)
        pose_label = self.pose2idx.get(row['pose'], 0)

        # Process multi-label pathologies
        pathology_labels = self._parse_multi_labels(
            row.get('pathologies', ''),
            self.pathology2idx
        )

        # Process multi-label implants
        implant_labels = self._parse_multi_labels(
            row.get('implants', ''),
            self.implant2idx
        )

        # Get report
        image_id = row['image_id']
        report_text = self.reports.get(image_id, {}).get('findings', '')

        # Tokenize report if tokenizer is available
        report_tokens = None
        if self.tokenizer:
            report_tokens = self.tokenizer(
                report_text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            # Remove batch dimension
            report_tokens = {k: v.squeeze(0) for k, v in report_tokens.items()}

        # Get pre-computed embedding if available
        report_embedding = None
        if self.return_embeddings and self.embeddings and image_id in self.embeddings:
            report_embedding = torch.tensor(self.embeddings[image_id], dtype=torch.float32)

        # Prepare metadata
        metadata = {
            'patient_id': row.get('patient_id', ''),
            'study_date': row.get('study_date', ''),
            'view': row.get('view', ''),
            'image_path': str(image_path)
        }

        sample = {
            'image': image,
            'region_label': torch.tensor(region_label, dtype=torch.long),
            'pose_label': torch.tensor(pose_label, dtype=torch.long),
            'pathology_labels': torch.tensor(pathology_labels, dtype=torch.float32),
            'implant_labels': torch.tensor(implant_labels, dtype=torch.float32),
            'report_text': report_text,
            'image_id': image_id,
            'metadata': metadata
        }

        if report_tokens is not None:
            sample['report_tokens'] = report_tokens

        if report_embedding is not None:
            sample['report_embedding'] = report_embedding

        return sample

    def _load_image(self, image_path: Path) -> Image.Image:
        """Load image from file (supports PNG, JPG, DICOM)"""
        if image_path.suffix.lower() in ['.dcm', '.dicom']:
            # Load DICOM
            dicom = pydicom.dcmread(str(image_path))
            image_array = dicom.pixel_array

            # Convert to PIL Image (handle different bit depths)
            if image_array.dtype != np.uint8:
                image_array = ((image_array - image_array.min()) /
                             (image_array.max() - image_array.min()) * 255).astype(np.uint8)

            # Convert grayscale to RGB
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)

            image = Image.fromarray(image_array)
        else:
            # Load regular image
            image = Image.open(image_path).convert('RGB')

        return image

    def _parse_multi_labels(self, label_string: str, label2idx: Dict[str, int]) -> List[int]:
        """Parse multi-label string to binary vector"""
        labels = [0] * len(label2idx)

        if pd.isna(label_string) or label_string == '':
            return labels

        # Parse comma-separated labels
        label_list = [l.strip() for l in str(label_string).split(',')]

        for label in label_list:
            if label in label2idx:
                labels[label2idx[label]] = 1

        return labels

    def get_class_weights(self, task: str = 'pathology') -> torch.Tensor:
        """
        Calculate class weights for imbalanced datasets.

        Args:
            task: One of 'region', 'pose', 'pathology', 'implant'

        Returns:
            Tensor of class weights
        """
        if task == 'region':
            label_col = 'spine_region'
            label2idx = self.region2idx
        elif task == 'pose':
            label_col = 'pose'
            label2idx = self.pose2idx
        elif task == 'pathology':
            label_col = 'pathologies'
            label2idx = self.pathology2idx
        elif task == 'implant':
            label_col = 'implants'
            label2idx = self.implant2idx
        else:
            raise ValueError(f"Unknown task: {task}")

        # Calculate class frequencies
        if task in ['pathology', 'implant']:
            # Multi-label case
            counts = np.zeros(len(label2idx))
            for _, row in self.annotations.iterrows():
                labels = self._parse_multi_labels(row[label_col], label2idx)
                counts += np.array(labels)
        else:
            # Single-label case
            counts = self.annotations[label_col].value_counts()
            counts = np.array([counts.get(label, 0) for label in label2idx.keys()])

        # Calculate weights (inverse frequency)
        total = counts.sum()
        weights = total / (len(counts) * (counts + 1))  # +1 to avoid division by zero

        return torch.tensor(weights, dtype=torch.float32)


class SpineXrayDataModule:
    """
    Data module for managing train/val/test datasets and dataloaders.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        tokenizer: Optional[Any] = None
    ):
        self.config = config
        self.tokenizer = tokenizer

        # Initialize preprocessor
        self.preprocessor = SpineImagePreprocessor(
            image_size=tuple(config['data']['image_size']),
            normalize_mean=config['data']['normalize_mean'],
            normalize_std=config['data']['normalize_std']
        )

        # Initialize augmentation
        self.augmentation = None
        if config['data']['use_augmentation']:
            self.augmentation = SpineAugmentation(
                config['data']['augmentation']
            )

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """Setup train/val/test datasets"""
        # Train dataset
        self.train_dataset = SpineXrayDataset(
            data_dir=self.config['data']['train_data_path'],
            transform=self.preprocessor.get_transform(train=True),
            augmentation=self.augmentation.get_transform() if self.augmentation else None,
            tokenizer=self.tokenizer,
            split="train"
        )

        # Validation dataset
        self.val_dataset = SpineXrayDataset(
            data_dir=self.config['data']['val_data_path'],
            transform=self.preprocessor.get_transform(train=False),
            tokenizer=self.tokenizer,
            split="val"
        )

        # Test dataset
        self.test_dataset = SpineXrayDataset(
            data_dir=self.config['data']['test_data_path'],
            transform=self.preprocessor.get_transform(train=False),
            tokenizer=self.tokenizer,
            split="test"
        )

    def train_dataloader(self):
        """Get training dataloader"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        """Get validation dataloader"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )

    def test_dataloader(self):
        """Get test dataloader"""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=True
        )
