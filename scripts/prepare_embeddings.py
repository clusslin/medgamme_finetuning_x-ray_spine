#!/usr/bin/env python3
"""
Prepare report embeddings for training
"""

import argparse
import sys
from pathlib import Path
import json
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.embeddings import ReportEmbedder, MedicalVocabularyBuilder


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare report embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--reports_file",
        type=str,
        required=True,
        help="Path to reports JSON file"
    )
    parser.add_argument(
        "--annotations_file",
        type=str,
        default=None,
        help="Path to annotations CSV file (for domain adaptation)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--train_embedder",
        action="store_true",
        help="Train embedder on domain data"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reports
    print(f"Loading reports from {args.reports_file}")
    with open(args.reports_file, 'r', encoding='utf-8') as f:
        reports = json.load(f)

    report_texts = [report.get('findings', '') for report in reports.values()]
    image_ids = list(reports.keys())

    print(f"Loaded {len(report_texts)} reports")

    # Build medical vocabulary
    print("\nBuilding medical vocabulary...")
    vocab_builder = MedicalVocabularyBuilder(
        medical_terms_file=config['embeddings'].get('medical_terms_file')
    )

    vocabulary = vocab_builder.build_vocabulary(report_texts, min_frequency=2)

    # Save vocabulary
    vocab_file = output_dir / "medical_vocabulary.txt"
    vocab_builder.save_vocabulary(str(vocab_file))

    # Print top terms
    print("\nTop 20 medical terms:")
    for term, freq in vocab_builder.get_top_terms(20):
        print(f"  {term}: {freq}")

    # Initialize embedder
    print("\nInitializing report embedder...")
    embedder = ReportEmbedder(
        model_name=config['embeddings']['model_name'],
        embedding_dim=config['embeddings']['embedding_dim'],
        max_seq_length=config['embeddings']['max_seq_length']
    )

    # Train embedder on domain if requested
    if args.train_embedder and args.annotations_file:
        print("\nTraining embedder on domain data...")

        # Load annotations
        annotations_df = pd.read_csv(args.annotations_file)

        # Create annotations dict
        annotations_dict = {}
        for _, row in annotations_df.iterrows():
            image_id = row['image_id']
            pathologies = row.get('pathologies', '').split(',') if pd.notna(row.get('pathologies')) else []
            annotations_dict[image_id] = {'pathologies': [p.strip() for p in pathologies]}

        # Create training pairs
        training_pairs = embedder.create_training_pairs(reports, annotations_dict)

        # Train
        embedder.train_on_domain(
            training_pairs,
            num_epochs=config['embeddings']['embedding_epochs'],
            batch_size=config['embeddings']['embedding_batch_size'],
            output_path=str(output_dir / "finetuned_embedder")
        )

    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = embedder.encode(report_texts, batch_size=32, show_progress=True)

    # Save embeddings
    embeddings_file = output_dir / "report_embeddings.pkl"
    embedder.save_embeddings(embeddings, image_ids, str(embeddings_file))

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Embeddings saved to {embeddings_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
