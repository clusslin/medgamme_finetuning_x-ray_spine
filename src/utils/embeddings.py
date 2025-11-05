"""
Text embedding utilities for medical reports.
Specialized for spine X-ray radiology reports.
"""

import json
import pickle
import re
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


class MedicalVocabularyBuilder:
    """
    Build medical vocabulary from spine X-ray reports.
    Extract domain-specific terms and phrases.
    """

    def __init__(
        self,
        medical_terms_file: Optional[str] = None,
        use_spacy: bool = True
    ):
        """
        Args:
            medical_terms_file: Path to predefined medical terms
            use_spacy: Whether to use spaCy for NLP processing
        """
        self.medical_terms_file = medical_terms_file
        self.use_spacy = use_spacy

        # Load medical terms if provided
        self.predefined_terms = set()
        if medical_terms_file and Path(medical_terms_file).exists():
            with open(medical_terms_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        self.predefined_terms.add(line.lower())

        # Load spaCy model
        self.nlp = None
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")

        self.vocabulary = set()
        self.term_frequencies = Counter()
        self.ngram_frequencies = Counter()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()

        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\-/()]', ' ', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract medical terms from text using various methods.

        Args:
            text: Input text

        Returns:
            List of extracted medical terms
        """
        terms = []

        # Clean text
        text = self.clean_text(text)

        # Method 1: Match predefined terms
        for term in self.predefined_terms:
            if term in text:
                terms.append(term)

        # Method 2: Use spaCy for named entity recognition and noun phrases
        if self.nlp:
            doc = self.nlp(text)

            # Extract noun phrases
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                if len(chunk_text.split()) <= 4:  # Limit to 4-word phrases
                    terms.append(chunk_text)

            # Extract entities
            for ent in doc.ents:
                if ent.label_ in ['DISEASE', 'SYMPTOM', 'BODY_PART']:
                    terms.append(ent.text.lower())

        # Method 3: Extract common medical patterns
        # Pattern: adjective + noun (e.g., "degenerative changes")
        pattern = r'\b(?:mild|moderate|severe|acute|chronic|diffuse|focal)\s+\w+(?:\s+\w+)?\b'
        matches = re.findall(pattern, text)
        terms.extend(matches)

        # Pattern: anatomical location (e.g., "L4-L5")
        pattern = r'\b[CTLS]\d+(?:-[CTLS]?\d+)?\b'
        matches = re.findall(pattern, text)
        terms.extend(matches)

        return terms

    def build_vocabulary(
        self,
        reports: List[str],
        min_frequency: int = 2
    ) -> Dict[str, int]:
        """
        Build vocabulary from a list of reports.

        Args:
            reports: List of report texts
            min_frequency: Minimum frequency for a term to be included

        Returns:
            Dictionary mapping terms to frequencies
        """
        print(f"Building vocabulary from {len(reports)} reports...")

        # Extract terms from all reports
        for report in tqdm(reports, desc="Processing reports"):
            terms = self.extract_medical_terms(report)
            self.term_frequencies.update(terms)

            # Also track n-grams (1-3 words)
            words = report.lower().split()
            for i in range(len(words)):
                # Unigrams
                self.ngram_frequencies[words[i]] += 1

                # Bigrams
                if i < len(words) - 1:
                    bigram = f"{words[i]} {words[i+1]}"
                    self.ngram_frequencies[bigram] += 1

                # Trigrams
                if i < len(words) - 2:
                    trigram = f"{words[i]} {words[i+1]} {words[i+2]}"
                    self.ngram_frequencies[trigram] += 1

        # Filter by frequency
        vocabulary = {
            term: freq
            for term, freq in self.term_frequencies.items()
            if freq >= min_frequency
        }

        self.vocabulary = set(vocabulary.keys())

        print(f"Built vocabulary with {len(vocabulary)} terms")
        return vocabulary

    def get_top_terms(self, n: int = 100) -> List[Tuple[str, int]]:
        """Get top N most frequent terms"""
        return self.term_frequencies.most_common(n)

    def save_vocabulary(self, output_path: str):
        """Save vocabulary to file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for term, freq in sorted(
                self.term_frequencies.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                f.write(f"{term}\t{freq}\n")

        print(f"Vocabulary saved to {output_path}")


class ReportEmbedder:
    """
    Generate and train embeddings for spine X-ray reports.
    Uses sentence transformers with domain adaptation.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        embedding_dim: int = 384,
        max_seq_length: int = 512,
        device: Optional[str] = None
    ):
        """
        Args:
            model_name: Pre-trained sentence transformer model
            embedding_dim: Embedding dimension
            max_seq_length: Maximum sequence length
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_seq_length = max_seq_length

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length

        print(f"Model loaded on {self.device}")

    def preprocess_report(self, text: str) -> str:
        """
        Preprocess report text before embedding.

        Args:
            text: Raw report text

        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize common abbreviations
        abbreviations = {
            'c-spine': 'cervical spine',
            't-spine': 'thoracic spine',
            'l-spine': 'lumbar spine',
            'ap': 'anteroposterior',
            'lat': 'lateral',
            'obl': 'oblique'
        }

        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)

        return text

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar

        Returns:
            Array of embeddings (N, embedding_dim)
        """
        # Preprocess
        processed_texts = [self.preprocess_report(text) for text in texts]

        # Encode
        embeddings = self.model.encode(
            processed_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def train_on_domain(
        self,
        train_data: List[Dict[str, str]],
        num_epochs: int = 10,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        output_path: Optional[str] = None
    ):
        """
        Fine-tune the embedding model on domain-specific data.

        Args:
            train_data: List of dicts with 'text1', 'text2', 'label' (similarity)
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            output_path: Path to save fine-tuned model
        """
        print(f"Fine-tuning embedding model on {len(train_data)} examples...")

        # Convert to InputExamples
        train_examples = []
        for item in train_data:
            example = InputExample(
                texts=[item['text1'], item['text2']],
                label=float(item['label'])
            )
            train_examples.append(example)

        # Create dataloader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size
        )

        # Define loss function
        train_loss = losses.CosineSimilarityLoss(self.model)

        # Train
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=int(len(train_dataloader) * 0.1),
            optimizer_params={'lr': learning_rate},
            show_progress_bar=True
        )

        # Save if path provided
        if output_path:
            self.model.save(output_path)
            print(f"Model saved to {output_path}")

    def create_training_pairs(
        self,
        reports: Dict[str, str],
        annotations: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Create training pairs from reports based on pathology similarity.

        Args:
            reports: Dict mapping image_id to report text
            annotations: Dict mapping image_id to annotations

        Returns:
            List of training pairs
        """
        pairs = []
        image_ids = list(reports.keys())

        print("Creating training pairs...")

        for i, id1 in enumerate(tqdm(image_ids)):
            for id2 in image_ids[i+1:]:
                # Get pathologies for both
                path1 = set(annotations[id1].get('pathologies', []))
                path2 = set(annotations[id2].get('pathologies', []))

                # Calculate Jaccard similarity
                if len(path1) == 0 and len(path2) == 0:
                    similarity = 1.0  # Both normal
                elif len(path1) == 0 or len(path2) == 0:
                    similarity = 0.0  # One normal, one abnormal
                else:
                    intersection = len(path1.intersection(path2))
                    union = len(path1.union(path2))
                    similarity = intersection / union if union > 0 else 0.0

                # Create pair
                pairs.append({
                    'text1': reports[id1],
                    'text2': reports[id2],
                    'label': similarity
                })

        print(f"Created {len(pairs)} training pairs")
        return pairs

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        image_ids: List[str],
        output_path: str
    ):
        """
        Save embeddings to file.

        Args:
            embeddings: Array of embeddings
            image_ids: List of image IDs
            output_path: Output file path
        """
        embedding_dict = {
            image_id: emb
            for image_id, emb in zip(image_ids, embeddings)
        }

        with open(output_path, 'wb') as f:
            pickle.dump(embedding_dict, f)

        print(f"Saved embeddings for {len(image_ids)} reports to {output_path}")

    def load_embeddings(self, input_path: str) -> Dict[str, np.ndarray]:
        """Load embeddings from file"""
        with open(input_path, 'rb') as f:
            embeddings = pickle.load(f)

        print(f"Loaded embeddings for {len(embeddings)} reports")
        return embeddings


class TFIDFEmbedder:
    """
    TF-IDF based embeddings for medical reports.
    Useful as a baseline or complementary to neural embeddings.
    """

    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 3)
    ):
        """
        Args:
            max_features: Maximum number of features
            ngram_range: N-gram range (min, max)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            lowercase=True,
            stop_words='english'
        )

        self.is_fitted = False

    def fit(self, texts: List[str]):
        """Fit TF-IDF vectorizer on texts"""
        self.vectorizer.fit(texts)
        self.is_fitted = True

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to TF-IDF vectors"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit() first.")

        return self.vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit and transform in one step"""
        return self.vectorizer.fit_transform(texts).toarray()

    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.vectorizer.get_feature_names_out().tolist()
