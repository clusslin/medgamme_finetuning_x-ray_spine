"""
MedGemma 27B model for spine X-ray analysis with multi-task learning.

This module implements:
1. Vision-language model based on MedGemma
2. Multi-task heads for different objectives
3. LoRA finetuning support
4. Efficient training with quantization
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoProcessor
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction heads for spine X-ray analysis.

    Tasks:
    1. Region classification (cervical/thoracic/lumbar)
    2. Pose classification (neutral/flexion/extension/oblique)
    3. Pathology detection (multi-label)
    4. Implant detection (multi-label)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_regions: int = 3,
        num_poses: int = 4,
        num_pathologies: int = 10,
        num_implants: int = 6,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_size: Size of input features from backbone
            num_regions: Number of spine regions
            num_poses: Number of pose types
            num_pathologies: Number of pathology classes
            num_implants: Number of implant types
            dropout: Dropout probability
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)

        # Shared feature extraction
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        # Region classification head
        self.region_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_regions)
        )

        # Pose classification head
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_poses)
        )

        # Pathology detection head (multi-label)
        self.pathology_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_pathologies)
        )

        # Implant detection head (multi-label)
        self.implant_head = nn.Sequential(
            nn.Linear(hidden_size // 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_implants)
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads.

        Args:
            features: Input features (batch_size, hidden_size)

        Returns:
            Dictionary with logits for each task
        """
        # Shared features
        shared = self.shared_fc(features)
        shared = self.dropout(shared)

        # Task-specific predictions
        outputs = {
            'region_logits': self.region_head(shared),
            'pose_logits': self.pose_head(shared),
            'pathology_logits': self.pathology_head(shared),
            'implant_logits': self.implant_head(shared)
        }

        return outputs


class MedGemmaSpineModel(nn.Module):
    """
    MedGemma 27B model adapted for spine X-ray analysis.

    Architecture:
    - MedGemma 27B as backbone (vision-language model)
    - Multi-task heads for classification tasks
    - Report generation capability
    - LoRA for efficient finetuning
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-27b",
        num_regions: int = 3,
        num_poses: int = 4,
        num_pathologies: int = 10,
        num_implants: int = 6,
        use_lora: bool = True,
        lora_config: Optional[Dict[str, Any]] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto"
    ):
        """
        Args:
            model_name: HuggingFace model name or path
            num_regions: Number of spine regions to classify
            num_poses: Number of pose types to classify
            num_pathologies: Number of pathology classes
            num_implants: Number of implant types
            use_lora: Whether to use LoRA for finetuning
            lora_config: LoRA configuration dictionary
            load_in_4bit: Load model in 4-bit quantization
            load_in_8bit: Load model in 8-bit quantization
            device_map: Device mapping for model
        """
        super().__init__()

        self.model_name = model_name
        self.use_lora = use_lora

        # Configure quantization
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load base model
        print(f"Loading model: {model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )
        except Exception as e:
            print(f"Warning: Could not load {model_name}. Using gemma-2b as fallback.")
            print(f"Error: {e}")
            # Fallback to a smaller Gemma model for testing
            self.model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-2b",
                quantization_config=quantization_config,
                device_map=device_map,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16
            )

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(
                "google/gemma-2b",
                trust_remote_code=True
            )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Prepare model for k-bit training if using quantization
        if load_in_4bit or load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA if requested
        if use_lora:
            self._apply_lora(lora_config)

        # Get hidden size from model config
        self.hidden_size = self.model.config.hidden_size

        # Multi-task heads
        self.task_heads = MultiTaskHead(
            hidden_size=self.hidden_size,
            num_regions=num_regions,
            num_poses=num_poses,
            num_pathologies=num_pathologies,
            num_implants=num_implants
        )

        # Feature projection for vision inputs
        self.vision_projection = nn.Sequential(
            nn.Linear(768, self.hidden_size),  # Assuming ViT-like vision encoder
            nn.LayerNorm(self.hidden_size),
            nn.GELU()
        )

        print(f"Model initialized with {self.count_parameters()} parameters")
        print(f"Trainable parameters: {self.count_trainable_parameters()}")

    def _apply_lora(self, lora_config: Optional[Dict[str, Any]] = None):
        """Apply LoRA to the model"""
        if lora_config is None:
            lora_config = {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": TaskType.CAUSAL_LM
            }

        peft_config = LoraConfig(**lora_config)
        self.model = get_peft_model(self.model, peft_config)
        print("LoRA applied to model")

    def count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def count_trainable_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def extract_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract features from the model.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            image_features: Optional image features

        Returns:
            Feature tensor (batch_size, hidden_size)
        """
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        # Get last hidden state
        hidden_states = outputs.hidden_states[-1]

        # Pool features (mean pooling over sequence)
        pooled_features = (hidden_states * attention_mask.unsqueeze(-1)).sum(1) / \
                         attention_mask.sum(1, keepdim=True)

        # If image features provided, combine with text features
        if image_features is not None:
            image_features = self.vision_projection(image_features)
            # Simple concatenation and projection
            pooled_features = (pooled_features + image_features) / 2

        return pooled_features

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
        generate: bool = False,
        max_new_tokens: int = 256
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            image_features: Optional image features (batch_size, feature_dim)
            labels: Optional labels for training
            generate: Whether to generate text
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with outputs for all tasks
        """
        if generate:
            # Text generation mode
            generated_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            return {
                'generated_ids': generated_ids,
                'generated_text': self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
            }

        # Classification mode
        # Extract features
        features = self.extract_features(input_ids, attention_mask, image_features)

        # Get predictions from task heads
        outputs = self.task_heads(features)

        # Calculate losses if labels provided
        if labels is not None:
            losses = self._calculate_losses(outputs, labels)
            outputs['losses'] = losses
            outputs['total_loss'] = sum(losses.values())

        return outputs

    def _calculate_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate losses for all tasks.

        Args:
            outputs: Model outputs
            labels: Ground truth labels

        Returns:
            Dictionary of losses
        """
        losses = {}

        # Region classification loss (cross-entropy)
        if 'region' in labels:
            losses['region_loss'] = F.cross_entropy(
                outputs['region_logits'],
                labels['region']
            )

        # Pose classification loss (cross-entropy)
        if 'pose' in labels:
            losses['pose_loss'] = F.cross_entropy(
                outputs['pose_logits'],
                labels['pose']
            )

        # Pathology detection loss (binary cross-entropy)
        if 'pathology' in labels:
            losses['pathology_loss'] = F.binary_cross_entropy_with_logits(
                outputs['pathology_logits'],
                labels['pathology']
            )

        # Implant detection loss (binary cross-entropy)
        if 'implant' in labels:
            losses['implant_loss'] = F.binary_cross_entropy_with_logits(
                outputs['implant_logits'],
                labels['implant']
            )

        return losses

    def generate_report(
        self,
        prompt: str,
        image_features: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate radiology report from prompt.

        Args:
            prompt: Input prompt
            image_features: Optional image features
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated report text
        """
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.model.device)

        # Generate
        outputs = self.forward(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            image_features=image_features,
            generate=True,
            max_new_tokens=max_length
        )

        return outputs['generated_text'][0]

    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        # Save task heads separately
        torch.save(
            self.task_heads.state_dict(),
            f"{save_directory}/task_heads.pt"
        )
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str, **kwargs):
        """Load model from directory"""
        model = cls(**kwargs)
        # Load task heads
        task_heads_path = f"{load_directory}/task_heads.pt"
        if torch.cuda.is_available():
            model.task_heads.load_state_dict(torch.load(task_heads_path))
        else:
            model.task_heads.load_state_dict(
                torch.load(task_heads_path, map_location='cpu')
            )
        print(f"Model loaded from {load_directory}")
        return model
