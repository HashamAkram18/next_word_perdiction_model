from dataclasses import dataclass, field
from typing import List, Optional
import os

@dataclass
class ModelConfig:
    arch: str = "stacked_lstm"  # Options: lstm, stacked_lstm, gru, stacked_gru, bi_lstm
    embedding_dim: int = 128
    hidden_units: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.2
    max_sequence_length: int = 50
    vocab_size: int = 5000

@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 0.002
    max_samples: int = 30000
    validation_split: float = 0.1
    early_stopping_patience: int = 8
    reduce_lr_patience: int = 4
    output_dir: str = "models/checkpoints"
    save_format: str = "keras"  # Modern .keras format

@dataclass
class InferenceConfig:
    top_k: int = 5
    temperature: float = 0.7
    max_gen_tokens: int = 8
    cache_capacity: int = 2048
