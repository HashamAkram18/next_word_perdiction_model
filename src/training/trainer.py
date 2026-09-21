import os
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ..config import ModelConfig, TrainingConfig
from ..data.preprocessor import TextTokenizer
from ..data.dataset import generate_ngram_training_data
from ..models.architectures import build_neural_model

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

class ModelTrainer:
    """End-to-end training pipeline for custom corpus datasets."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()

    def train_on_texts(
        self,
        texts: List[str],
        model_id: str,
        display_name: Optional[str] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Full training workflow:
        1. Fit tokenizer and save JSON artifact
        2. Generate sequence data (X, y)
        3. Build model architecture
        4. Train with sparse categorical crossentropy & early stopping
        5. Export model and metadata
        """
        output_dir = ROOT_DIR / self.training_config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Trainer] >>> Starting training run for '{model_id}' ({self.model_config.arch})")
        print(f"[Trainer] Total input lines/paragraphs: {len(texts):,}")

        # 1. Tokenization
        tokenizer = TextTokenizer(max_vocab_size=self.model_config.vocab_size)
        tokenizer.fit_on_texts(texts)
        vocab_size = tokenizer.num_words
        print(f"[Trainer] Vocabulary size built: {vocab_size:,} words")

        # 2. Dataset sequence generation
        X, y, _ = generate_ngram_training_data(
            texts=texts,
            tokenizer=tokenizer,
            max_sequence_length=self.model_config.max_sequence_length,
            max_samples=self.training_config.max_samples,
        )
        seq_length = X.shape[1]
        print(f"[Trainer] Generated {len(X):,} training sequences. Input sequence length: {seq_length}")

        # 3. Model construction
        model = build_neural_model(
            arch=self.model_config.arch,
            vocab_size=vocab_size,
            input_length=seq_length,
            embedding_dim=self.model_config.embedding_dim,
            hidden_units=self.model_config.hidden_units,
            dropout_rate=self.model_config.dropout_rate,
        )
        model.summary(print_fn=lambda x: print(f"  {x}"))

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.training_config.learning_rate)
        # Using sparse_categorical_crossentropy avoids large one-hot memory allocations
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        # 4. File paths
        model_filename = f"{model_id}.keras"
        tok_filename = f"{model_id}_tokenizer.json"
        meta_filename = f"{model_id}_meta.json"

        model_save_path = output_dir / model_filename
        tok_save_path = output_dir / tok_filename
        meta_save_path = output_dir / meta_filename

        callbacks = [
            EarlyStopping(
                monitor="loss",
                patience=self.training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=self.training_config.reduce_lr_patience,
                min_lr=1e-5,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(model_save_path),
                monitor="loss",
                save_best_only=True,
                verbose=0,
            ),
        ]

        # 5. Fit
        start_time = time.time()
        history = model.fit(
            X,
            y,
            epochs=self.training_config.epochs,
            batch_size=self.training_config.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        train_duration = round(time.time() - start_time, 2)

        # 6. Save artifacts
        model.save(str(model_save_path))
        tokenizer.save_json(tok_save_path)

        final_loss = float(history.history["loss"][-1])
        final_acc = float(history.history["accuracy"][-1])
        perplexity = round(float(np.exp(min(final_loss, 20))), 2)

        metadata = {
            "model_id": model_id,
            "name": display_name or model_id.replace("_", " ").title(),
            "arch": self.model_config.arch,
            "input_length": seq_length,
            "vocab_size": vocab_size,
            "model_filename": model_filename,
            "tokenizer_filename": tok_filename,
            "description": description or f"Trained on custom corpus ({self.model_config.arch.upper()})",
            "metrics": {
                "final_loss": round(final_loss, 4),
                "final_accuracy": round(final_acc, 4),
                "perplexity": perplexity,
                "epochs_trained": len(history.history["loss"]),
                "train_duration_sec": train_duration,
            },
        }

        with open(meta_save_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n[Trainer] >>> Training complete in {train_duration}s!")
        print(f"[Trainer] Loss: {final_loss:.4f} | Accuracy: {final_acc * 100:.2f}% | Perplexity: {perplexity}")
        print(f"[Trainer] Artifacts saved to: {output_dir}")
        return metadata
