import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from ..data.preprocessor import TextTokenizer

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS_DIR = ROOT_DIR / "models" / "checkpoints"

class ModelInfo:
    def __init__(
        self,
        model_id: str,
        name: str,
        arch_type: str,
        input_length: int,
        vocab_size: int,
        model_path: Path,
        tokenizer_path: Path,
        is_legacy: bool = False,
        description: str = "",
    ):
        self.model_id = model_id
        self.name = name
        self.arch_type = arch_type
        self.input_length = input_length
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.is_legacy = is_legacy
        self.description = description
        self.loaded_model = None
        self.loaded_tokenizer = None

    def load(self) -> Tuple[Any, TextTokenizer]:
        """Lazy loader for model and tokenizer."""
        if self.loaded_model is not None and self.loaded_tokenizer is not None:
            return self.loaded_model, self.loaded_tokenizer

        # 1. Load Tokenizer
        if self.tokenizer_path.suffix == ".json":
            self.loaded_tokenizer = TextTokenizer.load_json(self.tokenizer_path)
        else:
            with open(self.tokenizer_path, "rb") as f:
                raw_tok = pickle.load(f)
            self.loaded_tokenizer = TextTokenizer.from_keras_tokenizer(raw_tok)

        # 2. Load Model
        if self.is_legacy or self.model_path.suffix == ".pkl":
            with open(self.model_path, "rb") as f:
                self.loaded_model = pickle.load(f)
        else:
            self.loaded_model = tf.keras.models.load_model(self.model_path)

        return self.loaded_model, self.loaded_tokenizer

class ModelRegistry:
    """Central catalog managing legacy pickle models and newly trained models."""

    def __init__(self):
        self.registry: Dict[str, ModelInfo] = {}
        self._discover_models()

    def _discover_models(self) -> None:
        """Scan workspace for both legacy and modern trained models."""
        legacy_tok_path = ROOT_DIR / "tokenizer.pkl"

        # 1. Register Legacy models if present
        legacy_models = [
            ("uni_lstm", "Uni LSTM (Legacy)", "lstm", 194, 279, ROOT_DIR / "uni_lstm.pkl", "Single-layer Unidirectional LSTM"),
            ("st_lstm", "Stacked LSTM (Legacy)", "stacked_lstm", 56, 283, ROOT_DIR / "st_lstm.pkl", "4-layer Deep Stacked LSTM"),
            ("uni_GRU", "Uni GRU (Legacy)", "gru", 56, 283, ROOT_DIR / "uni_GRU.pkl", "Single-layer Gated Recurrent Unit"),
            ("st_GRU", "Stacked GRU (Legacy)", "stacked_gru", 56, 283, ROOT_DIR / "st_GRU.pkl", "Deep Stacked GRU"),
        ]

        for model_id, name, arch, seq_len, vocab, path, desc in legacy_models:
            if path.exists() and legacy_tok_path.exists():
                self.registry[model_id] = ModelInfo(
                    model_id=model_id,
                    name=name,
                    arch_type=arch,
                    input_length=seq_len,
                    vocab_size=vocab,
                    model_path=path,
                    tokenizer_path=legacy_tok_path,
                    is_legacy=True,
                    description=desc,
                )

        # 2. Register Modern Checkpoints in models/checkpoints/
        if CHECKPOINTS_DIR.exists():
            for meta_file in CHECKPOINTS_DIR.glob("*_meta.json"):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    m_id = meta["model_id"]
                    m_path = CHECKPOINTS_DIR / meta["model_filename"]
                    t_path = CHECKPOINTS_DIR / meta["tokenizer_filename"]
                    if m_path.exists() and t_path.exists():
                        self.registry[m_id] = ModelInfo(
                            model_id=m_id,
                            name=meta.get("name", m_id),
                            arch_type=meta.get("arch", "lstm"),
                            input_length=meta.get("input_length", 50),
                            vocab_size=meta.get("vocab_size", 1000),
                            model_path=m_path,
                            tokenizer_path=t_path,
                            is_legacy=False,
                            description=meta.get("description", "Trained model checkpoint"),
                        )
                except Exception as e:
                    print(f"[Registry] Error reading metadata {meta_file}: {e}")

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata summary for all available models."""
        self._discover_models()
        return {
            m_id: {
                "id": m.model_id,
                "name": m.name,
                "arch": m.arch_type,
                "input_length": m.input_length,
                "vocab_size": m.vocab_size,
                "is_legacy": m.is_legacy,
                "description": m.description,
            }
            for m_id, m in self.registry.items()
        }

    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Fetch model info by identifier."""
        if model_id not in self.registry:
            self._discover_models()
        return self.registry.get(model_id)
