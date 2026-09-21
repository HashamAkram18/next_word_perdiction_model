import time
import collections
from typing import List, Dict, Any, Tuple, Optional, Generator
import numpy as np
import tensorflow as tf

from ..data.dataset import pad_sequence
from ..models.registry import ModelRegistry, ModelInfo
from ..data.spellcheck import SpellChecker

class InferenceEngine:
    """
    High-throughput, ultra-low latency prediction engine.
    Supports Top-K, Top-p (Nucleus), Repetition Penalty, LRU Caching,
    Token Streaming, and real-time Spellchecking.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None, default_model_id: str = "uni_lstm"):
        self.registry = registry or ModelRegistry()
        self.active_model_id = default_model_id
        self.model = None
        self.tokenizer = None
        self.input_length = 50
        self.model_info: Optional[ModelInfo] = None
        
        # LRU Cache
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.cache_capacity = 2048

        # Spellchecker
        self.spellchecker = SpellChecker()

        self._load_active_model()

    def _load_active_model(self) -> None:
        """Load or switch the active neural network."""
        info = self.registry.get(self.active_model_id)
        if not info:
            available = list(self.registry.list_models().keys())
            if not available:
                raise RuntimeError("No models found in registry!")
            self.active_model_id = available[0]
            info = self.registry.get(self.active_model_id)

        self.model_info = info
        self.model, self.tokenizer = info.load()
        self.input_length = info.input_length
        self.cache.clear()

        # Update spellchecker with tokenizer vocabulary
        if hasattr(self.tokenizer, "word_index"):
            self.spellchecker.add_vocabulary(list(self.tokenizer.word_index.keys()))

        # Warm up execution graph
        dummy_input = np.zeros((1, self.input_length), dtype=np.int32)
        _ = self.model(dummy_input, training=False)
        print(f"[InferenceEngine] Activated model '{self.active_model_id}' (Input len: {self.input_length})")

    def switch_model(self, model_id: str) -> Dict[str, Any]:
        """Dynamically hot-swap active model."""
        if model_id not in self.registry.list_models():
            raise ValueError(f"Model ID '{model_id}' not found.")
        self.active_model_id = model_id
        self._load_active_model()
        return {
            "active_model": self.active_model_id,
            "name": self.model_info.name,
            "input_length": self.input_length,
            "vocab_size": self.model_info.vocab_size,
        }

    def _apply_sampling_and_penalties(
        self,
        raw_probs: np.ndarray,
        recent_token_ids: List[int],
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> np.ndarray:
        """
        Apply Repetition Penalty, Temperature Scaling, and Top-p (Nucleus) Filtering.
        """
        probs = np.array(raw_probs, dtype=np.float64)

        # 1. Repetition Penalty: reduce probability of recently seen tokens
        if repetition_penalty > 1.0 and recent_token_ids:
            for t_id in set(recent_token_ids):
                if 0 <= t_id < len(probs):
                    probs[t_id] /= repetition_penalty

        # Re-normalize
        probs = probs / np.maximum(np.sum(probs), 1e-12)

        # 2. Temperature scaling
        if temperature > 0 and temperature != 1.0:
            eps = 1e-12
            log_probs = np.log(np.maximum(probs, eps)) / max(temperature, 0.05)
            exp_probs = np.exp(log_probs - np.max(log_probs))
            probs = exp_probs / np.sum(exp_probs)

        # 3. Top-p (Nucleus) Sampling: filter out low-probability tail
        if 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Mask out tokens beyond cumulative threshold
            mask = cumulative_probs > top_p
            # Keep at least the highest probability token
            mask[0] = False
            filtered_indices = sorted_indices[mask]
            probs[filtered_indices] = 0.0

            # Re-normalize remaining nucleus
            total = np.sum(probs)
            if total > 0:
                probs = probs / total

        return probs

    def predict_next_candidates(
        self,
        text: str,
        top_k: int = 5,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
    ) -> Dict[str, Any]:
        """
        Predict Top-K candidate next words with Top-p nucleus filtering,
        repetition penalty, spellcheck issues, and telemetry.
        """
        start_time = time.perf_counter()
        cleaned_text = self.tokenizer.clean_text(text)

        # Real-time spellcheck detection
        spelling_issues = self.spellchecker.check_text(text)

        if not cleaned_text:
            return {
                "input_text": text,
                "top_word": "",
                "candidates": [],
                "misspelled_words": spelling_issues,
                "latency_ms": round((time.perf_counter() - start_time) * 1000, 2),
                "model_id": self.active_model_id,
                "cached": False,
            }

        # Check LRU cache
        cache_key = (self.active_model_id, cleaned_text, top_k, round(temperature, 2), round(top_p, 2), round(repetition_penalty, 2))
        if cache_key in self.cache:
            result = dict(self.cache[cache_key])
            self.cache.move_to_end(cache_key)
            result["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 2)
            result["misspelled_words"] = spelling_issues
            result["cached"] = True
            return result

        # Tokenize and pad
        tokens = self.tokenizer.texts_to_sequences([cleaned_text])[0]
        if not tokens:
            tokens = [0]
            
        padded_tokens = pad_sequence(tokens, maxlen=self.input_length, padding="pre")
        tensor_input = np.expand_dims(padded_tokens, axis=0)

        # Ultra-fast callable execution (sub-3ms)
        raw_probs = self.model(tensor_input, training=False).numpy()[0]

        # Apply sampling and penalties
        recent_tokens = tokens[-15:]
        probs = self._apply_sampling_and_penalties(
            raw_probs=raw_probs,
            recent_token_ids=recent_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Get top-k indices
        top_indices = np.argsort(probs)[::-1][: max(top_k, 1)]

        candidates = []
        for idx in top_indices:
            word = self.tokenizer.index_word.get(int(idx), None)
            if word and word != self.tokenizer.oov_token and idx != 0:
                prob = float(probs[idx])
                candidates.append({
                    "word": word,
                    "probability": round(prob, 4),
                    "confidence_pct": round(prob * 100, 1),
                })

        top_word = candidates[0]["word"] if candidates else ""
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        output = {
            "input_text": text,
            "top_word": top_word,
            "candidates": candidates[:top_k],
            "misspelled_words": spelling_issues,
            "latency_ms": round(elapsed_ms, 2),
            "model_id": self.active_model_id,
            "cached": False,
        }

        # Save to cache
        if len(self.cache) >= self.cache_capacity:
            self.cache.popitem(last=False)
        self.cache[cache_key] = output

        return output

    def generate_sequence(
        self,
        seed_text: str,
        num_words: int = 5,
        temperature: float = 0.75,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
    ) -> Dict[str, Any]:
        """
        Autoregressively generate next N words with Repetition Penalty and Top-p
        to eliminate looping and repetitive degradation.
        """
        start_time = time.perf_counter()
        current_text = seed_text
        generated_words: List[str] = []

        for _ in range(max(1, min(num_words, 30))):
            pred = self.predict_next_candidates(
                text=current_text,
                top_k=8,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            candidates = pred.get("candidates", [])
            if not candidates:
                break
            
            # Stochastic or greedy selection
            if temperature <= 0.1:
                next_word = candidates[0]["word"]
            else:
                words = [c["word"] for c in candidates]
                weights = [c["probability"] for c in candidates]
                s = sum(weights)
                if s > 0:
                    dist = [w / s for w in weights]
                    next_word = np.random.choice(words, p=dist)
                else:
                    next_word = words[0]

            generated_words.append(next_word)
            current_text = current_text + " " + next_word

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return {
            "seed_text": seed_text,
            "generated_words": generated_words,
            "completed_text": current_text,
            "latency_ms": round(elapsed_ms, 2),
            "model_id": self.active_model_id,
        }

    def generate_stream(
        self,
        seed_text: str,
        num_words: int = 5,
        temperature: float = 0.75,
        top_p: float = 0.9,
        repetition_penalty: float = 1.25,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream generated tokens one by one for WebSocket streaming."""
        current_text = seed_text
        for step in range(max(1, min(num_words, 30))):
            pred = self.predict_next_candidates(
                text=current_text,
                top_k=8,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            candidates = pred.get("candidates", [])
            if not candidates:
                break

            words = [c["word"] for c in candidates]
            weights = [c["probability"] for c in candidates]
            s = sum(weights)
            p_dist = [w / s for w in weights] if s > 0 else None
            next_word = np.random.choice(words, p=p_dist) if (temperature > 0.1 and p_dist) else words[0]

            current_text = current_text + " " + next_word
            yield {
                "token": next_word,
                "step": step + 1,
                "current_text": current_text,
            }
