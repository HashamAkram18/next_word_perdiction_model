import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class TextTokenizer:
    """Lightweight, fast, and serializable tokenizer for word-level language modeling."""

    def __init__(self, max_vocab_size: Optional[int] = None, oov_token: str = "<OOV>"):
        self.max_vocab_size = max_vocab_size
        self.oov_token = oov_token
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self.num_words: int = 0

    def clean_text(self, text: str) -> str:
        """Normalize punctuation, quotes, and whitespace."""
        text = text.lower()
        text = re.sub(r"[’‘]", "'", text)
        text = re.sub(r"[“”]", '"', text)
        # Keep apostrophes within words (e.g. don't, harry's), split other punctuation
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fit_on_texts(self, texts: List[str]) -> "TextTokenizer":
        """Build vocabulary from a list of text strings."""
        counts: Dict[str, int] = {}
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

        self.word_counts = counts
        sorted_tokens = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        if self.max_vocab_size:
            sorted_tokens = sorted_tokens[: self.max_vocab_size - 1]  # Reserve slot for OOV/pad

        # Indexing: 0 is reserved for padding, 1..N for words
        self.word_index = {}
        self.index_word = {}

        if self.oov_token:
            self.word_index[self.oov_token] = 1
            self.index_word[1] = self.oov_token
            start_idx = 2
        else:
            start_idx = 1

        for idx, (word, _) in enumerate(sorted_tokens, start=start_idx):
            self.word_index[word] = idx
            self.index_word[idx] = word

        self.num_words = len(self.word_index) + 1  # Including index 0 padding
        return self

    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """Convert list of texts to integer token sequences."""
        sequences = []
        oov_idx = self.word_index.get(self.oov_token, 0)
        for text in texts:
            cleaned = self.clean_text(text)
            tokens = cleaned.split()
            seq = [self.word_index.get(t, oov_idx) for t in tokens if t]
            sequences.append(seq)
        return sequences

    def sequence_to_text(self, sequence: List[int]) -> str:
        """Convert a sequence of token IDs back into a readable string."""
        return " ".join(self.index_word.get(idx, "") for idx in sequence if idx != 0).strip()

    def save_json(self, path: Union[str, Path]) -> None:
        """Export tokenizer vocabulary to portable JSON format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "max_vocab_size": self.max_vocab_size,
            "oov_token": self.oov_token,
            "num_words": self.num_words,
            "word_index": self.word_index,
            # string keys for JSON serialization
            "index_word": {str(k): v for k, v in self.index_word.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "TextTokenizer":
        """Load tokenizer from JSON format."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(max_vocab_size=data.get("max_vocab_size"), oov_token=data.get("oov_token", "<OOV>"))
        tok.num_words = data["num_words"]
        tok.word_index = data["word_index"]
        tok.index_word = {int(k): v for k, v in data["index_word"].items()}
        return tok

    @classmethod
    def from_keras_tokenizer(cls, keras_tokenizer) -> "TextTokenizer":
        """Create a TextTokenizer adapter from an existing Keras Tokenizer object."""
        tok = cls()
        tok.word_index = dict(keras_tokenizer.word_index)
        tok.index_word = {int(k): v for k, v in keras_tokenizer.index_word.items()}
        tok.num_words = len(tok.word_index) + 1
        return tok
