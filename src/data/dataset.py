import numpy as np
from typing import List, Tuple
from .preprocessor import TextTokenizer

def pad_sequence(seq: List[int], maxlen: int, padding: str = "pre") -> np.ndarray:
    """Pad or truncate a single 1D integer sequence."""
    if len(seq) >= maxlen:
        return np.array(seq[-maxlen:], dtype=np.int32)
    
    pad_len = maxlen - len(seq)
    if padding == "pre":
        return np.pad(seq, (pad_len, 0), mode="constant", constant_values=0)
    else:
        return np.pad(seq, (0, pad_len), mode="constant", constant_values=0)

def generate_ngram_training_data(
    texts: List[str],
    tokenizer: TextTokenizer,
    max_sequence_length: int = 40,
    max_samples: int = 15000,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Generate (X, y) sliding window training sequences from raw text.
    Uses windowed slicing to handle large literary corpora efficiently.
    """
    input_sequences: List[List[int]] = []
    sequences = tokenizer.texts_to_sequences(texts)

    for seq in sequences:
        if len(seq) < 2:
            continue
        # Generate n-grams bounded by max_sequence_length to avoid O(N^2) explosion
        for i in range(1, len(seq)):
            start_idx = max(0, i + 1 - max_sequence_length)
            n_gram = seq[start_idx : i + 1]
            input_sequences.append(n_gram)
            if len(input_sequences) >= max_samples:
                break
        if len(input_sequences) >= max_samples:
            break

    if not input_sequences:
        raise ValueError("No training sequences could be generated from the given text.")

    # Determine maximum length among generated samples
    max_len = min(max_sequence_length, max(len(s) for s in input_sequences))
    
    # Pad sequences
    padded = np.zeros((len(input_sequences), max_len), dtype=np.int32)
    for idx, seq in enumerate(input_sequences):
        padded[idx] = pad_sequence(seq, maxlen=max_len, padding="pre")

    X = padded[:, :-1]
    y = padded[:, -1]

    return X, y, tokenizer.num_words
