import argparse
import sys
from pathlib import Path
from typing import List

from src.config import ModelConfig, TrainingConfig
from src.training.trainer import ModelTrainer
from src.data.fetcher import prepare_all_datasets, DATA_RAW_DIR

def resolve_and_ensure_corpus(source: str) -> Path:
    """Find corpus file or auto-download/generate if missing."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    candidate = Path(source)

    if candidate.exists() and candidate.is_file():
        return candidate

    # Check in data/raw
    in_raw = DATA_RAW_DIR / source
    if in_raw.exists():
        return in_raw

    in_raw_txt = DATA_RAW_DIR / f"{source}.txt"
    if in_raw_txt.exists():
        return in_raw_txt

    # If not found, auto-fetch all datasets
    print(f"[CLI] Corpus '{source}' not found locally. Automatically fetching datasets...")
    datasets = prepare_all_datasets(download_online=True)

    if source in datasets and datasets[source].exists():
        return datasets[source]
    if in_raw_txt.exists():
        return in_raw_txt

    # Fallback to any existing file or error
    available = [f.stem for f in DATA_RAW_DIR.glob("*.txt")]
    print(f"[CLI Error] Dataset '{source}' could not be resolved.")
    print(f"Available datasets in data/raw: {available}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Modular Neural Retraining CLI for Next Word Prediction (Residual LSTM/GRU)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="dostoevsky_notes",
        help="Path to text corpus or preset name (e.g. 'dostoevsky_notes', 'harry_potter_lore', 'dostoevsky_core').",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="residual_gru",
        choices=[
            "lstm",
            "stacked_lstm",
            "residual_lstm",
            "gru",
            "stacked_gru",
            "residual_gru",
            "bi_lstm",
        ],
        help="Neural architecture to train (including Residual LSTM/GRU).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Identifier name for saving the model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--hidden-units", type=int, default=128, help="Hidden units per layer.")
    parser.add_argument("--max-seq-len", type=int, default=40, help="Maximum n-gram sequence length.")
    parser.add_argument("--vocab-size", type=int, default=4000, help="Maximum vocabulary size.")
    parser.add_argument("--max-samples", type=int, default=30000, help="Maximum sliding-window sequences to extract for training.")
    parser.add_argument("--learning-rate", type=float, default=0.003, help="Learning rate.")
    parser.add_argument("--fetch-datasets", action="store_true", help="Force refresh all remote datasets.")

    args = parser.parse_args()

    if args.fetch_datasets:
        prepare_all_datasets(download_online=True)

    data_path = resolve_and_ensure_corpus(args.data)
    print(f"[CLI] Loading training corpus from: {data_path}")
    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print("[CLI Error] No valid text found in corpus.")
        sys.exit(1)

    model_id = args.model_id or f"{data_path.stem}_{args.arch}"
    display_name = f"{data_path.stem.replace('_', ' ').title()} ({args.arch.upper()})"

    model_config = ModelConfig(
        arch=args.arch,
        embedding_dim=args.embedding_dim,
        hidden_units=args.hidden_units,
        max_sequence_length=args.max_seq_len,
        vocab_size=args.vocab_size,
    )

    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
    )

    trainer = ModelTrainer(model_config=model_config, training_config=training_config)
    trainer.train_on_texts(
        texts=lines,
        model_id=model_id,
        display_name=display_name,
        description=f"Trained on {data_path.name} using {args.arch.upper()} architecture.",
    )

if __name__ == "__main__":
    main()
