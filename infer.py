"""
CLI Next-Word Prediction & Sequence Continuation Script.

Examples:
    uv run python infer.py --text "i am a sick man i am a"
    uv run python infer.py --model corpus_gutenberg_dostoevsky_residual_gru --text "he was an important aspect of" --words 10
"""
import argparse
import sys
from src.inference.engine import InferenceEngine

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Run inference using trained neural models.")
    parser.add_argument(
        "--model",
        type=str,
        default="corpus_gutenberg_dostoevsky_residual_gru",
        help="Model ID to use for inference (default: corpus_gutenberg_dostoevsky_residual_gru)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="i am a sick man i am a",
        help="Input context prompt for prediction",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate words to display",
    )
    parser.add_argument(
        "--words",
        type=int,
        default=8,
        help="Number of next words to autoregressively generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (lower = more deterministic, higher = more creative)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling cutoff threshold",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.25,
        help="Repetition penalty factor (1.0 = none, 1.25 = recommended)",
    )

    args = parser.parse_args()

    engine = InferenceEngine(default_model_id=args.model)
    print(f"\n========================================================")
    print(f"Active Model: {args.model}")
    print(f"Input Prompt: \"{args.text}\"")
    print(f"========================================================")

    # 1. Single next-word prediction
    res = engine.predict_next_candidates(
        text=args.text,
        top_k=args.top_k,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"\nTop Predicted Next Word: '{res['top_word']}' (Latency: {res['latency_ms']} ms)")
    print(f"\nTop-{args.top_k} Candidate Probabilities:")
    for c in res["candidates"]:
        print(f"  • {c['word']:<18} {c['confidence_pct']:>5.1f}%  (prob: {c['probability']:.4f})")

    # 2. Multi-word sentence continuation
    if args.words > 0:
        gen = engine.generate_sequence(
            seed_text=args.text,
            num_words=args.words,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"\nGenerated Continuation ({args.words} words):")
        print(f"  \"{gen['completed_text']}\"")
    print(f"========================================================\n")

if __name__ == "__main__":
    main()
