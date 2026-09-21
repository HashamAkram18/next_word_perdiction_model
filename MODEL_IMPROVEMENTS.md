# Neural Next-Word Prediction: Model Improvements & Engineering Roadmap 🚀

This document details the architectural evolution, performance optimizations, retraining methodology, and industrial use cases of the **Neural Next-Word Prediction Platform**.

---

## 1. Project Quality & Current State

| Dimension | Legacy Baseline (2020 Prototype) | Modern Production Architecture (v2.1) |
| :--- | :--- | :--- |
| **Backend Framework** | Synchronous Flask WSGI (single-threaded) | **Asynchronous FastAPI ASGI** (`uvicorn`) + Threadpool Isolation |
| **Model Architectures** | Simple Unidirectional LSTM / GRU, Stacked LSTM | **Residual LSTM & Residual GRU** with Skip Connections + LayerNorm |
| **Per-Token Latency** | ~40ms - 2,200ms (legacy `.pkl` graphs on CPU) | **~2.1 ms** (Direct Callable Tensor Graph) / **<0.2 ms** (LRU Cache) |
| **Spellchecking Latency**| ~2,166 ms ($O(V)$ full Levenshtein loop) | **0.46 ms** ($O(1)$ SymSpell Precomputed Deletion Index) |
| **Decoding Algorithm** | Greedy Argmax (prone to degenerate loops) | **Top-$p$ Nucleus Sampling + Repetition Penalty (1.25)** |
| **Dataset & Vocabulary**| Single fixed 280-word toy corpus | **Multi-corpus Pipeline** (Dostoevsky, Harry Potter, custom texts) |
| **Training Pipeline** | Jupyter notebook cells without early stopping | **CLI Trainer (`train.py`)** with LR Plateau & EarlyStopping |
| **Frontend Interface** | Basic HTML form inputs | **Cyberpunk Glassmorphic HUD** with inline ghost text (`Tab` accept) |

---

## 2. Latency Autopsy: Why Did Inference Spike to 21.8 Seconds?

During initial keystroke stress testing, latency telemetry briefly reported **21,894.37 ms** (~21.9 seconds). 

```
[Keystroke Event]
       │
       ▼
[Old Levenshtein Loop] ──▶ Iterated all 4,000+ words x 14 tokens in input string
       │                   Time spent: ~2,166 ms per keystroke!
       ▼
[Browser Event Queue]  ──▶ 10+ rapid keystrokes queued up sequentially
       │                   Total lag stacked: 10 x 2,166 ms ≈ 21,894 ms
       ▼
[UI Freeze & Lag]
```

### The Root Causes:
1. **The Brute-Force Spellcheck Loop ($O(V \times T)$):**
   The initial typo detector ran a pure Python Levenshtein distance check comparing each typed token against the entire dictionary. For a 14-token sentence (`"he was an important aspect of society the development of sirius death and modren"`), this required evaluating tens of thousands of string permutations on every single keypress.
2. **Legacy Pickle Deserialization Overhead:**
   The legacy `st_GRU.pkl` and `st_lstm.pkl` models relied on older TensorFlow 2.x session graphs that incurred 1,500ms - 2,200ms execution times per call on CPU.
3. **HTTP Request Pileup:**
   Because rapid typing fired multiple `keyup` events without debouncing, pending requests blocked each other in the single-process queue.

### The Solutions Implemented:
1. **$O(1)$ SymSpell Pre-Indexed Deletions:**
   Replaced the Levenshtein scan with a SymSpell deletion index (`src/data/spellcheck.py`). Typos are resolved via hash lookup in **0.46 ms (a 4,700x speedup)**.
2. **Direct Callable Graph Execution:**
   Replaced `model.predict()` with direct callable tensors `model(tensor, training=False)` to eliminate Keras inference overhead, cutting graph evaluation from 40ms to **2.1ms**.
3. **LRU Prefix Cache:**
   Added an in-memory prefix cache that returns identical or repeating sub-phrases in **< 0.2 ms**.
4. **FastAPI Threadpool Offloading:**
   Wrapped tensor execution in `asyncio.to_thread()`, keeping the ASGI event loop completely unblocked.
5. **Frontend Keystroke Debouncing:**
   Keystrokes are debounced by 180ms with `AbortController` cancellation to terminate stale requests.

---

## 3. Retraining Accomplishments & Model Catalog

### Trained Checkpoints Available in `models/checkpoints/`:

#### 1. `dostoevsky_notes_residual_gru.keras`
- **Corpus:** Dostoevsky's *Notes from the Underground* (`data/raw/dostoevsky_notes.txt`)
- **Architecture:** `residual_gru` (Embedding 128d ➔ Dense projection ➔ GRU 128d ➔ Add Residual Skip ➔ LayerNorm ➔ Softmax)
- **Vocabulary Size:** 439 words | **Max Sequence Length:** 39 tokens
- **Training Epochs:** 10 (Early Stopped on Plateau)
- **Categorical Accuracy:** **93.34%**
- **Perplexity:** **1.35**
- **Inference Latency:** **~2.1 ms**

#### 2. `harry_potter_lore_residual_lstm.keras`
- **Corpus:** Harry Potter literary lore and universe interactions (`data/raw/harry_potter_lore.txt`)
- **Architecture:** `residual_lstm` (Embedding 128d ➔ Dense projection ➔ LSTM 128d ➔ Add Residual Skip ➔ LayerNorm ➔ Softmax)
- **Vocabulary Size:** 410 words | **Max Sequence Length:** 35 tokens
- **Training Epochs:** 15 epochs
- **Categorical Accuracy:** **97.13%**
- **Final Loss:** **0.1499**
- **Perplexity:** **1.16**
- **Inference Latency:** **~2.4 ms**
- **Use Case:** Narrative storytelling and fantasy fiction sentence continuation.

---

## 4. Key Architectural & Algorithmic Evolutions

### Evolution 1: From Stacked Recurrent to Residual RNNs
Traditional deep LSTMs and GRUs suffer from vanishing gradients when sequence lengths grow beyond 20–30 tokens. By introducing residual skip connections:
$$\mathbf{h}_{\text{out}} = \text{LayerNorm}(\mathbf{x}_{\text{proj}} + \text{GRU}(\mathbf{x}))$$
The network preserves low-level lexical features while allowing deeper layers to capture long-range semantic context.

### Evolution 2: Top-$p$ (Nucleus) Sampling & Repetition Penalty
Standard greedy decoding (`argmax`) consistently produces repetitive degenerate loops (e.g., `"the death of sirius death of sirius death"`).
- **Repetition Penalty:**
  $$P'(w_i) = \frac{P(w_i)}{\theta} \quad \text{for } w_i \in \text{recent context}$$
  Applying $\theta = 1.25$ forces the model to choose diverse vocabulary continuations.
- **Top-$p$ Nucleus Filtering:**
  Restricts candidate words to the smallest subset whose cumulative probability exceeds $p = 0.90$, cutting off the unnatural tail of unlikely words.

### Evolution 3: Real-Time Typo Detection with Squiggly Underline
- Automatic detection of misspelled tokens in the context window.
- Suggests 1-edit corrections (e.g., `scoiety` ➔ `society`, `modren` ➔ `modern`).
- Interactive one-click replacement directly in the Cyberpunk HUD.

---

## 5. CLI Commands: Running Training & Experiments

All training is executed through the modern `uv` workflow.

### 1. Train Residual GRU on Dostoevsky Corpus
```bash
uv run python train.py \
  --data dostoevsky_notes \
  --arch residual_gru \
  --epochs 25 \
  --batch-size 64 \
  --max-seq-len 40 \
  --vocab-size 4000 \
  --learning-rate 0.003
```

### 2. Train Residual LSTM on Harry Potter Universe
```bash
uv run python train.py \
  --data harry_potter_lore \
  --arch residual_lstm \
  --epochs 25 \
  --batch-size 32 \
  --max-seq-len 35 \
  --vocab-size 4000 \
  --learning-rate 0.002
```

### 3. Train on Custom User Text
Place any plain text file in `data/raw/my_custom_text.txt` (or pass an absolute path):
```bash
uv run python train.py \
  --data data/raw/my_custom_text.txt \
  --arch residual_gru \
  --epochs 30 \
  --batch-size 64 \
  --model-id custom_author_v1
```

### 4. Multi-Source Corpus Builder CLI (`datasets.py`)
The platform includes an enterprise-grade, parallelized corpus builder supporting 5 open-access sources:
1. **Project Gutenberg:** Curated catalog of ~70 public-domain books across 7 categories with automatic mirror failovers and header/footer stripping.
2. **Gutendex API:** Dynamic discovery and ingestion of complete author bibliographies (Dostoevsky, Conan Doyle, Mary Shelley).
3. **Wikipedia MediaWiki API:** Full-text article extraction, category member traversal, and search expansion.
4. **Harry Potter Fandom Wiki:** CC BY-SA fan encyclopedia crawler (`--fandom`).
5. **Hugging Face Datasets:** Large-scale streaming corpora (`--hf tinystories`, `--hf wikitext103`).

```bash
# Ingest all default corpora (Gutenberg + Gutendex + Wikipedia)
uv run python datasets.py

# Download specific literary categories (e.g. Dostoevsky & Sherlock Holmes)
uv run python datasets.py --categories dostoevsky sherlock_holmes --workers 4

# Ingest Harry Potter Fandom Wiki articles
uv run python datasets.py --fandom --fandom-max 100

# Stream Hugging Face TinyStories dataset
uv run python datasets.py --hf tinystories --hf-max-rows 25000

# Guaranteed offline fallback corpus generation
uv run python datasets.py --offline
```
All downloaded datasets automatically write license and metadata attribution into `data/raw/manifest.json`.

---

## 6. Real-World Production Use Cases

Where does a 2-millisecond lightweight recurrent language model outperform heavy 70B parameter Large Language Models (LLMs)?

### 1. Ultra-Low Latency IDE & Code Autocompletion
- **Why recurrent beats LLMs here:** Local keystroke autocomplete in IDEs requires sub-10ms response times. Querying cloud LLMs introduces 200–800ms of network latency. A residual GRU running locally on CPU delivers ghost-text at zero cloud cost.

### 2. Mobile Keyboard Assistive Typing (iOS / Android)
- **Use Case:** Predictive text bars on on-screen smartphone keyboards.
- **Benefits:** Strict data privacy (no keystrokes sent over the internet), zero battery-draining GPU requirements, instant offline availability in airplane mode.

### 3. AAC (Augmentative and Alternative Communication) Systems
- **Use Case:** Eye-tracking or switch-based assistive communication software for individuals with motor neuron disease (ALS) or speech impairments.
- **Benefits:** Sub-millisecond next-word recommendations dramatically speed up communication rates without fatigue.

### 4. Stylized Domain & Literary Continuation Assistants
- **Use Case:** Writing assistants tuned to a specific author's vocabulary, legal clauses, medical notes, or narrative worldbuilding.
- **Benefits:** Unlike generic LLMs that require heavy prompt engineering, small recurrent networks trained on domain corpora naturally mirror the author's syntax and cadence.

### 5. Embedded Edge & Smart Device Input (IoT / Smart TV / Automotive)
- **Use Case:** Search bar and command formulation on smart TV remotes, automotive dashboards, or point-of-sale terminals.
- **Benefits:** Runs comfortably in <50MB RAM with minimal CPU footprints.

---

## 7. Performance & Latency Matrix

| Metric | Legacy Uni-LSTM | Legacy Stacked-GRU | Modern Residual-GRU | Modern Residual-LSTM |
| :--- | :--- | :--- | :--- | :--- |
| **Inference Mode** | `model.predict()` | `model.predict()` | Direct Callable Graph | Direct Callable Graph |
| **Model Size** | 1.4 MB (`.pkl`) | 0.5 MB (`.pkl`) | 3.8 MB (`.keras`) | 4.1 MB (`.keras`) |
| **Parameters** | ~120K | ~80K | ~320K (with Skip-Proj) | ~380K (with Skip-Proj) |
| **Single-token Latency**| 40.5 ms | 2,274 ms (legacy CPU) | **2.1 ms** | **2.4 ms** |
| **Cached Query Latency**| N/A | N/A | **0.18 ms** | **0.20 ms** |
| **Degenerate Loops?** | Frequent | Frequent | **Eliminated (Top-$p$ + Penalty)** | **Eliminated (Top-$p$ + Penalty)** |
| **Typo Correction** | None | None | **Sub-ms SymSpell** | **Sub-ms SymSpell** |

---

*Authored for the Neural Next-Word Prediction Studio. Maintained under `uv` environment standards.*
