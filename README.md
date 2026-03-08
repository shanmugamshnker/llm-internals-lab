# LLM Internals

Learn how large language models actually work from tokenization to text generation through practical experiments running on your local machine.

No GPU required. No PyTorch. Just Python + Ollama.

## What You'll Learn

| # | Script | Concept |
|---|--------|---------|
| 01 | `01_tokenization.py` | How text becomes numbers (token IDs, subword splitting) |
| 02 | `02_embeddings.py` | How tokens become vectors, cosine similarity |
| 03 | `03_attention.py` | How tokens look at each other, context-dependent meaning |
| 04 | `04_forward_pass.py` | Full transformer pipeline, parameter counts, timing |
| 05 | `05_logits_and_softmax.py` | Raw scores → probabilities, model confidence |
| 06 | `06_sampling.py` | Temperature, top-k, top-p — controlling randomness |
| 07 | `07_autoregressive.py` | Token-by-token generation, the butterfly effect |
| 08 | `08_seed_determinism.py` | Proving the neural network is deterministic |

## Prerequisites

1. **Install Ollama**: https://ollama.com
2. **Pull the model**:
   ```bash
   ollama pull llama3.1
   ```
3. **Install uv** (Python package manager): https://docs.astral.sh/uv/

## Run

Run all demos in sequence:
```bash
uv run main.py
```

Run a single demo:
```bash
uv run 01_tokenization.py
```

## The Big Picture

```
"I like to eat"
     │
     ▼
1. TOKENIZE      → [I, like, to, eat]     (text → integer IDs)
     │
     ▼
2. EMBED          → 4096-dim vector/token  (IDs → vectors)
     │
     ▼
3. ATTENTION      → tokens exchange info   (who relates to whom?)
     │
     ▼
4. FEED-FORWARD   → process each token     (×32 transformer layers)
     │
     ▼
5. LOGITS         → 128k raw scores        (one per vocab entry)
     │
     ▼
6. SOFTMAX        → probabilities          (scores → 0-1, sum to 1)
     │
     ▼
7. SAMPLE         → pick one token         (★ ONLY source of randomness)
     │
     ▼
8. APPEND & LOOP  → "I like to eat pizza"  (repeat from step 3)
```

## Dependencies

Just `requests`. No ML libraries needed, we use Ollama's API for everything.
