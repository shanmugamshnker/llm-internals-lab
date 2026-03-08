"""
LLM Internals — Comprehensive Learning Project

See how text generation actually works inside a large language model,
step by step, using practical experiments with local Ollama (llama3.1).

Run all demos:  uv run main.py
Run one demo:   uv run 01_tokenization.py
"""

import importlib
import sys

DEMOS = [
    ("01_tokenization",      "Tokenization"),
    ("02_embeddings",        "Embeddings"),
    ("03_attention",         "Attention"),
    ("04_forward_pass",      "Forward Pass"),
    ("05_logits_and_softmax", "Logits & Softmax"),
    ("06_sampling",          "Sampling Strategies"),
    ("07_autoregressive",    "Autoregressive Generation"),
    ("08_seed_determinism",  "Seed & Determinism"),
]


def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         LLM INTERNALS — PRACTICAL DEMO SUITE           ║")
    print("║         Model: llama3.1 via local Ollama                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    for i, (module_name, title) in enumerate(DEMOS, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{i}/{len(DEMOS)}] {title}")
        print(f"{'─' * 60}\n")

        try:
            mod = importlib.import_module(module_name)
            mod.run()
        except Exception as e:
            print(f"  ERROR in {module_name}: {e}")
            print(f"  Skipping to next demo...\n")

        if i < len(DEMOS):
            print()

    # ── Final summary ───────────────────────────────────────────
    print("\n" + "═" * 60)
    print("""
  COMPLETE LLM PIPELINE:

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
""")
    print("═" * 60)
    print("  Done! All 8 demos completed.")
    print("═" * 60)


if __name__ == "__main__":
    main()
