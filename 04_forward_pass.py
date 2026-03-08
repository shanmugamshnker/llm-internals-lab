"""
04 — Forward Pass: The Full Transformer Pipeline

When you send a prompt to an LLM, it goes through a specific sequence of
operations called the "forward pass." This is the complete journey from text
to the next predicted token.

The pipeline:
  1. Tokenize — text → token IDs
  2. Embed — token IDs → vectors (one 4096-dim vector per token)
  3. Transformer Blocks (×32 for Llama 3.1 8B):
     a. Self-Attention — tokens exchange information
     b. Feed-Forward Network (FFN) — process each token independently
     c. Layer Normalization — keep values stable
  4. Final Layer Norm
  5. Output Projection — vectors → logits (one score per vocab entry)
  6. Sampling — logits → probabilities → chosen token

We use Ollama's timing fields to measure how long each phase takes.
"""

import requests

URL = "http://localhost:11434"
MODEL = "llama3.1"


def generate(prompt, **opts):
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False, "options": opts,
    })
    return resp.json()


def run():
    print("=" * 60)
    print("04 — FORWARD PASS: The Full Transformer Pipeline")
    print("=" * 60)

    # ── Architecture overview ───────────────────────────────────
    print("""
  LLAMA 3.1 8B ARCHITECTURE:

  ┌─────────────────────────────────────────┐
  │  Input: "The capital of France is"      │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  1. TOKENIZER                           │
  │     "The" "capital" "of" "France" "is"  │
  │     → [791, 6864, 315, 9822, 374]       │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  2. EMBEDDING TABLE                     │
  │     128,256 entries × 4,096 dimensions  │
  │     Each token ID → 4096-dim vector     │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  3. TRANSFORMER BLOCKS (×32)            │
  │  ┌───────────────────────────────────┐  │
  │  │  RMSNorm                          │  │
  │  │  Self-Attention (32 heads)        │  │
  │  │  + Residual Connection            │  │
  │  │  RMSNorm                          │  │
  │  │  Feed-Forward Network             │  │
  │  │    Linear(4096 → 14336)           │  │
  │  │    SiLU activation                │  │
  │  │    Linear(14336 → 4096)           │  │
  │  │  + Residual Connection            │  │
  │  └───────────────────────────────────┘  │
  │          × 32 layers                    │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  4. FINAL RMSNorm                       │
  │  5. OUTPUT PROJECTION (4096 → 128,256)  │
  │     → One score (logit) per vocab entry │
  └─────────────────┬───────────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────────┐
  │  6. SAMPLING                            │
  │     softmax → probabilities → pick one  │
  │     → "Paris"                           │
  └─────────────────────────────────────────┘
""")

    # ── Parameter count breakdown ───────────────────────────────
    print("  PARAMETER COUNT — What '8B parameters' means:\n")

    params = [
        ("Embedding table",     128_256 * 4_096),
        ("Per transformer block:", 0),
        ("  Attention (Q,K,V,O)", 4 * 4_096 * 4_096),
        ("  FFN (gate+up+down)",  3 * 4_096 * 14_336),
        ("  RMSNorm (×2)",        2 * 4_096),
        ("× 32 blocks",          32 * (4 * 4_096 * 4_096 + 3 * 4_096 * 14_336 + 2 * 4_096)),
        ("Final RMSNorm",        4_096),
        ("Output projection",    4_096 * 128_256),
    ]

    total = 0
    for name, count in params:
        if count > 0:
            total += count if "×" not in name else 0  # avoid double-counting
            if "block" in name.lower() and "×" in name:
                total = count + 128_256 * 4_096 + 4_096 + 4_096 * 128_256
            billions = count / 1e9
            print(f"    {name:<30} {count:>15,} ({billions:.2f}B)")
        else:
            print(f"    {name}")

    print(f"\n    {'TOTAL':<30} {total:>15,} ({total/1e9:.2f}B)")
    print(f"\n    At float16: {total * 2 / 1e9:.1f} GB of GPU memory")
    print(f"    At Q4 (4-bit quantized): ~{total * 0.5 / 1e9:.1f} GB — fits on your laptop!")

    # ── Timing the forward pass ─────────────────────────────────
    print("\n  TIMING — How long does a forward pass take?\n")

    prompts = [
        ("Short prompt",  "Hi"),
        ("Medium prompt", "Explain the theory of relativity in simple terms"),
        ("Long prompt",   "Write a detailed analysis of the economic impacts of " * 5),
    ]

    for label, prompt in prompts:
        data = generate(prompt, num_predict=20)

        # Ollama returns timing in nanoseconds
        prompt_ns = data.get("prompt_eval_duration", 0)
        gen_ns = data.get("eval_duration", 0)
        prompt_count = data.get("prompt_eval_count", 0)
        gen_count = data.get("eval_count", 0)

        prompt_ms = prompt_ns / 1e6
        gen_ms = gen_ns / 1e6
        tokens_per_sec = (gen_count / (gen_ns / 1e9)) if gen_ns > 0 else 0

        print(f"    {label} ({prompt_count} input tokens → {gen_count} output tokens):")
        print(f"      Prompt processing: {prompt_ms:.0f}ms ({prompt_count / (prompt_ns/1e9) if prompt_ns else 0:.0f} tok/s)")
        print(f"      Generation:        {gen_ms:.0f}ms ({tokens_per_sec:.0f} tok/s)")
        print(f"      Time per token:    {gen_ms/gen_count:.0f}ms" if gen_count else "")
        print()

    print("  Key insight: prompt tokens are processed in parallel (fast),")
    print("  but generation is sequential — one forward pass per new token.")
    print()


if __name__ == "__main__":
    run()
