"""
07 — Autoregressive Generation: Token by Token

LLMs generate text one token at a time. Each new token is appended to the
sequence, and then the entire sequence is fed back through the model to
predict the next one. This is the "autoregressive" loop.

Key ideas:
  - Generation is sequential: token N depends on all tokens 1..N-1.
  - Each token requires a full forward pass through the network.
  - A different choice at any step changes ALL subsequent tokens.
  - This is why generation is slow compared to prompt processing
    (which can be parallelized).
"""

import json
import requests

URL = "http://localhost:11434"
MODEL = "llama3.1"


def generate(prompt, **opts):
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False, "options": opts,
    })
    return resp.json()


def stream_generate(prompt, **opts):
    """Stream tokens one at a time, yielding each token string."""
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": True, "options": opts,
    }, stream=True)

    for line in resp.iter_lines():
        if line:
            data = json.loads(line)
            yield data["response"]
            if data.get("done"):
                break


def run():
    print("=" * 60)
    print("07 — AUTOREGRESSIVE GENERATION: Token by Token")
    print("=" * 60)

    # ── The autoregressive loop visualized ──────────────────────
    print("""
  THE LOOP:

  Prompt: "The best"
                    │
  Step 1: model("The best")           → "thing"
  Step 2: model("The best thing")     → "about"
  Step 3: model("The best thing about") → "learning"
  ...

  Each step = one full forward pass through all 32 transformer layers.
  The output token is appended, then the whole sequence is re-processed.
  (In practice, KV-caching avoids redundant computation for previous tokens.)
""")

    # ── Watch it happen ─────────────────────────────────────────
    prompt = "The most interesting thing about language models is"
    print(f'  Streaming generation: "{prompt}"\n')

    print(f"  {'Step':>4}  {'Token':<15}  Sequence so far")
    print(f"  {'─' * 4}  {'─' * 15}  {'─' * 45}")

    built = ""
    step = 0
    for tok in stream_generate(prompt, temperature=0.7, num_predict=25):
        built += tok
        step += 1
        display_tok = repr(tok)
        sequence_preview = built.strip()[:45]
        print(f"  {step:>4}  {display_tok:<15}  {sequence_preview}")

    print(f"\n  Total: {step} forward passes, each producing exactly 1 token.")

    # ── Divergence experiment ───────────────────────────────────
    # Run the same prompt twice (without a fixed seed) to show that
    # one different token choice causes the entire output to diverge.

    print("\n  DIVERGENCE EXPERIMENT")
    print("  Same prompt, two runs — watch where they split:\n")

    prompt = "In the year 2050, humanity will"
    print(f'  Prompt: "{prompt}"\n')

    runs = []
    for i in range(2):
        tokens = []
        for tok in stream_generate(prompt, temperature=0.8, num_predict=20):
            tokens.append(tok)
        runs.append(tokens)

    # Display side by side, highlighting the divergence point
    max_len = max(len(runs[0]), len(runs[1]))
    diverged = False
    diverge_step = None

    print(f"  {'Step':>4}  {'Run 1':<25}  {'Run 2':<25}")
    print(f"  {'─' * 4}  {'─' * 25}  {'─' * 25}")

    for i in range(max_len):
        t1 = runs[0][i] if i < len(runs[0]) else ""
        t2 = runs[1][i] if i < len(runs[1]) else ""
        marker = ""
        if not diverged and t1 != t2:
            diverged = True
            diverge_step = i + 1
            marker = " ← DIVERGED"
        print(f"  {i+1:>4}  {repr(t1):<25}  {repr(t2):<25}{marker}")

    if diverge_step:
        print(f"\n  Diverged at step {diverge_step}.")
    else:
        print(f"\n  (Runs happened to match — unusual with temp=0.8!)")

    print("  Once one token differs, the entire future sequence changes.")
    print("  This is the butterfly effect of autoregressive generation.")
    print()


if __name__ == "__main__":
    run()
