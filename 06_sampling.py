"""
06 — Sampling Strategies: Temperature, Top-k, Top-p

Once the model produces a probability distribution over ~128k tokens,
we need to *choose* one token. This is "sampling" — and different strategies
produce very different outputs.

Key strategies:
  - Temperature: scales logits before softmax. Low = sharp, high = flat.
  - Top-k: only consider the k most likely tokens, zero out the rest.
  - Top-p (nucleus): only consider tokens whose cumulative probability ≤ p.
  - These can be combined (and usually are in practice).

We use Ollama's `logprobs` to see how sampling parameters reshape the
probability distribution, and sample multiple times to show the effect.
"""

import math
import requests
from collections import Counter

URL = "http://localhost:11434"
MODEL = "llama3.1"


def generate(prompt, **opts):
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False, "options": opts,
    })
    return resp.json()


def generate_with_logprobs(prompt, top_n=10, **opts):
    """Generate 1 token and return chosen token + top_n probabilities."""
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "logprobs": True, "top_logprobs": top_n,
        "options": {"num_predict": 1, **opts},
    })
    data = resp.json()
    chosen = data["response"].strip()
    top = data["logprobs"][0]["top_logprobs"]
    results = [(entry["token"], math.exp(entry["logprob"])) for entry in top]
    return chosen, results


def sample_many(prompt, n=10, **opts):
    """Sample n completions and return the results."""
    results = []
    for _ in range(n):
        resp = generate(prompt, num_predict=8, **opts)
        results.append(resp["response"].strip())
    return results


def show_distribution(top_tokens, label, max_show=5):
    """Show a probability distribution with ASCII bars."""
    print(f"    {label}")
    for tok, prob in top_tokens[:max_show]:
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"      {tok:<12} {prob:>5.1%} {bar}")
    print()


def show_samples(results, label):
    """Show sampled outputs with frequency."""
    counts = Counter(results)
    unique = len(counts)
    print(f"    {label}")
    print(f"    Unique outputs: {unique}/{len(results)}")
    for text, c in counts.most_common(4):
        preview = text[:45] + ("..." if len(text) > 45 else "")
        print(f"      {c:>2}× {preview}")
    print()


def run():
    print("=" * 60)
    print("06 — SAMPLING: Temperature, Top-k, Top-p")
    print("=" * 60)

    prompt = "Once upon a time there was a"

    # ── Temperature ─────────────────────────────────────────────
    # Formula: P(token) = softmax(logits / temperature)
    # temp < 1 → sharper distribution (more deterministic)
    # temp = 1 → original distribution
    # temp > 1 → flatter distribution (more random)

    print(f'\n  TEMPERATURE — Prompt: "{prompt}"\n')
    print("  Temperature scales logits before softmax:")
    print("    P(token) = softmax(logits / T)")
    print("    T→0: always pick the top token (greedy)")
    print("    T=1: sample from original distribution")
    print("    T→∞: uniform random over all tokens\n")

    # First, show the base distribution (temp=1.0) so we can see what
    # the model's "natural" probabilities look like.
    _, top_tokens = generate_with_logprobs(prompt, top_n=5)
    show_distribution(top_tokens, "Base distribution (T=1.0) — what temperature modifies:")

    # Now show how different temperatures change the sampling behavior
    for temp in [0.01, 0.5, 1.0, 2.0]:
        label_map = {
            0.01: "T=0.01 (nearly greedy — always picks the top token)",
            0.5:  "T=0.5  (focused — slight variety)",
            1.0:  "T=1.0  (natural — original distribution)",
            2.0:  "T=2.0  (wild — very random, often incoherent)",
        }
        results = sample_many(prompt, n=8, temperature=temp, top_k=0, top_p=1.0)
        show_samples(results, label_map[temp])

    # ── Top-k ───────────────────────────────────────────────────
    # Only keep the k tokens with highest probability.
    # All other tokens get probability = 0.
    # Then renormalize so probabilities sum to 1.

    print("  TOP-K — Restrict to the K most likely tokens\n")
    print("  Only the top K tokens can be chosen. All others get probability 0.")
    print("  Then renormalize so probabilities sum to 1.\n")

    for k in [1, 5, 50]:
        label = f"top_k={k:<3} ({'greedy' if k == 1 else 'narrow' if k <= 10 else 'moderate'})"
        results = sample_many(prompt, n=8, temperature=1.0, top_k=k)
        show_samples(results, label)

    # ── Top-p (nucleus sampling) ────────────────────────────────
    # Sort tokens by probability (descending).
    # Keep adding tokens until their cumulative probability ≥ p.
    # Only those tokens can be sampled.
    # Advantage over top-k: adapts to the shape of the distribution.

    print("  TOP-P (NUCLEUS) — Restrict to tokens covering P% of probability\n")
    print("  Sort tokens by probability, keep adding until cumulative ≥ P.")
    print("  Adapts to confidence: sharp distribution → fewer tokens kept.\n")

    for p in [0.1, 0.5, 0.9]:
        label = f"top_p={p:<3} ({'very focused' if p <= 0.2 else 'focused' if p <= 0.5 else 'moderate'})"
        results = sample_many(prompt, n=8, temperature=1.0, top_k=0, top_p=p)
        show_samples(results, label)

    # ── Summary ─────────────────────────────────────────────────
    print("  PRACTICAL COMBINATIONS:")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │  Factual Q&A:  temp=0.1, top_p=0.9             │")
    print("  │  Creative:     temp=0.8, top_p=0.95            │")
    print("  │  Code:         temp=0.2, top_p=0.9             │")
    print("  │  Brainstorm:   temp=1.2, top_k=50              │")
    print("  └──────────────────────────────────────────────────┘")
    print()


if __name__ == "__main__":
    run()
