"""
05 — Logits and Softmax: Raw Scores → Probabilities

The transformer's output is a vector of ~128k raw scores called "logits" —
one per entry in the vocabulary. These aren't probabilities yet.

Key ideas:
  - Logits can be any real number (-∞ to +∞)
  - Softmax converts logits to probabilities: P(token_i) = e^(logit_i) / Σ e^(logit_j)
  - After softmax, all values are between 0 and 1, and they sum to 1
  - We use Ollama's `logprobs` and `top_logprobs` to see the actual
    log-probabilities the model assigns to each candidate token.

The shape of the distribution tells us how "confident" the model is:
  - Sharp/peaked = model is very sure (one token dominates)
  - Flat/spread = model is uncertain (many tokens are likely)
"""

import math
import requests

URL = "http://localhost:11434"
MODEL = "llama3.1"


def generate_with_logprobs(prompt, top_n=10, raw=False, **opts):
    """Generate 1 token and return the top_n token probabilities."""
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "raw": raw, "logprobs": True, "top_logprobs": top_n,
        "options": {"num_predict": 1, **opts},
    })
    data = resp.json()
    chosen = data["response"]
    # logprobs[0] = info about the first (only) generated token
    top = data["logprobs"][0]["top_logprobs"]
    # Convert log-probabilities to probabilities: prob = e^logprob
    results = [(entry["token"], math.exp(entry["logprob"])) for entry in top]
    return chosen, results


def ascii_bar(prob, width=30):
    filled = int(prob * width)
    return "█" * filled + "░" * (width - filled)


def run():
    print("=" * 60)
    print("05 — LOGITS & SOFTMAX: Raw Scores → Probabilities")
    print("=" * 60)

    # ── Explain the math ────────────────────────────────────────
    print("""
  THE MATH:

  The transformer outputs raw logits for each vocabulary entry:
    logits = [3.2, 1.1, 0.5, -2.0, 7.8, ...]  (128,256 values)

  Softmax converts these to probabilities:
    P(token_i) = e^(logit_i) / Σ e^(logit_j)

  Example with 4 tokens:
    logits  = [2.0,  1.0,  0.5,  -1.0]
    e^logit = [7.39, 2.72, 1.65, 0.37]   (exponentiate each)
    sum     = 12.13
    probs   = [0.61, 0.22, 0.14, 0.03]   (divide by sum)
                ↑ highest logit → highest probability

  Properties:
    - All probabilities ∈ [0, 1]
    - They sum to exactly 1.0
    - The relative ordering is preserved (higher logit = higher prob)
    - The differences are amplified (softmax is "soft" argmax)

  Ollama gives us log-probabilities (logprobs) for each token.
  We convert back: probability = e^(logprob).
""")

    # ── Confident prediction ────────────────────────────────────
    print("  EXPERIMENT 1: Confident prediction (sharp distribution)")
    print('  Prompt: "The capital of France is"\n')

    chosen, top_tokens = generate_with_logprobs("The capital of France is", top_n=10)

    print(f"  {'Token':<15} {'LogProb':>8} {'Prob':>7}  Distribution")
    print(f"  {'─' * 15} {'─' * 8} {'─' * 7}  {'─' * 30}")
    for tok, prob in top_tokens:
        logprob = math.log(prob) if prob > 0 else float('-inf')
        bar = ascii_bar(prob)
        print(f"  {tok:<15} {logprob:>8.3f} {prob:>6.1%}  {bar}")

    print(f"\n  → The model is very confident: \"{top_tokens[0][0]}\" dominates at {top_tokens[0][1]:.0%}.")
    print(f"    The top logit is much higher than the rest → sharp peak after softmax.")

    # ── Ambiguous prediction ────────────────────────────────────
    # We use raw mode here so the model does pure text completion
    # (no chat template), which gives a genuinely ambiguous continuation.
    print(f"\n  EXPERIMENT 2: Ambiguous prediction (flat distribution)")
    print(f'  Prompt: "I want to"\n')

    chosen, top_tokens = generate_with_logprobs("I want to", top_n=10, raw=True)

    print(f"  {'Token':<15} {'LogProb':>8} {'Prob':>7}  Distribution")
    print(f"  {'─' * 15} {'─' * 8} {'─' * 7}  {'─' * 30}")
    for tok, prob in top_tokens:
        logprob = math.log(prob) if prob > 0 else float('-inf')
        bar = ascii_bar(prob)
        print(f"  {tok:<15} {logprob:>8.3f} {prob:>6.1%}  {bar}")

    top_prob = top_tokens[0][1]
    print(f"\n  → No single token dominates (top token is only {top_prob:.0%}).")
    print(f"    The logits are closer together → flatter distribution after softmax.")

    # ── Why logprobs instead of probabilities? ──────────────────
    print("""
  WHY LOG-PROBABILITIES?

  Models work with logprobs because:
    - Probabilities can be astronomically small (1e-30) → underflow
    - Log-probs are numerically stable: log(1e-30) = -69, easy to store
    - Multiplying probabilities = adding logprobs (faster & more stable)

  Converting: prob = e^(logprob),  logprob = ln(prob)
    logprob = 0.0    → prob = 100%  (certain)
    logprob = -0.7   → prob ≈  50%
    logprob = -2.3   → prob ≈  10%
    logprob = -4.6   → prob ≈   1%
    logprob = -∞     → prob =   0%  (impossible)
""")

    print("  The logits → softmax → probabilities pipeline determines")
    print("  how the model expresses its confidence (or lack thereof).")
    print()


if __name__ == "__main__":
    run()
