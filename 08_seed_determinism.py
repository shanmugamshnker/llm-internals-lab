"""
08 — Seed & Determinism: Proving Where Randomness Lives

A transformer is a pure mathematical function — given the same input, it
ALWAYS produces the same logits. The only source of randomness is the
sampling step (the "dice roll" that picks a token from the distribution).

Key ideas:
  - The neural network (embed → attention → FFN → logits) is deterministic.
  - Randomness enters ONLY at sampling: when we pick from the probability distribution.
  - A "seed" initializes the random number generator (RNG) used for sampling.
  - Same seed + same prompt + same temperature = identical output, every time.
  - Different seed = different dice rolls = different output.
  - At temperature ≈ 0, there's effectively no randomness (always picks the top token).
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
    print("08 — SEED & DETERMINISM: Where Does Randomness Live?")
    print("=" * 60)

    prompt = "The secret to a good life is"

    # ── Experiment 1: Same seed = same output ───────────────────
    print(f'\n  Prompt: "{prompt}"')
    print(f"\n  EXPERIMENT 1: Same seed → identical output\n")

    seed = 42
    outputs = []
    for i in range(3):
        resp = generate(prompt, temperature=1.0, seed=seed, num_predict=15)
        text = resp["response"].strip()
        outputs.append(text)
        match = "✓ matches" if i > 0 and text == outputs[0] else ""
        print(f"    Run {i+1} (seed={seed}): {text[:60]}")
        if match:
            print(f"                         {match}")

    all_same = len(set(outputs)) == 1
    print(f"\n    All identical? {'YES' if all_same else 'NO'}")
    if all_same:
        print("    The RNG produced the same sequence of dice rolls each time.")

    # ── Experiment 2: Different seeds = different output ────────
    print(f"\n  EXPERIMENT 2: Different seeds → different outputs\n")

    seed_outputs = []
    for seed in [1, 2, 3, 4, 5]:
        resp = generate(prompt, temperature=1.0, seed=seed, num_predict=15)
        text = resp["response"].strip()
        seed_outputs.append(text)
        print(f"    Seed {seed}: {text[:60]}")

    unique = len(set(seed_outputs))
    print(f"\n    {unique} unique outputs from 5 seeds.")
    print("    Different seed = different RNG state = different token choices.")

    # ── Experiment 3: Temperature 0 needs no seed ───────────────
    # At temperature ≈ 0, softmax becomes argmax — the model always picks
    # the single most likely token. No randomness, no need for a seed.

    print(f"\n  EXPERIMENT 3: Temperature ≈ 0 (no randomness)\n")

    greedy_outputs = []
    for i in range(3):
        resp = generate(prompt, temperature=0.01, num_predict=15)
        text = resp["response"].strip()
        greedy_outputs.append(text)
        print(f"    Run {i+1} (temp=0.01, no seed): {text[:60]}")

    all_same = len(set(greedy_outputs)) == 1
    print(f"\n    All identical? {'YES' if all_same else 'NO (slight numerical noise possible)'}")
    print("    With temp≈0, the top token always wins — sampling is deterministic.")

    # ── Where randomness lives ──────────────────────────────────
    print("""
  SUMMARY: The Randomness Map

  ┌────────────────────────────────────────────────────────┐
  │  Tokenization     │  Deterministic (same text = same IDs)
  │  Embedding        │  Deterministic (table lookup)
  │  Attention        │  Deterministic (matrix math)
  │  Feed-Forward     │  Deterministic (matrix math)
  │  Logits           │  Deterministic (always the same scores)
  │  Softmax          │  Deterministic (same logits = same probs)
  │─────────────────────────────────────────────────────────
  │  SAMPLING         │  ★ RANDOM ★ (controlled by seed + temp)
  └────────────────────────────────────────────────────────┘

  Everything above the line is pure math — same input, same output.
  The ONLY randomness is in picking a token from the distribution.
  Control the seed → control the randomness → reproduce outputs.
""")


if __name__ == "__main__":
    run()
