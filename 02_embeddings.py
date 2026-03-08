"""
02 — Embeddings: Tokens → Vectors

After tokenization, each token ID is looked up in an "embedding table" — a giant
matrix where each row is a high-dimensional vector (4096 dimensions for Llama 3.1 8B).

Key ideas:
  - Each token gets a dense vector that encodes its meaning.
  - Similar meanings → vectors point in similar directions (high cosine similarity).
  - Unrelated meanings → vectors point in different directions (low cosine similarity).
  - The embedding table is learned during training — it's part of the model's parameters.

We use Ollama's /api/embed endpoint to get the final-layer embedding for whole
sentences (the model processes all tokens and gives us a single pooled vector).
"""

import math
import requests

URL = "http://localhost:11434"
MODEL = "llama3.1"


def embed(text):
    """Get the embedding vector for a text string."""
    resp = requests.post(f"{URL}/api/embed", json={
        "model": MODEL, "input": text,
    })
    return resp.json()["embeddings"][0]


def cosine_sim(a, b):
    """
    Cosine similarity: measures the angle between two vectors.
      cos(θ) = (A · B) / (|A| × |B|)
    Result ranges from -1 (opposite) through 0 (unrelated) to 1 (identical).
    """
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b)


def cosine_sim_verbose(a, b, label_a, label_b, show_steps=5):
    """Cosine similarity with step-by-step calculation shown."""
    print(f'    Calculating: "{label_a}" vs "{label_b}"\n')

    # Step 1 — Dot product
    print(f"    Step 1 — Dot product (first {show_steps} of {len(a)} multiplications):")
    for i in range(show_steps):
        product = a[i] * b[i]
        print(f"      a[{i}] × b[{i}] = {a[i]:.4f} × {b[i]:.4f} = {product:.8f}")
    print(f"      ... + {len(a) - show_steps} more ...")
    dot = sum(x * y for x, y in zip(a, b))
    print(f"      Sum of all {len(a)} products = {dot:.4f}")

    # Step 2 — Magnitudes
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    print(f"\n    Step 2 — Magnitudes:")
    print(f"      |A| = sqrt(a[0]² + a[1]² + ... + a[{len(a)-1}]²) = {mag_a:.4f}")
    print(f"      |B| = sqrt(b[0]² + b[1]² + ... + b[{len(b)-1}]²) = {mag_b:.4f}")

    # Step 3 — Divide
    result = dot / (mag_a * mag_b)
    print(f"\n    Step 3 — Divide:")
    print(f"      cosine = {dot:.4f} / ({mag_a:.4f} × {mag_b:.4f})")
    print(f"             = {dot:.4f} / {mag_a * mag_b:.4f}")
    print(f"             = {result:.4f}")

    bar_len = max(0, int(result * 20))
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"\n    Result: {result:.3f} [{bar}]")
    print()
    return result


def run():
    print("=" * 60)
    print("02 — EMBEDDINGS: Tokens → Vectors")
    print("=" * 60)

    # ── Get embeddings for several sentences ────────────────────
    texts = [
        "The king sat on his throne",
        "The queen sat on her throne",
        "I need to buy groceries",
        "The weather is nice today",
        "The monarch ruled the kingdom",
    ]

    print("\n  Converting sentences to vectors...\n")

    vectors = {}
    for t in texts:
        v = embed(t)
        vectors[t] = v
        # Show just a tiny slice of the 4096-dimensional vector
        preview = [round(x, 4) for x in v[:5]]
        print(f'  "{t}"')
        print(f"    → {len(v)} dimensions, first 5: {preview}")
        print()

    # ── Cosine similarity matrix ────────────────────────────────
    # This is the core insight: semantic similarity is encoded in vector geometry.
    # "king/throne" and "queen/throne" should be very similar.
    # "king/throne" and "buy groceries" should be dissimilar.

    # ── Detailed calculation for the first pair ─────────────────
    # Show the full step-by-step math so you can see exactly how
    # cosine similarity works with real numbers.

    print("  COSINE SIMILARITY — Step-by-step calculation:\n")
    print("  Formula: cos(θ) = (A · B) / (|A| × |B|)\n")

    a_text, b_text = "The king sat on his throne", "The queen sat on her throne"
    cosine_sim_verbose(vectors[a_text], vectors[b_text], a_text, b_text)

    # ── Compact results for all pairs ─────────────────────────
    print("  All pairs (1.0 = identical meaning, 0.0 = unrelated):\n")

    pairs = [
        ("The king sat on his throne",   "The queen sat on her throne"),
        ("The king sat on his throne",   "The monarch ruled the kingdom"),
        ("The king sat on his throne",   "I need to buy groceries"),
        ("The queen sat on her throne",  "The weather is nice today"),
        ("I need to buy groceries",      "The weather is nice today"),
    ]

    for a, b in pairs:
        sim = cosine_sim(vectors[a], vectors[b])
        bar_len = max(0, int(sim * 20))
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f'    {sim:.3f} [{bar}]')
        print(f'      "{a}"')
        print(f'      "{b}"')
        print()

    # ── What does a dimension mean? ─────────────────────────────
    print("  What do these 4096 numbers mean?")
    print("  No single dimension has a clear meaning like 'royalty' or 'food'.")
    print("  Instead, meaning is distributed across all dimensions together.")
    print("  The model learned these representations during training to capture")
    print("  patterns in billions of words of text.")
    print()


if __name__ == "__main__":
    run()
