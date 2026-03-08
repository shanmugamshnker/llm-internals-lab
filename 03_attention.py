"""
03 — Attention: How Tokens Look at Each Other

Attention is the core mechanism that makes transformers powerful. It lets each
token "look at" every other token in the sequence and decide which ones are
relevant for predicting the next word.

Key ideas:
  - Each token creates three vectors: Query (Q), Key (K), Value (V)
  - Q asks "what am I looking for?", K says "what do I contain?", V says "what do I offer?"
  - Attention score = how well Q and K match (dot product)
  - High score → that token's Value gets more influence on the output
  - This is why "bank" means different things in different contexts — it
    attends to different surrounding words.

We can't directly see attention weights through the Ollama API, but we CAN
demonstrate the *effect* of attention by showing how context changes predictions.
We use `logprobs` to see the actual probability distribution for each context.
"""

import math
import requests

URL = "http://localhost:11434"
MODEL = "llama3.1"


def get_top_predictions(prompt, top_n=5):
    """Get the top predicted next tokens with their probabilities.
    Uses raw mode for pure text completion (no chat template)."""
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False,
        "raw": True, "logprobs": True, "top_logprobs": top_n,
        "options": {"num_predict": 1},
    })
    data = resp.json()
    top = data["logprobs"][0]["top_logprobs"]
    return [(entry["token"], math.exp(entry["logprob"])) for entry in top]


def show_predictions(prompt, top_n=5):
    """Display the top predictions for a prompt."""
    preds = get_top_predictions(prompt, top_n)
    parts = []
    for tok, prob in preds:
        parts.append(f"{tok.strip()}({prob:.0%})")
    print(f"    → Top predictions: {', '.join(parts)}")


def run():
    print("=" * 60)
    print("03 — ATTENTION: How Tokens Look at Each Other")
    print("=" * 60)

    # ── Conceptual explanation ──────────────────────────────────
    print("""
  HOW SELF-ATTENTION WORKS (simplified):

  Input: "The cat sat on the mat"

  For the word "sat", attention computes:
    - Q("sat") · K("The")  = 0.1  (low — "The" isn't very relevant)
    - Q("sat") · K("cat")  = 0.8  (high — "cat" is the subject doing the sitting)
    - Q("sat") · K("on")   = 0.3  (medium — preposition matters somewhat)
    - Q("sat") · K("mat")  = 0.5  (medium — location context)

  These scores become weights (via softmax), then we take a weighted
  sum of the Value vectors:

    output = 0.05·V("The") + 0.47·V("cat") + 0.18·V("on") + 0.30·V("mat")

  Result: "sat" now carries information about WHO sat and WHERE.
  This happens for every token, at every layer, in every attention head.
""")

    # ── Demo: Context changes meaning ───────────────────────────
    # The word "bank" means completely different things depending on context.
    # The attention mechanism is what allows the model to disambiguate.
    # We use logprobs to see the actual probability distribution.

    print("  Demo: Same word, different context → different predictions\n")

    context_pairs = [
        ("I walked along the river bank and saw a",
         "I went to the bank to deposit my"),
        ("The bat flew out of the cave into the",
         "He picked up the baseball bat and hit the"),
        ("She played a note on the piano and the",
         "She left a note on the kitchen"),
    ]

    for prompt_a, prompt_b in context_pairs:
        print(f'    Context A: "{prompt_a}..."')
        show_predictions(prompt_a)

        print(f'    Context B: "{prompt_b}..."')
        show_predictions(prompt_b)
        print()

    # ── Multi-head attention ────────────────────────────────────
    print("  WHY MULTIPLE HEADS?")
    print("  Llama 3.1 8B uses 32 attention heads per layer.")
    print("  Each head can attend to different types of relationships:")
    print("    - Head 1 might track subject-verb agreement")
    print("    - Head 2 might track positional proximity")
    print("    - Head 3 might track semantic similarity")
    print("  Together, they capture rich patterns in language.")
    print()

    # ── Long-range dependencies ─────────────────────────────────
    print("  Demo: Long-range dependencies\n")
    print("  Attention can connect tokens that are far apart:\n")

    short = "The dog barked"
    long_ctx = "The dog, who had been sleeping peacefully in the corner of the living room all afternoon, suddenly barked"

    print(f'    Short: "{short}..."')
    show_predictions(short)

    print(f'    Long:  "{long_ctx[:50]}..."')
    show_predictions(long_ctx)

    print()
    print("  Even with many words between 'dog' and 'barked', the model keeps")
    print("  track of who is doing what — thanks to attention.")
    print()


if __name__ == "__main__":
    run()
