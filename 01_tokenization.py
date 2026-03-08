"""
01 — Tokenization: Text → Token IDs

LLMs never see raw text. Before anything else, text is split into "tokens"
— subword pieces — and each piece gets an integer ID from a fixed vocabulary.

Key ideas:
  - A token can be a whole word ("the"), a subword ("un" + "happiness"), or
    a single character. The tokenizer decides how to split.
  - Different text types (English prose, code, non-English) tokenize differently.
  - The token count determines how much compute the model uses.

We use Ollama's `context` field (returned after generation) to see the actual
token IDs the model used internally. We strip out chat-template tokens
(like <|start_header_id|>) to show only the text tokens.
"""

import requests

URL = "http://localhost:11434"
MODEL = "llama3.1"

# Llama 3.1 special token IDs — used by the chat template, not part of user text.
SPECIAL_IDS = {128000, 128001, 128004, 128006, 128007, 128008, 128009, 128010, 128011}


def generate(prompt, **opts):
    resp = requests.post(f"{URL}/api/generate", json={
        "model": MODEL, "prompt": prompt, "stream": False, "options": opts,
    })
    return resp.json()


def extract_text_tokens(context, eval_count=1):
    """
    Extract just the user-text token IDs from Ollama's context array.

    The context looks like:
      [128006, 882, 128007, 271, <TEXT TOKENS>, 128009, 128006, 78191, 128007, 271, <GEN TOKENS>]
       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
       <|start_header_id|>user<|end_header_id|>\\n\\n   <|eot|><|start_header_id|>assistant...

    Strategy: find the first <|eot_id|> (128009) after the initial header.
    Text tokens are between the header (first 4 tokens) and that 128009.
    """
    # Skip the user header: <|start_header_id|>user<|end_header_id|>\n\n = 4 tokens
    content_start = 4
    # Find <|eot_id|> (128009) which marks end of user content
    try:
        content_end = context.index(128009, content_start)
    except ValueError:
        # Fallback: strip generated tokens from end and skip specials
        content_end = len(context) - eval_count
    return context[content_start:content_end]


def run():
    print("=" * 60)
    print("01 — TOKENIZATION: Text → Token IDs")
    print("=" * 60)

    # ── Basic tokenization ──────────────────────────────────────
    # We ask the model to predict just 1 token (num_predict=1) so it returns
    # fast. The `context` field in the response contains the token IDs for
    # the entire conversation — we extract just the user-text tokens.

    prompt = "The capital of France is"
    data = generate(prompt, num_predict=1)
    text_ids = extract_text_tokens(data["context"], data.get("eval_count", 1))

    print(f'\n  Input text:  "{prompt}"')
    print(f"  Token IDs:   {text_ids}")
    print(f"  Token count: {len(text_ids)}")
    print()
    print("  The model converted your text into these integer IDs.")
    print("  Each ID maps to a subword piece in the model's vocabulary (~128k entries).")

    # ── Compare different inputs ────────────────────────────────
    # Different types of text produce very different token counts.
    # Code tends to use more tokens per character because variable names
    # and symbols are less common in training data.

    print("\n  Comparing token counts across text types:\n")
    examples = [
        ("Short English",   "Hello world"),
        ("Longer English",  "The quick brown fox jumps over the lazy dog"),
        ("Python code",     "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
        ("Repeated text",   "ha " * 20),
        ("Numbers",         "3.14159265358979323846"),
    ]

    print(f"  {'Type':<18} {'Tokens':>6}  Text (first 50 chars)")
    print(f"  {'─' * 18} {'─' * 6}  {'─' * 50}")

    for label, text in examples:
        data = generate(text, num_predict=1)
        text_ids = extract_text_tokens(data["context"], data.get("eval_count", 1))
        count = len(text_ids)
        preview = text[:50] + ("..." if len(text) > 50 else "")
        print(f"  {label:<18} {count:>6}  {preview}")

    # ── Subword splitting ───────────────────────────────────────
    # Modern tokenizers use algorithms like BPE (Byte-Pair Encoding).
    # Common words stay whole, but rare/long words get split into pieces.
    # "unhappiness" might become ["un", "happi", "ness"] — each piece gets
    # its own token ID.

    print("\n  Subword splitting demo:")
    print("  The tokenizer breaks uncommon words into smaller known pieces.\n")

    words = ["hello", "unhappiness", "transformers", "counterintuitively", "123456789"]
    for word in words:
        data = generate(word, num_predict=1)
        word_ids = extract_text_tokens(data["context"], data.get("eval_count", 1))
        print(f'    "{word}" → {len(word_ids)} token(s): {word_ids}')

    print()
    print("  Takeaway: The model's \"alphabet\" is ~128k subword pieces.")
    print("  Everything — text, code, numbers — gets mapped to these pieces first.")
    print()


if __name__ == "__main__":
    run()
