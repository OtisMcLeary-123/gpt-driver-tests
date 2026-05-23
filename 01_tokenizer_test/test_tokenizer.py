"""
01_tokenizer_test — GPT Tokenizer Behavior on Coordinate Strings

Reproduces the analysis from GPT-Driver Section 3.2:
The GPT tokenizer splits coordinate numbers inconsistently, meaning
nearby numerical values can have very different token representations.
This test visualizes and verifies that behavior.
"""

import tiktoken

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


def tokenize(text: str) -> list[str]:
    """Return human-readable token strings for a given text."""
    return [enc.decode([t]) for t in enc.encode(text)]


def test_coordinate_tokenization():
    """Show how the tokenizer splits typical trajectory coordinates."""
    coords = [
        "(0.12, 2.98)",
        "(1.45, 5.67)",
        "(3.45, 18.90)",
        "(12.05, 4.12)",
        "(-2.34, 0.01)",
    ]

    print("=== Coordinate Tokenization ===")
    for c in coords:
        tokens = tokenize(c)
        print(f"  {c:20s} -> {tokens}  ({len(tokens)} tokens)")


def test_nearby_values_different_tokens():
    """
    Key insight: numerically close values can tokenize very differently.
    E.g., 1.99 vs 2.00 — a 0.01 difference may produce different token counts.
    """
    pairs = [
        ("1.99", "2.00"),
        ("9.99", "10.00"),
        ("19.9", "20.0"),
        ("0.09", "0.10"),
    ]

    print("\n=== Nearby Values — Token Inconsistency ===")
    for a, b in pairs:
        ta, tb = tokenize(a), tokenize(b)
        print(f"  {a:6s} -> {str(ta):30s} ({len(ta)} tok)  |  {b:6s} -> {str(tb):30s} ({len(tb)} tok)")


def test_token_count_vs_precision():
    """More decimal places = more tokens, showing cost of precision."""
    values = ["2.3", "2.34", "2.345", "2.3456"]

    print("\n=== Precision vs Token Count ===")
    for v in values:
        tokens = tokenize(v)
        print(f"  {v:8s} -> {str(tokens):30s} ({len(tokens)} tokens)")


def test_trajectory_total_tokens():
    """
    Count total tokens for a full 6-waypoint trajectory (as in the paper).
    The paper uses t=6 future timesteps with (x, y) coordinates.
    """
    trajectory = "[(0.12, 2.98), (0.45, 5.67), (1.23, 8.45), (2.10, 11.30), (2.89, 14.56), (3.45, 18.90)]"
    tokens = enc.encode(trajectory)
    print(f"\n=== Full Trajectory Token Cost ===")
    print(f"  Trajectory: {trajectory}")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Tokens: {tokenize(trajectory)}")


if __name__ == "__main__":
    test_coordinate_tokenization()
    test_nearby_values_different_tokens()
    test_token_count_vs_precision()
    test_trajectory_total_tokens()
