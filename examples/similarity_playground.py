"""
Similarity playground — understand how semantic similarity works.

No LLM required. Runs entirely locally. Built for Privacy

Run with:
  python examples/similarity_playground.py
"""
from diffprompt.core.embedder import similarity, batch_similarity


def main():
    print("Similarity Playground")
    print("=" * 40)
    print("Scale: 0.0 = completely different  |  1.0 = identical meaning\n")

    pairs = [
        # Identical
        ("Paris is the capital of France",
         "Paris is the capital of France"),

        # Same meaning, different words
        ("Paris is the capital of France",
         "France's capital city is Paris"),

        # Related topic, different meaning
        ("Paris is the capital of France",
         "London is the capital of the United Kingdom"),

        # Completely unrelated
        ("Paris is the capital of France",
         "The quarterly earnings report exceeded expectations"),

        # Emotional vs factual versions of same topic
        ("I've been struggling with anxiety and don't know what to do",
         "Anxiety is a mental health condition characterized by worry"),

        # Brief vs verbose answer
        ("Paris.",
         "The capital of France is Paris, a major European city known for the Eiffel Tower."),

        # v1 vs v2 style outputs
        ("That sounds really difficult. Let me help you think through this carefully.",
         "Try meditation."),
    ]

    scores = batch_similarity(pairs)

    for (a, b), score in zip(pairs, scores):
        bar = "█" * int(score * 20)
        print(f"  {score:.2f}  {bar}")
        print(f"       A: \"{a[:60]}\"")
        print(f"       B: \"{b[:60]}\"")
        print()


if __name__ == "__main__":
    main()