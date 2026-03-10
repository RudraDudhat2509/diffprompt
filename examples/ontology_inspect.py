"""
Ontology inspection example — see what dimensions diffprompt infers for your prompt.

Useful for understanding how your prompt gets analyzed before running a full diff.

Run with:
  python examples/ontology_inspect.py
"""
import asyncio
from diffprompt.core.ontology import Ontology
from diffprompt.core.embedder import similarity


YOUR_PROMPT = "You are a helpful assistant. Answer clearly and completely."


async def main():
    print("Ontology Inspector")
    print("=" * 40)
    print(f"Prompt: {YOUR_PROMPT}\n")

    ontology = Ontology()

    print("Inferring dimensions...")
    await ontology.infer(YOUR_PROMPT)

    print("\nDimensions found:")
    for dimension, tags in ontology.dimensions.items():
        print(f"  {dimension}: {tags}")

    print("\nBuilding anchor sentences...")
    await ontology.build_anchors(YOUR_PROMPT)

    print("\nAnchor sentences:")
    for dimension, tag_anchors in ontology.anchors.items():
        print(f"\n  {dimension}:")
        for tag, anchor in tag_anchors.items():
            print(f"    {tag:15} → \"{anchor}\"")

    print("\nTagging test inputs:")
    test_inputs = [
        "What is the capital of France?",
        "I've been feeling really anxious lately",
        "Explain recursion like I'm five years old",
        "WHAT IS 2+2???",
        "Write a python function that sorts a list"
    ]

    for inp in test_inputs:
        tags = ontology.tag(inp)
        tag_str = "  ".join(f"{k}:{v}" for k, v in tags.items())
        print(f"  {inp[:45]:<45} → {tag_str}")


if __name__ == "__main__":
    asyncio.run(main())