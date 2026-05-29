"""Example — analyze a batch of words and print their segmentations.

Run::

    python examples/02_batch.py
"""
from tugamorph import PortugueseMorphAnalyzer


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    words = [
        "desfazer",        # prefix + verbal infinitive
        "infelizmente",    # prefix + root + adverb suffix
        "casinha",         # diminutive
        "biologia",        # scientific suffix beats the bi- prefix
        "nacionalismo",    # abstract-noun suffix
        "disseram",        # suppletive irregular, caught whole
    ]

    results = analyzer.analyze_batch(words)
    for r in results:
        print(f"{r.original:14} -> {r.segmentation_str()}")


if __name__ == "__main__":
    main()
