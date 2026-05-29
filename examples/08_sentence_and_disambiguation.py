"""Example — sentence-level analysis and POS-driven disambiguation.

Run::

    python examples/08_sentence_and_disambiguation.py
"""
from tugamorph import PortugueseMorphAnalyzer


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    print("Per-word analysis of a sentence")
    print("-------------------------------")
    sentence = "O gato comeria o peixe"
    for r in analyzer.analyze_sentence(sentence):
        print(f"{r.original:10} {r.pos_tag or '?':6} {r.segmentation_str()}")

    print()
    print("A trusted POS tag overrides the verbal/suffix tie")
    print("-------------------------------------------------")
    word = "cantaria"
    for tag in (None, "VERB", "NOUN"):
        r = analyzer.analyze(word, pos_tag=tag)
        reading = "verbal" if r.verbal and r.verbal.tense_mood != "none" else "suffix"
        print(f"pos_tag={str(tag):5} -> {reading:6} {r.segmentation_str()}")


if __name__ == "__main__":
    main()
