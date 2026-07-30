"""Example — export analyses as feature dicts, vectors, and JSON for ML.

Run::

    python examples/06_feature_export.py
"""
from tugamorph import PortugueseMorphAnalyzer


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    word = "dir-se-ia"
    r = analyzer.analyze(word)

    print("Feature dict (selected keys)")
    print("----------------------------")
    d = r.to_feature_dict()
    for key in ["word", "root", "has_clitic", "clitic_position",
                "is_irregular", "lemma_guess", "tense_mood", "stress_pattern"]:
        print(f"  {key:18} {d.get(key)}")

    print()
    print("17-dim feature vector")
    print("---------------------")
    print(" ", r.to_feature_vector())

    print()
    print("JSON")
    print("----")
    print(analyzer.analyze("infelizmente").to_json())


if __name__ == "__main__":
    main()
