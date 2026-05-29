"""Example — verbal morphology and suppletive irregular forms.

Run::

    python examples/03_verbs_and_irregulars.py
"""
from tugamorph import PortugueseMorphAnalyzer


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    verbs = ["cantar", "comemos", "partiram", "disseram", "fizemos", "vou"]
    for w in verbs:
        r = analyzer.analyze(w)
        v = r.verbal
        if v is None:
            print(f"{w:12} (no verbal reading) -> {r.segmentation_str()}")
            continue
        lemma = f" lemma={v.lemma_guess}" if v.is_irregular else ""
        conj = v.conjugation_class if v.conjugation_class else "?"
        print(
            f"{w:12} {v.tense_mood:12} "
            f"person={v.person or '?'} number={v.number or '?'} "
            f"conj={conj} irregular={v.is_irregular}{lemma}"
        )


if __name__ == "__main__":
    main()
