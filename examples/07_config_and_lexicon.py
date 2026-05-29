"""Example — tune the pipeline with AnalysisConfig and extend the lexicon.

Run::

    python examples/07_config_and_lexicon.py
"""
from tugamorph import (
    PortugueseMorphAnalyzer,
    AnalysisConfig,
    PREFIX_TABLE,
    PrefixCategory,
)


def main() -> None:
    word = "desfazer"

    print("Default config")
    print("--------------")
    default = PortugueseMorphAnalyzer()
    print(f"{word} -> {default.segment(word)}")

    print()
    print("Prefix extraction disabled")
    print("--------------------------")
    no_prefix = PortugueseMorphAnalyzer(AnalysisConfig(extract_prefixes=False))
    print(f"{word} -> {no_prefix.segment(word)}")

    print()
    print("Extend the lexicon")
    print("------------------")
    # Register a new prefix, then build an analyzer that sees it.
    PREFIX_TABLE.append(("proto", PrefixCategory.TEMPORAL))
    extended = PortugueseMorphAnalyzer()
    print(f"prototipo -> {extended.segment('prototipo')}")

    # Patch a per-instance whole-word irregular entry.
    extended._irregular_whole_words["deram"] = ("dar", "pret_perf", 3, "pl")
    r = extended.analyze("deram")
    print(f"deram -> lemma_guess={r.verbal.lemma_guess} tense={r.verbal.tense_mood}")


if __name__ == "__main__":
    main()
