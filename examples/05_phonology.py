"""Example — syllables, stress, and diphthong detection.

Run::

    python examples/05_phonology.py
"""
from tugamorph import PortugueseMorphAnalyzer


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    words = ["casa", "cantar", "rápido", "pão", "pai", "saída"]
    header = f"{'word':10} {'syllables':28} {'stress':14} flags"
    print(header)
    print("-" * len(header))

    for w in words:
        p = analyzer.analyze(w).phonology
        flags = []
        if p.has_nasal_diphthong:
            flags.append("nasal-diphthong")
        if p.has_oral_diphthong:
            flags.append("oral-diphthong")
        if p.has_hiatus:
            flags.append("hiatus")
        syl = ".".join(p.syllables)
        print(f"{w:10} {syl:28} {str(p.stress_pattern):14} {', '.join(flags) or '-'}")


if __name__ == "__main__":
    main()
