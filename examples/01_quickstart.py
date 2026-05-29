"""Example — analyze one Portuguese word and read its parts.

Run::

    python examples/01_quickstart.py
"""
from tugamorph import PortugueseMorphAnalyzer


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    result = analyzer.analyze("extraordinariamente")
    print(result)
    print()

    # Reach into the structured result directly.
    print("root:       ", result.root)
    print("prefixes:   ", [(p[0], str(p[1])) for p in result.prefixes])
    print("suffix:     ", (result.suffix[0], str(result.suffix[1])) if result.suffix else None)
    print("stress:     ", str(result.phonology.stress_pattern))
    print("syllables:  ", result.phonology.syllables)

    # The convenience segmenter skips the object entirely.
    print()
    print("segment('impossibilidade'):", analyzer.segment("impossibilidade"))


if __name__ == "__main__":
    main()
