"""Example — clitic pronouns and hyphenated compounds.

Run::

    python examples/04_clitics_and_compounds.py
"""
from tugamorph import PortugueseMorphAnalyzer, CliticPosition


def main() -> None:
    analyzer = PortugueseMorphAnalyzer()

    print("Clitics")
    print("-------")
    for w in ["disse-me", "dir-se-ia", "viu-o"]:
        r = analyzer.analyze(w)
        c = r.clitic
        if c and c.position != CliticPosition.NONE:
            print(
                f"{w:12} clitic='{c.form}' {str(c.position):10} "
                f"person={c.person} number={c.number} case={c.case}  "
                f"-> {r.segmentation_str()}"
            )
        else:
            print(f"{w:12} (no clitic) -> {r.segmentation_str()}")

    print()
    print("Compounds")
    print("---------")
    for w in ["guarda-chuva", "couve-flor", "segunda-feira"]:
        r = analyzer.analyze(w)
        print(f"{w:14} compound={r.is_compound} parts={r.compound_parts}")


if __name__ == "__main__":
    main()
