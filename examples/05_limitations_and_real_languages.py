"""05 - What this sandbox cannot show you (and what to do about it).

Read this after the four experimental scripts. It is the honest boundary of
the project: the gap between Kisima and real morphologically complex
languages, and the concrete next steps for closing it.

No code here beyond a short demonstration of the ways real languages break
the assumptions Kisima was built on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from morphgnn.kisima import build_verb, build_noun, generate_lexicon


def main():
    print("== 1. Allomorphy ==")
    # Kisima: a fixed affix per (slot, feature, harmony class).
    w = build_verb("makta", "1sg", "past", "3sg.obj", "ind", neg=True)
    print(f"  Kisima  {w.surface}  ({' + '.join(s for _, _, s in w.morphs if s)})")
    print("  Real:   Turkish 'el' -> 'elim' but 'kitap' -> 'kitabım' (k->ğ, p->b")
    print("          before a vowel-initial suffix). Allomorphy is conditioned by")
    print("          phonology, not merely by harmony class: your affix inventory")
    print("          becomes a function, and edges must be able to select variants.")

    print("\n== 2. Fusional exponence ==")
    n = build_noun("talen", "1sg.poss", "pl", "gen")
    print(f"  Kisima  {n.surface}  (one affix per slot, separable)")
    print("  Real:   Irish 'an bhfuil' eclipsis realises negation inside the")
    print("          verb; Latin 'amo/amavi/amatus' fuses aspect, voice, person.")
    print("          Segmentation is not given: boundary detection becomes")
    print("          segmentation + ambiguity, and paradigm graphs get multi-edge.")

    print("\n== 3. Suppletion and irregulars ==")
    rng_lex = generate_lexicon(seed=1, n_verbs=2, n_nouns=2)
    print("  Kisima  every form of a lexeme shares the root string:",
          list(rng_lex)[0], "->", rng_lex[list(rng_lex)[0]][0].surface)
    print("  Real:   Irish 'téigh' (go) vs 'chuaigh' (went); English go/went.")
    print("          Some edges must learn to say 'unpredictable', and a graph")
    print("          model needs an out-of-rule pathway or it will confidently")
    print("          regularise what speakers know is exceptional.")

    print("\n== 4. Data ==")
    print("  Kisima  unlimited, exact labels, one script, no noise.")
    print("  Real:   epistolary/spoken corpora are small, dialect-split, and")
    print("          inconsistently annotated. Available starting points:")
    print("            - SIGMORPHON shared-task data (reinflection, 60+ languages)")
    print("            - Universal Dependencies (dependency graphs for parsing,")
    print("              and the obvious bridge from example 01's chain edges)")
    print("            - GiellaLT / Apertium lexc transducers (real morphology rules)")
    print("            - Paninian and Inuit/Yupik materials for polysynthesis")

    print("\n== 5. What transfers, and what must be re-earned ==")
    print("  Transferable: the message-passing implementation, the habit of")
    print("  splitting by lexeme, the ablation pattern (edges on/off), and the")
    print("  control-condition mindset from example 04.")
    print("  Must be re-earned on real data: the encodings (which edges are")
    print("  real), the labels (segmentation is not given), and every accuracy")
    print("  number. A synthetic result tells you a mechanism can exist; only")
    print("  real data tells you whether it helps.")


if __name__ == "__main__":
    main()
