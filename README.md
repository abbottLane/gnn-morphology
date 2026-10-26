# Graph Neural Networks for Morphological Analysis

**Mission.** Teach graph neural networks through a problem where graphs are
not decoration: the morphology of complex languages. Every example in this
repository is a small, readable experiment that answers one question about
when graph structure helps an NLP model, and when it does not.

Most GNN tutorials classify citation networks. This one builds word graphs —
characters as nodes, dependency and agreement relations as edges — and asks
concrete linguistic questions: where does one morpheme end? Which vowel does
a suffix take? What is the missing form in a paradigm?

## Why morphology is the right teaching problem

Agglutinative and polysynthetic languages (Turkish, Finnish, Plains Cree,
Kunwinjku) express in a word what English does in a clause. Their words have
structure — ordered morpheme slots, agreement across positions, harmony
relations that ignore distance — and that structure is exactly what graph
representations are for:

| Linguistic fact | Graph formulation | Example |
|---|---|---|
| Morphemes concatenate in slots | node classification on a character chain | `01` |
| Suffix vowels agree with a distant root vowel | long-range edges (harmony) | `02` |
| Forms of a lexeme constrain each other | paradigm as a graph; completion as link prediction | `03` |
| Some tasks are about inventory, not arrangement | when to *not* use a GNN | `04` |
| Real languages have allomorphy and suppletion | what this sandbox cannot show you | `05` |

## The language in this repo

All experiments run on **Kisima**, a synthetic language defined in
[`src/morphgnn/kisima.py`](src/morphgnn/kisima.py) in about a page of rules:
agglutinating verb and noun templates, vowel harmony, and full paradigms.
Because Kisima is generated, every label is exactly correct — no annotation
noise to hide behind, and every result is reproducible from a seed.

Kisima is a wind tunnel. It has no fusional morphology, no suppletion, and
no phonology beyond concatenation. Tutorial `05` is about the distance
between it and languages like Irish, Plains Cree, or Kunwinjku — and about
what to read next.

## Start here

```bash
pip install -r requirements.txt
python examples/01_morpheme_boundaries.py
python examples/02_vowel_harmony.py
python examples/03_paradigm_completion.py
python examples/04_graph_classification.py
```

Each script prints its own results and a paragraph explaining what to
conclude. Runs in under two minutes on CPU; no GPU, no graph libraries —
message passing is implemented from scratch in
[`src/morphgnn/layers.py`](src/morphgnn/layers.py) so you can read it.

## What you will actually learn

1. **Message passing from first principles.** `layers.py` implements
   `h_i <- phi(h_i, mean_j psi(h_i, h_j))` with plain `index_add_`. A GCN is
   the special case where messages ignore the receiver; you can see the
   reduction rather than take it on faith.
2. **Graphs as inductive bias.** Example `01` shows a chain GNN beating a
   window model on unseen roots with identical features and capacity:
   structure is worth something specific, not merely "better".
3. **Edges encode linguistics.** Example `02` isolates the harmony edge —
   the accurate dependency — and shows the difference it makes at distance.
4. **When graphs hurt.** Example `04` gives you the control condition.
   A GNN is degree-invariant; if your task is a bag-of-parts question,
   pooling without messages does just as well and trains faster.
5. **Evaluation discipline.** Every example splits by *root*, not by word,
   so memorisation cannot masquerade as generalisation.

## Suggested exercises

Each example ends with an obvious next experiment; a few worth doing:

- `01`: replace character edges with morpheme-type edges (all suffixes of a
  slot linked). Does the GNN need the chain at all?
- `02`: add a second harmony domain (e.g. rounding). Which edges must exist
  for the model to have any chance?
- `03`: predict the *surface* form rather than the feature vector and score
  with exact match; where does the rule-based realisation stop helping?
- `04`: make the task structural (e.g. is the object suffix adjacent to the
  root?) and watch the GNN take the lead back.

## Repository layout

```
src/morphgnn/kisima.py    the synthetic language: rules and paradigms
src/morphgnn/graphs.py    word -> graph encodings (chain, star, paradigm)
src/morphgnn/layers.py    GCN and general message passing, from scratch
examples/                 five self-contained tutorials (start here)
archive/                  original 2022 sandbox code, kept for provenance
```

## Background reading

- Kipf & Welling (2017), *Semi-Supervised Classification with Graph
  Convolutional Networks* — the GCN this repo reimplements.
- Gilmer et al. (2017), *Neural Message Passing for Quantum Chemistry* — the
  general framework behind `MessagePassingLayer`.
- Cotterell, Kirov, Hulden & Eisner (2016–2019) on paradigm completion and
  inflection generation — the task in example `03`.
- Lane, Harrigan & Arppe (2022), *Interactive Word Completion for Plains
  Cree*, and Lane & Bird (2020), *Interactive Word Completion for
  Morphologically Complex Languages* — the applied motivation: help
  language learners and speakers build complex words.
- Vylomova et al., SIGMORPHON shared tasks on morphological reinflection —
  real data, once you have finished the wind tunnel.

## Scope and honesty

This is an educational sandbox, not a benchmark. The synthetic language is
simple by construction; results here are about *mechanisms*, and no number
in this repository should be quoted as a claim about any real language.
The next honest step is a real corpus — and `05` says which ones.

Licence: MIT (see `LICENSE`). Contributions and corrections welcome.
