"""Tests: the language generator, graph encodings, and layers must be exact.

Run: pytest -q
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from morphgnn.graphs import paradigm_graph, word_chain, word_star
from morphgnn.kisima import (BACK, FRONT, build_noun, build_verb,
                             generate_lexicon, harmony)
from morphgnn.layers import GCNLayer, add_self_loops, scatter_mean


# ---------- language rules ----------

def test_harmony_reads_last_vowel():
    assert harmony("makta") == 1        # a -> back
    assert harmony("tilek") == 0        # e -> front
    assert harmony("tako") == 1         # o -> back (last vowel wins)
    assert harmony("toka") == 1         # a -> back


def test_affix_allomorphs_follow_root_harmony():
    front = build_verb("tilek", "1sg", "past", "none", "ind", neg=False)
    back = build_verb("tako", "1sg", "past", "none", "ind", neg=False)
    assert front.surface.startswith("ni") and "li" in front.surface     # front allomorphs
    assert back.surface.startswith("nu") and "lu" in back.surface       # back allomorphs
    assert all(ch not in BACK for ch in front.surface if ch in "aou")


def test_zero_exponents_are_annotated_not_dropped():
    w = build_verb("makta", "3sg", "pres", "none", "ind", neg=False)
    assert w.surface == "makta"
    glosses = [g for _, g, _ in w.morphs]
    assert "3sg" in glosses and "pres" in glosses  # invisible structure kept


def test_boundaries_match_annotation():
    w = build_noun("talen", "1sg.poss", "pl", "gen")
    parts = [s for _, _, s in w.morphs if s]
    # boundaries are cumulative character offsets of each non-final morpheme
    expected, acc = [], 0
    for part in parts[:-1]:
        acc += len(part)
        expected.append(acc)
    assert w.boundaries == expected
    assert sum(len(p) for p in parts) == len(w.surface)


def test_lexicon_is_deterministic_and_complete():
    a, b = generate_lexicon(seed=5, n_verbs=4, n_nouns=4), generate_lexicon(seed=5, n_verbs=4, n_nouns=4)
    assert sorted(a) == sorted(b)
    assert all([w.surface for w in a[r]] == [w.surface for w in b[r]] for r in a)
    verb_forms = {len(v) for r, v in a.items() if v[0].pos == "V"}
    assert len(verb_forms) == 1  # every verb has the same paradigm size


# ---------- graph encodings ----------

def test_chain_edges_are_bidirectional_and_contiguous():
    w = build_verb("makta", "1sg", "past", "none", "ind", neg=False)
    x, ei = word_chain(w, harmony_edges=False)
    assert x.shape[0] == len(w.surface)
    assert ei.shape[0] == 2
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    for i in range(len(w.surface) - 1):
        assert (i, i + 1) in pairs and (i + 1, i) in pairs


def test_harmony_edges_link_root_vowel_to_suffix_vowels():
    w = build_verb("makta", "1sg", "past", "3sg.obj", "ind", neg=False)
    _, base = word_chain(w, harmony_edges=False)
    _, harm = word_chain(w, harmony_edges=True)
    assert harm.shape[1] > base.shape[1]          # harmony adds edges
    root_end = w.surface.index(w.lemma) + len(w.lemma)
    pairs = set(zip(harm[0].tolist(), harm[1].tolist()))
    ctrl = max(i for i in range(root_end) if w.surface[i] in "aieou")
    suffix_vowels = [i for i, c in enumerate(w.surface) if c in "aieou" and i >= root_end]
    assert suffix_vowels, "test form should exercise a suffix vowel"
    assert all((ctrl, v) in pairs for v in suffix_vowels)


def test_star_graph_has_same_nodes_fewer_structures():
    w = build_noun("talen", "1sg.poss", "pl", "gen")
    xc, _ = word_chain(w, harmony_edges=False)
    xs, _ = word_star(w)
    assert xc.shape == xs.shape


def test_paradigm_edges_join_single_feature_differences():
    forms = [build_verb("makta", s, t, "none", "ind", neg=False)
             for s in ("1sg", "2sg") for t in ("past", "pres")]
    ei, feats, surfaces = paradigm_graph(forms)
    assert len(feats) == len(forms) == len(surfaces)
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs   # differ only in tense
    assert (0, 2) in pairs                       # differ only in subject


# ---------- layers ----------

def test_scatter_mean_averages_groups():
    src = torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 10.0]])
    idx = torch.tensor([0, 0, 1])
    out = scatter_mean(src, idx, 2)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0], [10.0, 10.0]]))


def test_self_loops_add_one_per_node():
    ei = torch.tensor([[0, 1], [1, 0]])
    loops = add_self_loops(ei, 2)
    assert loops.shape[1] == 4
    pairs = set(zip(loops[0].tolist(), loops[1].tolist()))
    assert (0, 0) in pairs and (1, 1) in pairs


def test_gcn_is_permutation_equivariant():
    """Relabelling nodes relabels outputs: the layer has no node-order bias."""
    torch.manual_seed(0)
    layer = GCNLayer(4, 3)
    x = torch.randn(5, 4)
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    perm = torch.tensor([4, 0, 3, 1, 2])      # perm[i] = new label of old node i
    inv = torch.argsort(perm)                 # so features move as x[inv]
    out = layer(x, ei)
    out_relabelled = layer(x[inv], perm[ei])
    assert torch.allclose(out_relabelled, out[inv], atol=1e-6)


def test_gcn_isolated_node_still_updates():
    """A self-loop means an unconnected node is not zeroed out."""
    layer = GCNLayer(3, 2)
    x = torch.randn(1, 3)
    out = layer(x, torch.zeros(2, 0, dtype=torch.long))
    assert out.shape == (1, 2) and torch.isfinite(out).all()
