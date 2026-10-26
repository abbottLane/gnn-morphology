"""Turning words and paradigms into graphs.

Two encodings, deliberately contrasted in the tutorials:

``word_chain``
    Characters are nodes; edges link neighbours (plus an optional
    root-vowel "harmony edge" from the root's last vowel to every
    suffix vowel). This is the graph the linguistics suggests.

``word_star``
    The same nodes, but every character links only to the first.
    Structurally a bag with a hub: order information is destroyed.
    It exists as a control condition — if graph structure is doing
    real work, the chain must beat the star on
    structure-sensitive tasks.

``paradigm_graph``
    One lexeme's word forms as nodes, with edges between forms
    that differ in exactly one feature (e.g. past vs future of the
    same person). This is the graph for paradigm completion.
"""
from __future__ import annotations

import torch

from .kisima import Word

ALPHABET = sorted(set("ptkmnslrwy") | set("aieou"))
C2I = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 = padding


def _char_features(word: str) -> torch.Tensor:
    """One-hot character identity + is-vowel + harmony class of the char."""
    rows = []
    for ch in word:
        row = [0.0] * (len(ALPHABET) + 1 + 2)
        row[C2I.get(ch, 0)] = 1.0
        if ch in "aieou":
            row[len(ALPHABET) + 1] = 1.0
            row[len(ALPHABET) + 1 + (0 if ch in "ie" else 1)] = 1.0
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float)


def _edges(pairs: list[tuple[int, int]]) -> torch.Tensor:
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def word_chain(word: Word, harmony_edges: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Character chain for one word, optionally with harmony edges.

    The harmony edge connects the root's last vowel (the harmony
    controller) to every vowel outside the root — exactly the
    dependency vowel harmony imposes. Removing those edges is the
    key ablation in tutorial 02.
    """
    s = word.surface
    pairs = [(i, i + 1) for i in range(len(s) - 1)] + \
            [(i + 1, i) for i in range(len(s) - 1)]
    if harmony_edges:
        root_start = sum(len(x) for _, _, x in word.morphs if _ != "ROOT" and False)  # placeholder, computed below
        # locate root span from the annotation
        start, span = 0, None
        for slot, _g, surf in word.morphs:
            if slot == "ROOT":
                span = (start, start + len(surf))
                break
            start += len(surf)
        if span is not None:
            vowels = [i for i in range(span[0], span[1]) if s[i] in "aieou"]
            if vowels:
                ctrl = vowels[-1]
                for i, ch in enumerate(s):
                    if ch in "aieou" and not (span[0] <= i < span[1]):
                        pairs += [(ctrl, i), (i, ctrl)]
    return _char_features(s), _edges(pairs)


def word_star(word: Word) -> tuple[torch.Tensor, torch.Tensor]:
    s = word.surface
    pairs = [(0, i) for i in range(1, len(s))] + [(i, 0) for i in range(1, len(s))]
    return _char_features(s), _edges(pairs)


def paradigm_graph(forms: list[Word]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Nodes = forms of one lexeme; edges = differ in one morphosyntactic feature.

    Node features are character n-gram histograms of the surface form
    (bigrams), computed later by the caller; here we return the edge
    structure and each node's feature string.
    """
    def feats(w: Word):
        return tuple(g for _, g, _ in w.morphs)

    pairs = []
    for i in range(len(forms)):
        for j in range(i + 1, len(forms)):
            fi, fj = feats(forms[i]), feats(forms[j])
            if len(fi) == len(fj) and sum(a != b for a, b in zip(fi, fj)) == 1:
                pairs += [(i, j), (j, i)]
    return _edges(pairs), [f.feature_string for f in forms], [f.surface for f in forms]
