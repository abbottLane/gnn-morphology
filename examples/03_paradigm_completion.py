"""03 - Paradigm completion: a link-prediction view of morphology.

A paradigm is a grid: one lexeme, one form per morphosyntactic cell
(person x tense x mood for verbs). Given a lexeme with one cell hidden,
predict the hidden form. This is the classic "fill in the paradigm"
problem in computational morphology (cf. Cotterell et al.'s work on
paradigm completion and inflection generation).

Here the paradigm IS the graph: each known form is a node, and edges
join forms differing in exactly one feature. The hidden cell sits in
that graph; its neighbours are the forms that constrain it most. We
compare two strategies:

  * nearest-neighbour transfer: copy the surface of the single closest
    neighbour, then repair harmony;
  * GNN: pool the features of all neighbours to predict the hidden
    form's feature vector, then realise the surface with the rule
    system (evaluation is on FEATURES, so results are readable).

Run: python examples/03_paradigm_completion.py
"""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from morphgnn.graphs import paradigm_graph
from morphgnn.kisima import generate_lexicon
from morphgnn.layers import GCNLayer

MOODS = ["ind", "imp", "neg.imp"]


class FeatureGraph:
    """Node = one form; features = multi-hot over (slot, gloss).

    The feature vocabulary is global (shared across paradigms), so the
    input width is the same for every lexeme.
    """

    def __init__(self, forms, vocab):
        self.forms = forms
        self.vocab = vocab
        self.f2i = {v: i for i, v in enumerate(vocab)}
        x = torch.zeros(len(forms), len(vocab))
        for i, w in enumerate(forms):
            for slot, g, _ in w.morphs:
                x[i, self.f2i[f"{slot}={g}"]] = 1.0
        self.x = x
        self.edge_index, _, _ = paradigm_graph(forms)

    def mask(self, idx):
        xm = self.x.clone()
        xm[idx] = 0.0
        return xm


class ParadigmGNN(nn.Module):
    def __init__(self, d_in, d=64, layers=3):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.convs = nn.ModuleList(GCNLayer(d, d) for _ in range(layers))
        self.head = nn.Linear(d, d_in)

    def forward(self, x, ei, idx):
        h = self.emb(x)
        for conv in self.convs:
            h = torch.relu(conv(h, ei))
        return self.head(h[idx])


def main():
    lexicon = generate_lexicon(seed=0)
    roots = sorted(lexicon)
    random.Random(7).shuffle(roots)
    train_roots, test_roots = roots[:48], roots[48:]

    vocab = sorted({f"{slot}={g}" for r in roots for w in lexicon[r] for slot, g, _ in w.morphs})

    def samples(rs):
        out = []
        for r in rs:
            forms = lexicon[r]
            g = FeatureGraph(forms, vocab)
            for i in range(len(forms)):
                out.append((g, i))
        return out

    train, test = samples(train_roots), samples(test_roots)
    d_in = len(vocab)
    torch.manual_seed(0)
    model = ParadigmGNN(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(15):
        model.train()
        random.shuffle(train)
        for g, i in train:
            logits = model(g.mask(i), g.edge_index, torch.tensor([i]))
            loss = bce(logits[0], g.x[i])
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    def evaluate(pred_fn, name):
        exact = 0
        for g, i in test:
            pred = pred_fn(g, i)
            exact += int(torch.equal(pred, g.x[i].bool()))
        print(f"{name:28s} exact feature-vector match: {exact / len(test):.3f}")

    with torch.no_grad():
        def gnn_pred(g, i):
            return model(g.mask(i), g.edge_index, torch.tensor([i]))[0] > 0

        def nn_pred(g, i):
            # single nearest neighbour in the graph: most similar known form
            sims = (g.x * g.x[i]).sum(1)
            sims[i] = -1
            j = int(sims.argmax())
            return g.x[j].bool()

        evaluate(gnn_pred, "GNN (pool neighbours)")
        evaluate(nn_pred, "nearest neighbour (1 form)")
    print(f"\ntest paradigms: {len(test_roots)} unseen lexemes, {len(test)} held-out cells")
    print("\nThe nearest-neighbour baseline copies one form and therefore cannot "
          "compose information from several; the GNN pools the whole "
          "neighbourhood, which is what a paradigm's structure offers.")


if __name__ == "__main__":
    main()
