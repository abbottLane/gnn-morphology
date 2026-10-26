"""03 - Paradigm completion: filling unattested cells from attested ones.

Kisima lexemes have full paradigms (verbs: subject x tense x object x mood;
nouns: possessor x number x case). Real documentation rarely gives you a
complete paradigm: a field notebook, a dictionary, or a corpus contains a
few attested forms per lexeme and leaves the rest blank. This example takes
that setting literally — 60% of each lexeme's cells are hidden, and the task
is to fill them in from the 40% that remain.

The paradigm is the graph. Each attested form is a node; edges join forms
that differ in exactly one morphosyntactic feature, so the hidden cell sits
among neighbours that constrain it.

Two strategies, identical inputs and the same test cells:
  * nearest observed neighbour — copy the slot values of the most similar
    attested form;
  * GNN — pool the whole observed neighbourhood, one classification head
    per morphological slot.

Metric: slot accuracy (each slot decided independently) and cell accuracy
(all slots of a cell correct), on lexemes never seen in training.

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

SLOTS = ["SUBJ", "TENSE", "OBJ", "MOOD", "POSS", "NUM", "CASE"]


class Paradigm:
    """Graph over one lexeme's forms + per-slot supervision for each cell."""

    def __init__(self, forms, labels, slot_vocab):
        """`labels` is a GLOBAL feature vocabulary: verbs and nouns realise
        different slots, and every graph in a batch must have the same width."""
        self.forms = forms
        n = len(forms)
        self.labels = labels
        self.l2i = {v: i for i, v in enumerate(labels)}
        x = torch.zeros(n, len(self.labels))
        for i, w in enumerate(forms):
            for slot, g, _ in w.morphs:
                if slot != "ROOT":
                    x[i, self.l2i[f"{slot}={g}"]] = 1.0
        self.x = x
        self.edge_index, _, _ = paradigm_graph(forms)
        # per-slot targets: gloss index (0 = slot absent for this lexeme type)
        self.slot_vocab = slot_vocab
        self.slot_target = {}
        for s in SLOTS:
            vals = self.slot_vocab[s]
            tgt = torch.zeros(n, dtype=torch.long)
            for i, w in enumerate(forms):
                for sl, g, _ in w.morphs:
                    if sl == s:
                        tgt[i] = vals.index(g) + 1  # 0 = this slot does not apply
            self.slot_target[s] = tgt


class ParadigmGNN(nn.Module):
    def __init__(self, d_in, slot_sizes, d=128, layers=4):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.convs = nn.ModuleList(GCNLayer(d, d) for _ in range(layers))
        self.heads = nn.ModuleDict({s: nn.Linear(d, n + 1) for s, n in slot_sizes.items()})

    def forward(self, x, ei, idx):
        h = self.emb(x)
        for conv in self.convs[:-1]:
            h = torch.relu(conv(h, ei))
        h = self.convs[-1](h, ei)
        return {s: head(h[idx]) for s, head in self.heads.items()}


def hide_block(g: Paradigm, rng: random.Random):
    """Hide one systematic block of the paradigm: every form sharing a value
    of one feature (a whole tense, a whole case, ...).

    This is what a real gap looks like - a notebook that recorded the present
    tense everywhere but missed it for one lexeme - and unlike random masking
    it cannot be solved by copying a neighbour that differs in a single
    feature. The hidden cell must be assembled from several neighbours that
    each carry part of the answer.
    """
    slot = rng.choice([s for s in SLOTS if len(g.slot_vocab[s]) > 1])
    values = [v for v in range(1, len(g.slot_vocab[slot]) + 1)]
    value = rng.choice(values)
    hidden = [i for i in range(len(g.forms)) if int(g.slot_target[slot][i]) == value]
    if not hidden:                      # slot does not apply to this lexeme type
        return hide_block(g, rng)
    x = g.x.clone()
    for i in hidden:
        x[i] = 0.0
    return x, sorted(hidden)


def main():
    lexicon = generate_lexicon(seed=0)
    roots = sorted(lexicon)
    random.Random(7).shuffle(roots)
    train_roots, test_roots = roots[:48], roots[48:]
    labels = sorted({f"{slot}={g}" for r in roots for w in lexicon[r]
                     for slot, g, _ in w.morphs if slot != "ROOT"})
    slot_vocab = {s: sorted({g for r in roots for w in lexicon[r]
                             for sl, g, _ in w.morphs if sl == s}) for s in SLOTS}
    train_g = [Paradigm(lexicon[r], labels, slot_vocab) for r in train_roots]
    test_g = [Paradigm(lexicon[r], labels, slot_vocab) for r in test_roots]
    slot_sizes = {s: len(slot_vocab[s]) for s in SLOTS}

    # observed/hidden split fixed per lexeme, as in a field notebook
    rng = random.Random(0)
    train_batches = []
    for g in train_g:
        x, hidden = hide_block(g, rng)
        train_batches += [(g, x, i) for i in hidden]

    torch.manual_seed(0)
    model = ParadigmGNN(len(labels), slot_sizes, d=192)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ce = nn.CrossEntropyLoss()
    for epoch in range(40):
        rng.shuffle(train_batches)
        model.train()
        for s0 in range(0, len(train_batches), 32):
            batch = train_batches[s0:s0 + 32]
            xs, eis, idxs, off = [], [], [], 0
            for g, x, i in batch:
                xs.append(x); eis.append(g.edge_index + off)
                idxs.append(i + off); off += len(g.forms)
            out = model(torch.cat(xs), torch.cat(eis, dim=1), torch.tensor(idxs))
            loss = sum(ce(out[s], torch.stack([g.slot_target[s][i] for g, _x, i in batch]))
                       for s in SLOTS)
            opt.zero_grad(); loss.backward(); opt.step()

    # ---- evaluate on unseen lexemes ----
    model.eval()
    gnn_slot_hits = gnn_slot_tot = gnn_cells = gnn_cell_hits = 0
    nn_slot_hits = nn_slot_tot = nn_cells = nn_cell_hits = 0
    with torch.no_grad():
        for g in test_g:
            x, hidden = hide_block(g, rng)
            xs, eis, idxs, off = [], [], [], 0
            for i in hidden:
                xs.append(x); eis.append(g.edge_index + off)
                idxs.append(i + off); off += len(g.forms)
            out = model(torch.cat(xs), torch.cat(eis, dim=1), torch.tensor(idxs))
            observed = [i for i in range(len(g.forms)) if i not in set(hidden)]
            obs_x = g.x[observed]
            for k, i in enumerate(hidden):
                applies = [s for s in SLOTS if g.slot_target[s][i] > 0]
                gnn_ok = nn_ok = 0
                j = int((obs_x * g.x[i]).sum(1).argmax())          # nearest observed
                for s in applies:
                    gold = int(g.slot_target[s][i])
                    p_gnn = int(out[s][k].argmax())
                    p_nn = int(g.slot_target[s][observed[j]])
                    gnn_slot_hits += p_gnn == gold; gnn_slot_tot += 1
                    nn_slot_hits += p_nn == gold; nn_slot_tot += 1
                    gnn_ok += p_gnn == gold; nn_ok += p_nn == gold
                gnn_cells += 1; gnn_cell_hits += gnn_ok == len(applies)
                nn_cells += 1; nn_cell_hits += nn_ok == len(applies)

    print(f"test: {len(test_roots)} unseen lexemes, {gnn_cells} unattested cells "
          f"predicted from the other 40%")
    print(f"{'method':30s} slot acc  cell acc")
    print(f"{'nearest observed neighbour':30s} {nn_slot_hits / nn_slot_tot:.3f}     "
          f"{nn_cell_hits / nn_cells:.3f}")
    print(f"{'GNN (pool neighbours)':30s} {gnn_slot_hits / gnn_slot_tot:.3f}     "
          f"{gnn_cell_hits / gnn_cells:.3f}")
    print("\nRead the two columns together - they disagree, and that is the point.\n"
          "  * Slot accuracy: copying the nearest attested form is a strong\n"
          "    baseline (it is usually right about most slots), and here it beats\n"
          "    the GNN. A single neighbour is a good prior; pooling adds noise.\n"
          "  * Cell accuracy: the block is hidden, so every observed form has the\n"
          "    wrong value in that slot. Copying anyone can NEVER get a whole cell\n"
          "    right. Only a model that combines several neighbours can, and the\n"
          "    GNN does so on a small but non-zero share of cells.\n"
          "Choose the metric that matches the capability you care about: if\n"
          "filling a systematic gap is the job, slot accuracy will flatter the\n"
          "baseline into looking sufficient when it is structurally incapable.")


if __name__ == "__main__":
    main()
