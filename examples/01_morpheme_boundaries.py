"""01 - Morpheme boundary detection: why structure helps.

Task: given a Kisima word you have never seen (its root never appeared in
training), mark where one morpheme ends and the next begins.

We compare, on identical data and identical capacity:
  * GNN over the character chain graph (structure-aware), and
  * an MLP over per-character windows (the same features, no graph).

The split is by ROOT, not by word: a model that memorised training words
cannot transfer their boundaries to new roots. Whatever the GNN knows at
test time, it learned about *patterns*, e.g. that a boundary often follows
certain affix shapes regardless of the root between them.

Run: python examples/01_morpheme_boundaries.py
"""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from morphgnn.graphs import word_chain
from morphgnn.kisima import generate_lexicon
from morphgnn.layers import GCNLayer


class BoundaryData(Dataset):
    def __init__(self, words):
        self.items = []
        for w in words:
            x, ei = word_chain(w, harmony_edges=False)
            y = torch.zeros(len(w.surface) - 1, dtype=torch.long)
            for b in w.boundaries:
                y[b - 1] = 1
            self.items.append((x, ei, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    xs, eis, ys, bvec = [], [], [], []
    off = 0
    for g, (x, ei, y) in enumerate(batch):
        xs.append(x)
        ys.append(y)
        eis.append(ei + off)
        bvec.append(torch.full((x.shape[0],), g, dtype=torch.long))
        off += x.shape[0]
    return torch.cat(xs), torch.cat(eis, dim=1), torch.cat(ys), torch.cat(bvec)


class GapGNN(nn.Module):
    """Node states via GCN layers; classify each within-word gap."""

    def __init__(self, d_in, d=64, layers=3):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.convs = nn.ModuleList(GCNLayer(d, d) for _ in range(layers))
        self.head = nn.Linear(2 * d, 2)

    def forward(self, x, ei, batch):
        h = self.emb(x)
        for conv in self.convs[:-1]:
            h = torch.relu(conv(h, ei))
        h = self.convs[-1](h, ei)
        same = batch[:-1] == batch[1:]
        idx = same.nonzero(as_tuple=True)[0]
        pair = torch.cat([h[idx], h[idx + 1]], dim=1)
        return self.head(pair)


class WindowMLP(nn.Module):
    """Same inputs, no graph: classify each gap from a +-2 char window."""

    def __init__(self, d_in, d=64):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.net = nn.Sequential(nn.Linear(5 * d, 2 * d), nn.ReLU(),
                                 nn.Linear(2 * d, 2))

    def forward(self, x, batch):
        h = self.emb(x)
        outs = []
        for g in batch.unique():
            xg = h[batch == g]
            pad = torch.zeros(2, xg.shape[1])
            xp = torch.cat([pad, xg, pad])
            wins = torch.stack([xp[i:i + 5].reshape(-1) for i in range(len(xg) - 1)])
            outs.append(self.net(wins))
        return torch.cat(outs)


def f1(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def train_evaluate(model, train, test, kind, epochs=25, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    tl = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, ei, y, bvec in tl:
            logits = model(x, ei, bvec) if kind == "gnn" else model(x, bvec)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    preds, gold = [], []
    with torch.no_grad():
        for x, ei, y, bvec in DataLoader(test, batch_size=64, collate_fn=collate):
            logits = model(x, ei, bvec) if kind == "gnn" else model(x, bvec)
            preds.append(logits.argmax(-1))
            gold.append(y)
    return f1(torch.cat(preds), torch.cat(gold))


def main():
    lexicon = generate_lexicon(seed=0)
    roots = sorted(lexicon)
    random.Random(42).shuffle(roots)
    train = [w for r in roots[:64] for w in lexicon[r]]
    test = [w for r in roots[64:] for w in lexicon[r]]
    train_ds, test_ds = BoundaryData(train), BoundaryData(test)
    d_in = train_ds.items[0][0].shape[1]
    print(f"train {len(train)} words from 64 roots; test {len(test)} words "
          f"from 16 held-out roots")
    for name, model, kind in [("GNN (chain graph)", GapGNN(d_in), "gnn"),
                              ("MLP (char window)", WindowMLP(d_in), "mlp")]:
        score = train_evaluate(model, train_ds, test_ds, kind)
        print(f"{name:22s} boundary F1 on unseen roots: {score:.3f}")
    print("\nReading the result: both models see identical characters. The gap "
          "between them is what neighbourhood structure buys you - message "
          "passing lets a suffix shape be recognised wherever it occurs, "
          "while the window model must relearn it at every offset.")


if __name__ == "__main__":
    main()
