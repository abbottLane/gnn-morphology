"""04 - Graph classification: where message passing becomes a liability.

So far we used graphs to make tasks easier. This file shows the opposite
case, deliberately. Task: classify a Kisima word as a verb or a noun.

  * A graph-level GNN (message passing + mean pooling) reads the word
    graph. It learns a degree-invariant representation, which is exactly
    what a GNN is designed to do.
  * A plain bag-of-characters MLP reads the same characters, no structure.

For this task the MLP matches or beats the GNN: noun and verb paradigms
differ in their *inventory* of affixes, which is a bag property, not a
structural one. Graph structure helps when the answer depends on *which
parts are adjacent or aligned* (examples 01-03), and costs you when it
does not. Knowing which regime you are in is a modelling decision, not
a religious one.

Run: python examples/04_graph_classification.py
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
from morphgnn.layers import GCNLayer, global_mean_pool


class WordData(Dataset):
    def __init__(self, words):
        self.items = []
        for w in words:
            x, ei = word_chain(w, harmony_edges=True)
            y = 0 if w.pos == "V" else 1
            self.items.append((x, ei, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    xs, eis, ys, bvec = [], [], [], []
    off = 0
    for g, (x, ei, y) in enumerate(batch):
        xs.append(x); eis.append(ei + off); ys.append(y)
        bvec.append(torch.full((x.shape[0],), g, dtype=torch.long))
        off += x.shape[0]
    return torch.cat(xs), torch.cat(eis, dim=1), torch.tensor(ys), torch.cat(bvec)


class GraphClassifier(nn.Module):
    def __init__(self, d_in, d=64, layers=3):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.convs = nn.ModuleList(GCNLayer(d, d) for _ in range(layers))
        self.head = nn.Linear(d, 2)

    def forward(self, x, ei, bvec, n_graphs):
        h = self.emb(x)
        for conv in self.convs:
            h = torch.relu(conv(h, ei))
        return self.head(global_mean_pool(h, bvec, n_graphs))


class BagMLP(nn.Module):
    def __init__(self, d_in, d=64):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 2))

    def forward(self, x, ei, bvec, n_graphs):
        pooled = global_mean_pool(self.emb(x), bvec, n_graphs)  # mean, no messages
        return self.head(pooled)


def run(model_cls, train, test, d_in, epochs=20, seed=0):
    torch.manual_seed(seed)
    model = model_cls(d_in)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    tl = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    for _ in range(epochs):
        model.train()
        for x, ei, y, bvec in tl:
            logits = model(x, ei, bvec, int(y.shape[0]))
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, ei, y, bvec in DataLoader(test, batch_size=64, collate_fn=collate):
            pred = model(x, ei, bvec, int(y.shape[0])).argmax(-1)
            correct += int((pred == y).sum()); total += int(y.shape[0])
    return correct / total


def main():
    lexicon = generate_lexicon(seed=3)
    roots = sorted(lexicon)
    random.Random(11).shuffle(roots)
    train = [w for r in roots[:60] for w in lexicon[r]]
    test = [w for r in roots[60:] for w in lexicon[r]]
    train_ds, test_ds = WordData(train), WordData(test)
    d_in = train_ds.items[0][0].shape[1]
    print(f"train {len(train)} / test {len(test)} words "
          f"({sum(1 for w in test if w.pos == 'V')} verbs, "
          f"{sum(1 for w in test if w.pos == 'N')} nouns)")
    for name, cls in [("GNN + mean pool", GraphClassifier),
                      ("bag MLP (no edges)", BagMLP)]:
        print(f"{name:22s} POS accuracy on unseen lexemes: {run(cls, train_ds, test_ds, d_in):.3f}")
    print("\nBoth are near ceiling: the task is about which affixes are "
          "present, not how they are arranged. Use this file as the control "
          "when you read examples 01-03: structure helps tasks that are "
          "about arrangement, and this is not one of them.")


if __name__ == "__main__":
    main()
