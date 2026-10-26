"""02 - Vowel harmony: long-range dependencies and why edges matter.

Kisima suffix vowels agree with the harmony class of the root's LAST
vowel, however far away it is. We mask one suffix vowel and ask each
model to restore it (5-way classification: a i e o u):

  * window MLP   -- sees +-2 characters around the mask;
  * chain GNN    -- messages travel along the character chain;
  * chain + harmony edge -- the root's last vowel is wired directly
    to every suffix vowel (the edge the linguistics suggests).

Accuracy is reported by DISTANCE from the harmony controller. The
window model is competitive at distance 1-2 and has no way to succeed
at distance 6+ except by memorising affix shapes; the plain chain
must relay the harmony information step by step; the harmony edge
makes the dependency one hop. Watch the distance buckets.

Run: python examples/02_vowel_harmony.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from morphgnn.graphs import word_chain
from morphgnn.kisima import generate_lexicon
from morphgnn.layers import GCNLayer

VOWELS = list("aieou")


class HarmonyData(Dataset):
    """One masked suffix vowel per word.

    ``isolated_only`` keeps only masks whose +-2 window contains no other
    suffix vowel: otherwise the affix's own second vowel leaks the harmony
    class to the window model, and every model looks equally good. That
    leak is instructive rather than embarrassing - it is the first thing
    to check when a structure ablation shows no difference.
    """

    def __init__(self, words, harmony_edges, isolated_only=False):
        self.items = []
        for w in words:
            x, ei = word_chain(w, harmony_edges=harmony_edges)
            s = w.surface
            # root span
            start, span = 0, None
            for slot, _g, surf in w.morphs:
                if slot == "ROOT":
                    span = (start, start + len(surf))
                    break
                start += len(surf)
            ctrl = max(i for i in range(span[0], span[1]) if s[i] in VOWELS)
            suffix_vowels = [i for i, ch in enumerate(s)
                             if ch in VOWELS and not (span[0] <= i < span[1])]
            if not suffix_vowels:
                continue
            pos = suffix_vowels[len(suffix_vowels) // 2]
            if isolated_only:
                far = [p for p in suffix_vowels
                       if p != pos and abs(p - pos) <= 2]
                if far or pos - ctrl < 3:
                    continue
            xm = x.clone()
            xm[pos] = 0.0
            self.items.append((xm, ei, pos, VOWELS.index(s[pos]), pos - ctrl))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    xs, eis, poss, ys, ds, bvec = [], [], [], [], [], []
    off = 0
    for g, (x, ei, pos, y, d) in enumerate(batch):
        xs.append(x); eis.append(ei + off)
        poss.append(pos + off); ys.append(y); ds.append(d)
        bvec.append(torch.full((x.shape[0],), g, dtype=torch.long))
        off += x.shape[0]
    return (torch.cat(xs), torch.cat(eis, dim=1), torch.tensor(poss),
            torch.tensor(ys), torch.tensor(ds), torch.cat(bvec))


class MaskGNN(nn.Module):
    def __init__(self, d_in, d=64, layers=3):
        super().__init__()
        self.emb = nn.Linear(d_in, d)
        self.convs = nn.ModuleList(GCNLayer(d, d) for _ in range(layers))
        self.head = nn.Linear(d, len(VOWELS))

    def forward(self, x, ei, poss):
        h = self.emb(x)
        for conv in self.convs:
            h = torch.relu(conv(h, ei))
        return self.head(h[poss])



def run(model_fn, train, test, epochs=20, seed=0):
    torch.manual_seed(seed)
    model = model_fn()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    tl = DataLoader(train, batch_size=32, shuffle=True, collate_fn=collate)
    ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, ei, poss, y, _d, bvec in tl:
            logits = model(x, ei, poss)
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct, total = {}, {}
    with torch.no_grad():
        for x, ei, poss, y, d, bvec in DataLoader(test, batch_size=64, collate_fn=collate):
            pred = model(x, ei, poss).argmax(-1)
            for pi, yi, di in zip(pred.tolist(), y.tolist(), d.tolist()):
                bucket = "1-2" if di <= 2 else ("3-5" if di <= 5 else "6+")
                total[bucket] = total.get(bucket, 0) + 1
                correct[bucket] = correct.get(bucket, 0) + int(pi == yi)
    acc = {b: correct.get(b, 0) / max(total.get(b, 0), 1) for b in ["1-2", "3-5", "6+"]}
    acc["all"] = sum(correct.values()) / max(sum(total.values()), 1)
    return acc


def run_mlp(train, test, d_in, epochs=20, seed=0):
    """Window baseline trained per-word (not batched graphs)."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(5 * d_in, 128), nn.ReLU(), nn.Linear(128, len(VOWELS)))

    def window(x, pos):
        pad = torch.zeros(2, x.shape[1])
        xp = torch.cat([pad, x, pad])
        return xp[pos:pos + 5].reshape(-1)

    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    ce = nn.CrossEntropyLoss()
    items = train.items
    for _ in range(epochs):
        net.train()
        order = torch.randperm(len(items))
        for i in order.tolist():
            x, _ei, pos, y, _d = items[i]
            loss = ce(net(window(x, pos)).unsqueeze(0), torch.tensor([y]))
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    correct, total = {}, {}
    with torch.no_grad():
        for x, _ei, pos, y, d in test.items:
            pi = int(net(window(x, pos)).argmax())
            bucket = "1-2" if d <= 2 else ("3-5" if d <= 5 else "6+")
            total[bucket] = total.get(bucket, 0) + 1
            correct[bucket] = correct.get(bucket, 0) + int(pi == y)
    acc = {b: correct.get(b, 0) / max(total.get(b, 0), 1) for b in ["1-2", "3-5", "6+"]}
    acc["all"] = sum(correct.values()) / max(sum(total.values()), 1)
    return acc


def main():
    lexicon = generate_lexicon(seed=0)
    roots = sorted(lexicon)
    import random
    random.Random(42).shuffle(roots)
    train_words = [w for r in roots[:64] for w in lexicon[r]]
    test_words = [w for r in roots[64:] for w in lexicon[r]]
    d_in = None
    results = {}
    for mode, isolated, label in [("any mask", False, "ALL MASKS (local context leaks,"),
                                  ("isolated mask", True, "ISOLATED MASKS (no other suffix")]:
        results = {}
        for name, edges in [("chain GNN", False), ("chain + harmony edge", True)]:
            train = HarmonyData(train_words, edges, isolated)
            test = HarmonyData(test_words, edges, isolated)
            d_in = train.items[0][0].shape[1]
            results[name] = run(lambda: MaskGNN(d_in), train, test)
        train_m = HarmonyData(train_words, False, isolated)
        test_m = HarmonyData(test_words, False, isolated)
        results["window MLP"] = run_mlp(train_m, test_m, d_in)
        print(f"\n{label} vowel within +-2)")
        print(f"  {'model':24s} {'all':>6} {'dist 1-2':>9} {'dist 3-5':>9} {'dist 6+':>8}  "
              f"(n={len(train_m)})")
        for name, acc in results.items():
            print(f"  {name:24s} {acc['all']:.3f}  {acc['1-2']:.3f}     "
                  f"{acc['3-5']:.3f}     {acc['6+']:.3f}")
    print("\nThe window model has no access to the root vowel once it is more "
          "than two characters away. The plain chain must relay harmony "
          "step by step and degrades with distance; the harmony edge makes "
          "the linguistically real dependency a single hop.")


if __name__ == "__main__":
    main()
