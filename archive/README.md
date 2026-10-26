# Archived original code (2022)

These three files are the repository's first commit, kept for provenance:

- `driver.py` — an IMDB-sentiment sketch that never ran (it constructs test
  graphs with the wrong arguments and imports a GNN that cannot consume
  them).
- `datagraph.py` — a spaCy dependency-parse wrapper with a real bug worth
  reading: `_create_node_features` indexes nodes and arcs from two different
  sets, so the feature matrix rows and the node list can disagree.
- `gnn.py` — a two-layer "GNN" that applies `torch.spmm` to a dense
  adjacency matrix and then executes demo code at import time.

They are not imported anywhere. The live code lives in
`src/morphgnn/`, where the same ideas are implemented correctly and
explained: see `README.md` and `examples/`.
