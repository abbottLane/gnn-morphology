"""Graph neural network layers in pure PyTorch — no graph libraries.

Everything here is written to be read. A graph is represented the way
message passing actually computes on it: a node-feature matrix ``x`` and
an ``edge_index`` pair of index tensors (sources, destinations). Batched
graphs are one big disconnected graph plus a ``batch`` vector saying which
graph each node belongs to. That is all PyTorch Geometric does under the
hood; writing it once by hand is worth more than importing it ten times.

Notation: for an edge j -> i ("j sends a message to i"),
``edge_index = [sources, destinations]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean of rows of `src` grouped by `index` — the aggregation primitive."""
    out = torch.zeros(dim_size, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, torch.ones(index.shape[0], dtype=src.dtype, device=src.device))
    return out / count.clamp(min=1).unsqueeze(1)


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    loops = torch.arange(num_nodes, device=edge_index.device)
    loop_edges = torch.stack([loops, loops])
    return torch.cat([edge_index, loop_edges], dim=1)


class GCNLayer(nn.Module):
    """One GCN step (Kipf & Welling, 2017) with symmetric normalisation.

    H' = act( D^-1/2 (A + I) D^-1/2 H W ).  We implement it as
    message passing so the connection to the general framework is visible.
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        ei = add_self_loops(edge_index, x.shape[0])
        src, dst = ei[0], ei[1]
        deg = torch.zeros(x.shape[0], device=x.device)
        deg.index_add_(0, dst, torch.ones(dst.shape[0], device=x.device))
        norm = deg[src].rsqrt() * deg[dst].rsqrt()          # D^-1/2 on both ends
        messages = self.lin(x)[src] * norm.unsqueeze(1)      # scale each message
        out = torch.zeros_like(messages[:1]).expand(x.shape[0], -1).clone()
        out.index_add_(0, dst, messages)
        return out


class MessagePassingLayer(nn.Module):
    """The general update: h_i <- phi(h_i, mean_j psi(h_i, h_j)).

    With psi ignoring h_i and phi a linear map this reduces to a GCN;
    letting psi see both endpoints makes messages content-dependent.
    """

    def __init__(self, d: int):
        super().__init__()
        self.psi = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        pair = torch.cat([x[dst], x[src]], dim=1)           # (receiver, sender)
        agg = scatter_mean(self.psi(pair), dst, x.shape[0])
        return self.phi(torch.cat([x, agg], dim=1))


def global_mean_pool(x: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
    """Graph-level readout: mean of each graph's node states."""
    return scatter_mean(x, batch, num_graphs)
