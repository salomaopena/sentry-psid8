"""Correlation models for SENTRY-C.

Two scorers share ONE interface (`CorrelationModel`), so the protocol of the
thesis (H3) can compare them on identical inputs:

  * `TemporalRuleScorer` — a fixed-window, fixed-precedence-rule baseline with no
    learned parameters. It exists to answer a predictive-merit question, and
    only that: does the GNN detect and rank convergent incidents better than a
    competent engineer would with hand-written rules? This module is a
    comparator for H3, not a deliverable on interpretability or explainability
    — that dimension is explicitly out of scope for this work.
  * `HeteroGNNScorer` — a small typed-attention GNN (HGT/GAT-style) over the
    heterogeneous event graph. Deliberately small: a research prototype meant to
    train on one GPU, not a production system.

Both consume a correlation subgraph (from `graph_builder`) rooted at a seed and
return `(probability, severity)`. Both refuse to score a subgraph whose root is
not seed-eligible — the constraint is re-checked here rather than trusted,
because a scorer may be called on graphs assembled elsewhere.

Torch is imported lazily so that the rule baseline, the graph, the alerts and
the metrics all remain usable (and testable) on a machine with no GPU and no
PyTorch installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import networkx as nx

from .graph_builder import (NODE_PHYSICAL, NODE_NETWORK, EDGE_TEMPORAL,
                            EDGE_SPATIAL, EDGE_PRECEDES, SEED_ELIGIBLE,
                            SeedConstraintError)
from .network_stream import ANOMALY_TYPES


class CorrelationModel(ABC):
    """Scores a correlation subgraph rooted at a seed physical event."""

    @abstractmethod
    def score(self, subgraph: nx.MultiDiGraph, seed: str) -> tuple[float, float]:
        """Return (convergent-incident probability, severity), both in [0, 1]."""
        raise NotImplementedError

    @staticmethod
    def _check_seed(subgraph: nx.MultiDiGraph, seed: str) -> dict:
        d = subgraph.nodes[seed]
        if d.get("ntype") != NODE_PHYSICAL:
            raise SeedConstraintError(f"node {seed} is not a PhysicalEvent")
        if d["cls"] not in SEED_ELIGIBLE:
            raise SeedConstraintError(
                f"class {d['cls']!r} is consequence-only and cannot be scored as "
                "a correlation seed")
        return d


class TemporalRuleScorer(CorrelationModel):
    """Fixed-rule baseline: the H3 predictive-merit comparator.

    Rule set, all parameters explicit:
      1. A seed with no network evidence scores its own confidence, damped by
         `no_network_damping` — video alone is not convergence.
      2. Each network event within `window` of the seed adds evidence, weighted
         by `type_weight[anomaly_type]` and by its own confidence, decaying
         linearly with temporal distance.
      3. A network event that PRECEDES the physical event is worth
         `precedence_bonus`x more than one that follows it: reconnaissance
         before an intrusion is the signature the thesis is after.
      4. Severity rises with the number of distinct anomaly types and with the
         presence of consequence-only successors (a fire after vandalism is
         worse than vandalism alone).
    """

    def __init__(self,
                 window: float = 300.0,
                 type_weight: dict[str, float] | None = None,
                 precedence_bonus: float = 1.5,
                 no_network_damping: float = 0.35):
        self.window = float(window)
        self.type_weight = type_weight or {t: 1.0 for t in ANOMALY_TYPES}
        self.precedence_bonus = float(precedence_bonus)
        self.no_network_damping = float(no_network_damping)

    def score(self, subgraph: nx.MultiDiGraph, seed: str) -> tuple[float, float]:
        sd = self._check_seed(subgraph, seed)
        p_conf = float(sd.get("confidence", 0.0))
        t0, t1 = float(sd["t_start"]), float(sd["t_end"])

        evidence, types = 0.0, set()
        for n, d in subgraph.nodes(data=True):
            if d.get("ntype") != NODE_NETWORK:
                continue
            t = float(d["t"])
            gap = 0.0 if t0 <= t <= t1 else (t0 - t if t < t0 else t - t1)
            if gap > self.window:
                continue
            decay = max(0.0, 1.0 - gap / self.window)
            w = self.type_weight.get(d["anomaly_type"], 1.0)
            bonus = self.precedence_bonus if t < t0 else 1.0
            evidence += w * float(d.get("confidence", 0.0)) * decay * bonus
            types.add(d["anomaly_type"])

        if not types:
            return p_conf * self.no_network_damping, p_conf * self.no_network_damping

        prob = 1.0 - (1.0 - p_conf) * np.exp(-evidence)     # soft OR of evidence
        n_succ = sum(1 for _, _, e in subgraph.edges(data=True)
                     if e.get("etype") == EDGE_PRECEDES)
        severity = float(np.clip(0.4 * prob + 0.4 * min(len(types) / 3.0, 1.0)
                                 + 0.2 * min(n_succ / 2.0, 1.0), 0.0, 1.0))
        return float(np.clip(prob, 0.0, 1.0)), severity


# --------------------------------------------------------------------------
# Feature encoding shared by the GNN (kept here so the baseline and the GNN see
# exactly the same information — otherwise the H3 comparison is unfair).
# --------------------------------------------------------------------------
PHYS_DIM = 4 + len(SEED_ELIGIBLE)      # [conf, duration, is_seed, n_evidence] + class one-hot
NET_DIM = 2 + len(ANOMALY_TYPES)       # [conf, signed_dt] + type one-hot


def encode_physical(d: dict, seed_t0: float) -> np.ndarray:
    v = np.zeros(PHYS_DIM, dtype=np.float32)
    v[0] = float(d.get("confidence", 0.0))
    v[1] = float(d["t_end"]) - float(d["t_start"])
    v[2] = 1.0 if d.get("seed") else 0.0
    v[3] = float(len(d.get("evidence", {}) or {}))
    order = sorted(SEED_ELIGIBLE)
    if d["cls"] in order:                    # consequence-only classes get no one-hot:
        v[4 + order.index(d["cls"])] = 1.0   # they are context, never seeds
    return v


def encode_network(d: dict, seed_t0: float) -> np.ndarray:
    v = np.zeros(NET_DIM, dtype=np.float32)
    v[0] = float(d.get("confidence", 0.0))
    v[1] = float(d["t"]) - seed_t0           # signed: negative = precedes the seed
    v[2 + ANOMALY_TYPES.index(d["anomaly_type"])] = 1.0
    return v


def gather_neighbors(subgraph: nx.MultiDiGraph, seed: str):
    """Graph -> (x_seed, x_neigh, kinds, rel_types), pure numpy.

    This is where the graph/feature contract lives, so it is deliberately free of
    torch and covered by the offline test suite: an encoding bug here would
    silently corrupt every GNN result, and a GPU is not needed to catch it.

    Only the three relations that carry signal are followed
    (temporal_proximity, spatial_colocation, precedes); `asset_relation` edges
    lead to Asset nodes, which are context for the graph, not messages for the
    seed. Consequence-only successors (fire/crime/accident) DO arrive here, via
    `precedes` — that is the whole point of admitting them downstream.

    PER-RELATION MESSAGES (deliberate): a neighbour connected by two relations —
    e.g. a network event that is both within the temporal window AND on the same
    asset — is emitted ONCE PER RELATION, so the typed attention can weigh
    "co-located" evidence differently from merely "co-occurring" evidence. The
    returned lists are therefore aligned per EDGE, not per node; `len(kinds)` is
    the number of typed messages, not the number of distinct neighbours.
    """
    d_seed = subgraph.nodes[seed]
    if d_seed.get("ntype") != NODE_PHYSICAL:
        raise SeedConstraintError(f"node {seed} is not a PhysicalEvent")
    if d_seed["cls"] not in SEED_ELIGIBLE:
        raise SeedConstraintError(
            f"class {d_seed['cls']!r} is consequence-only and cannot be a seed")

    seed_t0 = float(d_seed["t_start"])
    width = max(PHYS_DIM, NET_DIM)

    x_seed = np.zeros(PHYS_DIM, dtype=np.float32)
    x_seed[:] = encode_physical(d_seed, seed_t0)

    feats, kinds, rels = [], [], []
    for _, nb, e in subgraph.out_edges(seed, data=True):
        et = e.get("etype")
        if et not in (EDGE_TEMPORAL, EDGE_SPATIAL, EDGE_PRECEDES):
            continue                      # asset_relation: context, not a message
        d = subgraph.nodes[nb]
        if d.get("ntype") == NODE_NETWORK:
            v = encode_network(d, seed_t0)
        elif d.get("ntype") == NODE_PHYSICAL:
            v = encode_physical(d, seed_t0)
        else:
            continue
        padded = np.zeros(width, dtype=np.float32)
        padded[:len(v)] = v
        feats.append(padded); kinds.append(d["ntype"]); rels.append(et)

    x_neigh = (np.stack(feats) if feats
               else np.zeros((0, width), dtype=np.float32))
    return x_seed, x_neigh, kinds, rels


class HeteroGNNScorer(CorrelationModel):
    """Small typed-attention GNN over the heterogeneous event graph.

    Message passing runs on two typed relations that carry the physical/network
    coupling (`temporal_proximity`, `spatial_colocation`) plus the directed
    `precedes` relation that admits consequence-only nodes. Node features come
    from `encode_physical` / `encode_network`; a readout at the seed produces the
    probability and the severity.

    The model is intentionally shallow (2 layers): the graphs are small (one
    seed, a handful of network events, a few successors) and the dataset the
    thesis will build is small. Depth would overfit — the lesson already learned
    with the TFM in the main paper, and documented in ARCHITECTURE.md §6.
    """

    def __init__(self, hidden: int = 32, heads: int = 2, layers: int = 2,
                 device: str = "cpu"):
        self.hidden, self.heads, self.layers, self.device = hidden, heads, layers, device
        self._net = None      # built lazily so importing this module never needs torch

    # -- torch is only required when the GNN is actually used --
    def _build(self):
        import torch
        import torch.nn as nn

        class TypedAttentionLayer(nn.Module):
            """One round of typed attention: each relation gets its own
            projection, and the seed attends over its typed neighbourhoods."""

            def __init__(self, dim, heads):
                super().__init__()
                self.heads = heads
                self.rel = nn.ModuleDict({
                    EDGE_TEMPORAL: nn.Linear(dim, dim),
                    EDGE_SPATIAL: nn.Linear(dim, dim),
                    EDGE_PRECEDES: nn.Linear(dim, dim),
                })
                self.q = nn.Linear(dim, dim)
                self.k = nn.Linear(dim, dim)
                self.v = nn.Linear(dim, dim)
                self.out = nn.Linear(dim, dim)
                self.norm = nn.LayerNorm(dim)

            def forward(self, h_seed, neigh, rel_types):
                # neigh: (N, dim); rel_types: list[str] of length N
                if neigh.shape[0] == 0:
                    return self.norm(h_seed)
                proj = torch.stack([self.rel[r](neigh[i])
                                    for i, r in enumerate(rel_types)])
                q = self.q(h_seed).unsqueeze(0)                 # (1, dim)
                k, v = self.k(proj), self.v(proj)               # (N, dim)
                att = torch.softmax((q @ k.T) / (k.shape[-1] ** 0.5), dim=-1)
                msg = (att @ v).squeeze(0)
                return self.norm(h_seed + self.out(msg))

        class Net(nn.Module):
            def __init__(self, hidden, heads, layers):
                super().__init__()
                self.emb_p = nn.Linear(PHYS_DIM, hidden)
                self.emb_n = nn.Linear(NET_DIM, hidden)
                self.layers = nn.ModuleList(
                    [TypedAttentionLayer(hidden, heads) for _ in range(layers)])
                self.head_prob = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                               nn.Linear(hidden, 1))
                self.head_sev = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                              nn.Linear(hidden, 1))

            def forward(self, x_seed, x_neigh, kinds, rel_types):
                h = self.emb_p(x_seed)
                if x_neigh.shape[0]:
                    hn = torch.stack([
                        self.emb_p(x_neigh[i]) if kinds[i] == NODE_PHYSICAL
                        else self.emb_n(x_neigh[i][:NET_DIM])
                        for i in range(x_neigh.shape[0])])
                else:
                    hn = torch.zeros(0, h.shape[-1])
                for layer in self.layers:
                    h = layer(h, hn, rel_types)
                return (torch.sigmoid(self.head_prob(h)).squeeze(),
                        torch.sigmoid(self.head_sev(h)).squeeze())

        self._net = Net(self.hidden, self.heads, self.layers).to(self.device)
        return self._net

    def tensors(self, subgraph: nx.MultiDiGraph, seed: str):
        """Graph -> torch tensors. Thin wrapper over `gather_neighbors`, which
        holds the actual logic and is torch-free (and therefore unit-tested)."""
        import torch
        x_seed, x_neigh, kinds, rels = gather_neighbors(subgraph, seed)
        return (torch.from_numpy(x_seed), torch.from_numpy(x_neigh), kinds, rels)

    def score(self, subgraph: nx.MultiDiGraph, seed: str) -> tuple[float, float]:
        import torch
        net = self._net or self._build()
        xs, xn, kinds, rels = self.tensors(subgraph, seed)
        net.eval()
        with torch.no_grad():
            p, s = net(xs, xn, kinds, rels)
        return float(p), float(s)

    def parameters(self):
        return (self._net or self._build()).parameters()

    def state_dict(self):
        return (self._net or self._build()).state_dict()

    def load_state_dict(self, sd):
        (self._net or self._build()).load_state_dict(sd)
