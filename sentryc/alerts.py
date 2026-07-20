"""Convergent alert records for SENTRY-C.

A convergent alert is emitted for a SEED physical event whose correlation
subgraph carries enough network evidence. It is deliberately richer than the
unimodal alert: it names the asset that ties the two views together, and it
records `time_to_detection`.

Why `time_to_detection` is first class: the thesis claims lead time
over unimodal detectors as a primary benefit of convergence -
a port scan against a camera may precede the intrusion it enables. The metric is
therefore computed from the FIRST contributing event of any modality (physical
or network) to the moment the convergent alert can be emitted, so an alert that
fires only after the physical event has ended shows no lead over a physical-only
detector, and the metric says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import networkx as nx

from .graph_builder import NODE_NETWORK, NODE_PHYSICAL


@dataclass
class ConvergentAlert:
    """Operator-facing record of a convergent incident."""

    cls: str                              # class of the seed physical event
    physical_evidence: list[dict]         # contributing physical events
    network_evidence: list[dict]          # contributing network events
    severity: float                       # [0, 1], from the correlation model
    probability: float                    # convergent-incident probability
    time_to_detection: float              # emission time - first contributing event
    correlated_asset: str | None          # the asset joining both views
    emitted_at: float                     # decision instant on the shared clock
    first_evidence_at: float              # earliest contributing event
    seed_node: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    # Interop with sentry/metrics.py: convergent alerts are scored with the same
    # temporal-IoU machinery as physical events, so they must expose the same
    # keys. The span is the seed's span.
    def as_event(self, class_id: int) -> dict:
        spans = [(e["t_start"], e["t_end"]) for e in self.physical_evidence] or \
                [(self.emitted_at, self.emitted_at)]
        return {"class_id": class_id,
                "t_start": min(s for s, _ in spans),
                "t_end": max(e for _, e in spans),
                "confidence": float(self.probability)}


def emission_time(subgraph: nx.MultiDiGraph, seed: str) -> float:
    """Instant at which the convergent decision can be taken: the last
    contributing event must have happened. Physical events contribute at
    `t_end` (the tube is only confirmed once it ends); network events at `t`."""
    times = []
    for n, d in subgraph.nodes(data=True):
        if d.get("ntype") == NODE_PHYSICAL:
            times.append(float(d["t_end"]))
        elif d.get("ntype") == NODE_NETWORK:
            times.append(float(d["t"]))
    return max(times) if times else float(subgraph.nodes[seed]["t_end"])


def first_evidence_time(subgraph: nx.MultiDiGraph) -> float:
    """Earliest instant at which ANY contributing event began."""
    times = []
    for _, d in subgraph.nodes(data=True):
        if d.get("ntype") == NODE_PHYSICAL:
            times.append(float(d["t_start"]))
        elif d.get("ntype") == NODE_NETWORK:
            times.append(float(d["t"]))
    return min(times) if times else 0.0


def build_alert(subgraph: nx.MultiDiGraph, seed: str, probability: float,
                severity: float) -> ConvergentAlert:
    """Assemble the operator-facing record from a scored correlation subgraph."""
    phys, net = [], []
    for n, d in subgraph.nodes(data=True):
        if d.get("ntype") == NODE_PHYSICAL:
            phys.append({"node": n, "cls": d["cls"], "class_id": d["class_id"],
                         "t_start": d["t_start"], "t_end": d["t_end"],
                         "confidence": d.get("confidence", 0.0),
                         "asset_id": d.get("asset_id"),
                         "evidence": d.get("evidence", {}),
                         "is_seed": bool(d.get("seed"))})
        elif d.get("ntype") == NODE_NETWORK:
            net.append({"node": n, "anomaly_type": d["anomaly_type"],
                        "t": d["t"], "confidence": d.get("confidence", 0.0),
                        "asset_id": d.get("asset_id"),
                        "evidence": d.get("evidence", {})})

    t_emit = emission_time(subgraph, seed)
    t_first = first_evidence_time(subgraph)
    sd = subgraph.nodes[seed]

    return ConvergentAlert(
        cls=sd["cls"],
        physical_evidence=sorted(phys, key=lambda e: e["t_start"]),
        network_evidence=sorted(net, key=lambda e: e["t"]),
        severity=float(severity),
        probability=float(probability),
        time_to_detection=float(t_emit - t_first),
        correlated_asset=sd.get("asset_id"),
        emitted_at=float(t_emit),
        first_evidence_at=float(t_first),
        seed_node=seed,
    )
