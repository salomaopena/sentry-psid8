"""Heterogeneous event graph for SENTRY-C.

Builds, per correlation window, a typed graph over three node kinds
(PhysicalEvent, NetworkEvent, Asset) and four edge kinds (temporal_proximity,
spatial_colocation, asset_relation, precedes).

LOAD-BEARING CONSTRAINT — the primary-fusion class set
-----------------------------------------------------
Only five of the eight PSID-8 classes may SEED a correlation subgraph:

    intrusion, suspicious_behavior, vandalism, fall, suspicious_object

These are the classes that satisfy the three fusion-justification conditions of
the thesis: (i) causal/temporal plausibility with network events, (ii) ambiguity
asymmetry — the physical evidence alone underdetermines the incident, so a
network signal can disambiguate it, and (iii) network coverage of the asset
involved.

The other three classes — accident, fire, crime — remain fully part of PSID-8
and of the standalone SENTRY evaluation, but they are modelled here as
SEQUENCE/CONSEQUENCE events: they may enter the graph only as successors of a
seed, reachable through a directed `precedes` edge with a positive time delta
(e.g. vandalism against a suppression sensor preceding a fire). They are never
correlation seeds on their own, because a fire is not disambiguated by a port
scan; asserting such a correlation would be a causal claim the thesis does not
support.

This is enforced in code (`SEED_ELIGIBLE`, `_assert_seedable`) rather than left
as a convention, so a future edit cannot silently violate it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import networkx as nx

from .network_stream import NetworkEvent

# PSID-8 canonical class ids (identical to sentry/: do not renumber).
CLASS_NAMES = ["accident", "suspicious_behavior", "crime", "fire",
               "intrusion", "suspicious_object", "fall", "vandalism"]
CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

#: Classes eligible to seed a correlation subgraph (see module docstring).
SEED_ELIGIBLE: frozenset[str] = frozenset({
    "intrusion", "suspicious_behavior", "vandalism", "fall", "suspicious_object",
})
#: Classes that may only appear downstream of a seed, via a `precedes` edge.
CONSEQUENCE_ONLY: frozenset[str] = frozenset({"accident", "fire", "crime"})

assert SEED_ELIGIBLE | CONSEQUENCE_ONLY == set(CLASS_NAMES), \
    "every PSID-8 class must be either seed-eligible or consequence-only"

NODE_PHYSICAL, NODE_NETWORK, NODE_ASSET = "PhysicalEvent", "NetworkEvent", "Asset"
EDGE_TEMPORAL = "temporal_proximity"
EDGE_SPATIAL = "spatial_colocation"
EDGE_ASSET = "asset_relation"
EDGE_PRECEDES = "precedes"


class SeedConstraintError(ValueError):
    """Raised when a subgraph is asked to be seeded by a consequence-only class."""


@dataclass(frozen=True)
class Asset:
    """A monitored asset: the join point between the physical and network views."""
    asset_id: str            # matches SENTRY's camera_id and the IDS asset_id
    zone: str | None = None  # declared physical zone (for spatial_colocation)
    segment: str | None = None   # network segment
    meta: dict[str, Any] = field(default_factory=dict)


def physical_event_class(alert: dict) -> str:
    """Class name of a SENTRY alert record (which carries `class_id`)."""
    return CLASS_NAMES[int(alert["class_id"])]


def _assert_seedable(alert: dict) -> None:
    cls = physical_event_class(alert)
    if cls not in SEED_ELIGIBLE:
        raise SeedConstraintError(
            f"class {cls!r} is consequence-only and cannot seed a correlation "
            f"subgraph; seed-eligible classes are {sorted(SEED_ELIGIBLE)}. "
            "Attach it as a successor of a seed via a `precedes` edge instead."
        )


class FusionJustification:
    """Scores that decide whether a physical/network pair is worth correlating.

    gain      : P(e_n | e_p, class) - P(e_n)  — how much the physical event
                raises the probability of the network anomaly above its base
                rate. A pair with gain <= 0 carries no evidence.
    amb_p     : ambiguity of the physical event — how underdetermined the
                incident is given video alone (1 = fully ambiguous).
    amb_n     : ambiguity of the network event, likewise.

    All three are supplied by the caller (estimated from data or from a declared
    prior), never hardcoded here. The thresholds `theta` (minimum gain) and
    `tau` (minimum ambiguity, i.e. minimum room for disambiguation) are
    configurable and belong to the experiment, not to the library.
    """

    def __init__(self,
                 p_joint: dict[tuple[str, str], float] | None = None,
                 p_base: dict[str, float] | None = None,
                 amb_physical: dict[str, float] | None = None,
                 amb_network: dict[str, float] | None = None):
        self.p_joint = p_joint or {}      # (class, anomaly_type) -> P(e_n | e_p)
        self.p_base = p_base or {}        # anomaly_type -> P(e_n)
        self.amb_physical = amb_physical or {}   # class -> Amb(e_p)
        self.amb_network = amb_network or {}     # anomaly_type -> Amb(e_n)

    def gain(self, cls: str, anomaly_type: str) -> float:
        return (self.p_joint.get((cls, anomaly_type), 0.0)
                - self.p_base.get(anomaly_type, 0.0))

    def amb_p(self, cls: str) -> float:
        return self.amb_physical.get(cls, 0.0)

    def amb_n(self, anomaly_type: str) -> float:
        return self.amb_network.get(anomaly_type, 0.0)

    def passes(self, cls: str, anomaly_type: str,
               theta: float, tau: float) -> bool:
        """A pair is retained iff the physical event raises the anomaly's
        probability by at least `theta` AND there is at least `tau` ambiguity on
        both sides — i.e. something actually remains to be disambiguated."""
        return (self.gain(cls, anomaly_type) >= theta
                and min(self.amb_p(cls), self.amb_n(anomaly_type)) >= tau)


class EventGraphBuilder:
    """Builds one heterogeneous graph per correlation window.

    Parameters
    ----------
    delta_t : correlation window (same time unit as the event timestamps).
    assets  : known assets, keyed by asset_id.
    justification : FusionJustification instance (optional; when omitted, no
                    pair is filtered and every temporal/spatial edge is kept).
    theta, tau : fusion-justification thresholds (see FusionJustification).
    """

    def __init__(self, delta_t: float, assets: Sequence[Asset] = (),
                 justification: FusionJustification | None = None,
                 theta: float = 0.0, tau: float = 0.0):
        self.delta_t = float(delta_t)
        self.assets = {a.asset_id: a for a in assets}
        self.just = justification
        self.theta = theta
        self.tau = tau

    # ---------- node ids ----------
    @staticmethod
    def _pid(i: int) -> str: return f"P{i}"
    @staticmethod
    def _nid(i: int) -> str: return f"N{i}"
    @staticmethod
    def _aid(asset_id: str) -> str: return f"A:{asset_id}"

    def build(self,
              physical: Sequence[dict],
              network: Sequence[NetworkEvent],
              seeds: Sequence[int] | None = None) -> nx.MultiDiGraph:
        """Assemble the graph.

        physical : SENTRY alert records. Each MUST carry `class_id`, `t_start`,
                   `t_end` and `asset_id` (the camera that produced it).
        network  : NetworkEvent list (already windowed by the caller if desired).
        seeds    : indices into `physical` to be marked as correlation seeds.
                   Defaults to every seed-eligible physical event. Passing the
                   index of a consequence-only class raises SeedConstraintError.
        """
        G = nx.MultiDiGraph()

        # --- asset nodes ---
        for aid, a in self.assets.items():
            G.add_node(self._aid(aid), ntype=NODE_ASSET, asset_id=aid,
                       zone=a.zone, segment=a.segment)

        # --- physical event nodes ---
        for i, ev in enumerate(physical):
            cls = physical_event_class(ev)
            G.add_node(self._pid(i), ntype=NODE_PHYSICAL, index=i, cls=cls,
                       class_id=int(ev["class_id"]),
                       t_start=float(ev["t_start"]), t_end=float(ev["t_end"]),
                       confidence=float(ev.get("confidence", 0.0)),
                       asset_id=ev.get("asset_id"),
                       evidence=ev.get("evidence_terms", {}),
                       seed=False,
                       seed_eligible=cls in SEED_ELIGIBLE)

        # --- seeds (constraint enforced here) ---
        if seeds is None:
            seeds = [i for i, ev in enumerate(physical)
                     if physical_event_class(ev) in SEED_ELIGIBLE]
        for i in seeds:
            _assert_seedable(physical[i])          # raises for accident/fire/crime
            G.nodes[self._pid(i)]["seed"] = True

        # --- network event nodes ---
        for j, ne in enumerate(network):
            G.add_node(self._nid(j), ntype=NODE_NETWORK, index=j,
                       anomaly_type=ne.anomaly_type, t=float(ne.timestamp),
                       confidence=float(ne.confidence), asset_id=ne.asset_id,
                       evidence=ne.evidence)

        # --- asset_relation edges ---
        for i, ev in enumerate(physical):
            aid = ev.get("asset_id")
            if aid in self.assets:
                G.add_edge(self._pid(i), self._aid(aid), etype=EDGE_ASSET)
        for j, ne in enumerate(network):
            if ne.asset_id in self.assets:
                G.add_edge(self._nid(j), self._aid(ne.asset_id), etype=EDGE_ASSET)

        # --- physical <-> network edges (temporal + spatial), justification-filtered ---
        for i, ev in enumerate(physical):
            cls = physical_event_class(ev)
            p_lo, p_hi = float(ev["t_start"]), float(ev["t_end"])
            for j, ne in enumerate(network):
                dt = _interval_gap(p_lo, p_hi, ne.timestamp)
                if dt > self.delta_t:
                    continue
                if self.just is not None and not self.just.passes(
                        cls, ne.anomaly_type, self.theta, self.tau):
                    continue
                attrs = dict(etype=EDGE_TEMPORAL, dt=dt)
                if self.just is not None:
                    attrs.update(gain=self.just.gain(cls, ne.anomaly_type),
                                 amb_p=self.just.amb_p(cls),
                                 amb_n=self.just.amb_n(ne.anomaly_type))
                G.add_edge(self._pid(i), self._nid(j), **attrs)
                G.add_edge(self._nid(j), self._pid(i), **attrs)   # undirected pair

                if self._colocated(ev.get("asset_id"), ne.asset_id):
                    G.add_edge(self._pid(i), self._nid(j), etype=EDGE_SPATIAL)
                    G.add_edge(self._nid(j), self._pid(i), etype=EDGE_SPATIAL)

        # --- precedes edges: seed -> consequence (directed, positive delta) ---
        for i, ev in enumerate(physical):
            if not G.nodes[self._pid(i)]["seed"]:
                continue
            for k, ev2 in enumerate(physical):
                if k == i:
                    continue
                cls2 = physical_event_class(ev2)
                delta = float(ev2["t_start"]) - float(ev["t_end"])
                if delta <= 0 or delta > self.delta_t:
                    continue
                if cls2 in CONSEQUENCE_ONLY or cls2 in SEED_ELIGIBLE:
                    # a seed may precede a consequence, or another primary event;
                    # the direction is always seed -> successor, never the reverse
                    G.add_edge(self._pid(i), self._pid(k), etype=EDGE_PRECEDES,
                               dt=delta)
        return G

    def _colocated(self, asset_a: str | None, asset_b: str | None) -> bool:
        if asset_a is None or asset_b is None:
            return False
        if asset_a == asset_b:
            return True
        za = self.assets.get(asset_a)
        zb = self.assets.get(asset_b)
        return bool(za and zb and za.zone is not None and za.zone == zb.zone)

    # ---------- subgraph extraction ----------
    def correlation_subgraphs(self, G: nx.MultiDiGraph) -> dict[str, nx.MultiDiGraph]:
        """One subgraph per seed: the seed, everything it reaches, and the assets
        involved. Consequence-only nodes appear here — and only here — because
        they are reachable from a seed."""
        out = {}
        for n, d in G.nodes(data=True):
            if d.get("ntype") == NODE_PHYSICAL and d.get("seed"):
                reach = nx.descendants(G, n) | {n}
                out[n] = G.subgraph(reach).copy()
        return out


def _interval_gap(lo: float, hi: float, t: float) -> float:
    """Temporal distance between the instant `t` and the interval [lo, hi]."""
    if lo <= t <= hi:
        return 0.0
    return lo - t if t < lo else t - hi
