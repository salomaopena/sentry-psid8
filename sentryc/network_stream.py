"""Network-anomaly event stream for SENTRY-C.

This module defines ONLY the normalized event representation and the adapter
interface that feeds it. It is deliberately not an IDS: the detection model
(CNN/LSTM over traffic features) is a separate, out-of-scope component. Keeping
the schema and the detector apart is what allows a real IDS, a replayed capture,
or a synthetic generator to be swapped without touching the fusion layer.

Design decision (load-bearing): `anomaly_type` is a CLOSED vocabulary. An open
vocabulary would make the fusion-justification probabilities of
`graph_builder.py` unestimable, because P(e_n | e_p, class) is conditioned on
the anomaly type. New types must be added here explicitly, not invented by an
adapter at runtime.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

# Closed vocabulary of network anomaly types.
ANOMALY_TYPES = (
    "port_scan",            # reconnaissance against the asset or its segment
    "new_device",           # previously unseen device on the segment
    "exfiltration",         # anomalous outbound volume / destination
    "dos_camera",           # denial of service targeting a camera endpoint
    "unauthorized_access",  # authentication failure or use outside policy
)


@dataclass
class NetworkEvent:
    """A single normalized network-anomaly event.

    timestamp    : event time on the SAME clock as physical events (see the
                   clock-alignment note in ARCHITECTURE.md §10).
    asset_id     : identifier of the asset the anomaly concerns (e.g. a camera
                   or a controller); must match an `Asset` id known to the graph.
    anomaly_type : one of ANOMALY_TYPES.
    confidence   : detector confidence in [0, 1].
    evidence     : free-form detector-specific payload (feature attributions,
                   source/destination, packet counts). Never used for matching,
                   only carried into the convergent alert for the operator.
    """

    timestamp: float
    asset_id: str
    anomaly_type: str
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.anomaly_type not in ANOMALY_TYPES:
            raise ValueError(
                f"unknown anomaly_type {self.anomaly_type!r}; the vocabulary is "
                f"closed: {ANOMALY_TYPES}"
            )
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1], got {self.confidence}")

    def to_dict(self) -> dict:
        return asdict(self)


class NetworkEventSource(ABC):
    """Adapter interface. Any producer of network anomalies implements this.

    Implementations may wrap a trained IDS, replay a labelled capture, or
    generate synthetic events for controlled experiments. The fusion layer never
    knows which it is talking to.
    """

    @abstractmethod
    def events(self, t_start: float, t_end: float) -> Iterable[NetworkEvent]:
        """Yield every event whose timestamp lies in [t_start, t_end]."""
        raise NotImplementedError

    def assets(self) -> set[str]:
        """Asset ids this source can report on. Override when known; the default
        is derived lazily by the caller from the events themselves."""
        return set()


class ListEventSource(NetworkEventSource):
    """Trivial source backed by an in-memory list. Used by tests and by replay
    of a pre-recorded/synthetic stream."""

    def __init__(self, events: Iterable[NetworkEvent]):
        self._events = sorted(events, key=lambda e: e.timestamp)

    def events(self, t_start: float, t_end: float) -> list[NetworkEvent]:
        return [e for e in self._events if t_start <= e.timestamp <= t_end]

    def assets(self) -> set[str]:
        return {e.asset_id for e in self._events}
