"""SENTRY-C: the physical-cyber convergence layer.

Additive to SENTRY: consumes its structured alert records, never alters how the
visual model is trained or evaluated. See ARCHITECTURE.md section 10.
"""
__version__ = "0.1.0"

from .network_stream import (NetworkEvent, NetworkEventSource, ListEventSource,
                             ANOMALY_TYPES)
from .graph_builder import (EventGraphBuilder, Asset, FusionJustification,
                            SEED_ELIGIBLE, CONSEQUENCE_ONLY, SeedConstraintError,
                            CLASS_NAMES, CLASS_ID)
from .alerts import ConvergentAlert, build_alert
from .gnn_correlation import (CorrelationModel, TemporalRuleScorer, HeteroGNNScorer,
                              gather_neighbors, encode_physical, encode_network,
                              PHYS_DIM, NET_DIM)

__all__ = [
    "NetworkEvent", "NetworkEventSource", "ListEventSource", "ANOMALY_TYPES",
    "EventGraphBuilder", "Asset", "FusionJustification", "SEED_ELIGIBLE",
    "CONSEQUENCE_ONLY", "SeedConstraintError", "CLASS_NAMES", "CLASS_ID",
    "ConvergentAlert", "build_alert",
    "CorrelationModel", "TemporalRuleScorer", "HeteroGNNScorer",
    "gather_neighbors", "encode_physical", "encode_network", "PHYS_DIM", "NET_DIM",
]
