# Changelog

## [Unreleased] — SENTRY-C convergence layer

### Added

- **`sentryc/` package** — the physical-cyber convergence layer. Additive: it
  consumes SENTRY's existing structured alert records and does not modify how the
  visual model is trained or evaluated.
  - `network_stream.py` — normalized `NetworkEvent` schema and the
    `NetworkEventSource` adapter interface. Closed `anomaly_type` vocabulary:
    `port_scan`, `new_device`, `exfiltration`, `dos_camera`,
    `unauthorized_access`. This module is not an IDS; the detector is a separate
    component.
  - `graph_builder.py` — heterogeneous event graph (`PhysicalEvent`,
    `NetworkEvent`, `Asset`; edges `temporal_proximity`, `spatial_colocation`,
    `asset_relation`, `precedes`) with the fusion-justification score
    P(e_n|e_p,class) − P(e_n) and configurable thresholds θ, τ.
  - `gnn_correlation.py` — `HeteroGNNScorer` (small typed-attention GNN) and
    `TemporalRuleScorer` (interpretable baseline) behind one interface, so the
    thesis protocol (H3) can compare them on identical inputs.
  - `alerts.py` — `ConvergentAlert` with `time_to_detection` computed from the
    first contributing event of either modality (lead time is a primary claim).
  - `metrics.py` — convergent event F1/AUC (reusing the temporal-IoU matching of
    `sentry/metrics.py`), lead-time gain with bootstrap CI, false-positive
    reduction, and modality contribution.
- **`tests/test_sentryc.py`** — 14 tests, pure numpy/networkx, no GPU required.
- **`ARCHITECTURE.md` §10** — the convergence layer, its constraints and failure
  modes.

### PSID-8 class roles in the convergence layer

The eight PSID-8 classes remain intact and fully annotated for the standalone
SENTRY evaluation. Within SENTRY-C they play two distinct roles:

- **Primary fusion nodes (may seed a correlation subgraph):**
  `intrusion`, `suspicious_behavior`, `vandalism`, `fall`, `suspicious_object`.
  These satisfy the three fusion-justification conditions (causal/temporal
  plausibility, ambiguity asymmetry, network coverage of the asset).
- **Sequence/consequence-only nodes (never seeds):**
  `accident`, `fire`, `crime`. They may appear only as successors of a seed,
  reachable through a directed `precedes` edge with a positive time delta.

This distinction is enforced in code (`SEED_ELIGIBLE` / `CONSEQUENCE_ONLY`,
`SeedConstraintError`, re-checked at scoring time) and asserted by the test
suite in both directions — rejection as a seed, and reachability as a successor.

### Unchanged

- `sentry/modules.py`, `sentry/ultralytics_adapter.py`, `sentry/tubes.py`: the
  frozen visual pipeline behind the published results.
- `sentry/metrics.py`: imported by `sentryc/metrics.py`, never edited by it.

### Results

[PENDING] — no convergence experiment has been run. No number will enter any
write-up unless it comes from a real logged run.
