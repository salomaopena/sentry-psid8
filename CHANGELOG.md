# Changelog

## [Unreleased] — Code audit: critical and medium fixes

A structured audit of the repository (engineering + academic review) found and
fixed several real defects, verified by new automated tests (31 tests total
across four suites: `tests/run_tests.py`, `tests/test_sentryc.py`,
`tests/test_psid8_scripts.py`, `tests/test_train.py`). None of the frozen
Level S / Level E results reported in the paper are affected: `sentry/modules.py`,
`sentry/ultralytics_adapter.py` and `sentry/tubes.py` were not touched.

### Fixed (critical)

- **`sentry/train.py` was silently non-functional.** The script hard-coded
  `decoded = None`, so its loss branch never executed and
  `backward()`/`optimizer.step()` never ran for any epoch, while it printed
  `"epoch N: ok"` and saved a checkpoint containing only the TFM's random
  initialization. Rewritten to mirror the supervised approach validated on
  Kaggle (v8DetectionLoss + auxiliary L_tc); now raises `RuntimeError` if an
  epoch produces no gradient, instead of reporting false success. See
  `tests/test_train.py::test_stage_b_actually_updates_tfm_weights`.
- **`psid8/scripts/coco_to_yolo.py` never used its `images_dir` argument** and
  wrote a flat `labels/` folder incompatible with the `clips/<clip_id>/frames+
  labels` contract the rest of the pipeline (`sentry/data.py::VideoClipDataset`,
  `build_splits.py`) expects. Rewritten with two explicit layouts:
  `--layout clips` (default; validates every referenced image exists, writes
  labels next to the clip's frames) and `--layout flat` (legacy single-folder
  image-pilot workflow, now genuinely copying/symlinking images, including
  previously-dropped unannotated/background images). See
  `tests/test_psid8_scripts.py`.
- **`psid8/scripts/integrity_check.py` used `datetime.UTC`**, which requires
  Python 3.11+, while `pyproject.toml` declares `requires-python = ">=3.10"`.
  Changed to `datetime.timezone.utc` (available since Python 3.2).
- **`pyproject.toml` dependencies were out of sync with `requirements.txt`**:
  `torch`, `ultralytics`, `opencv-python`, `scikit-learn` and `tqdm` were
  missing, so `pip install -e .` produced an incomplete environment. Synced.

### Fixed (medium)

- **Duplicated, diverging `aggregate()` implementations** in
  `sentry/aggregate.py` and `sentry/seeds.py`, with different output schemas
  for nominally the same computation. `sentry/aggregate.py` now delegates its
  numeric core to `sentry/seeds.py::aggregate` (single source of truth) and
  adds only a human-readable `"formatted"` field on top. See
  `tests/run_tests.py::test_aggregate_cli_matches_seeds_core`.

### Changed (performance: time and memory)

- **Stage B training is now batched across clips.** The old loop called the
  model once per *(clip, frame)* pair; `ConvGRUCell`/`MotionGate`/
  `TemporalFeatureMemory` already support an arbitrary batch dimension, so the
  rewritten loop (`sentry/train.py` + new `sentry.data.collate_batched`) stacks
  the mini-batch's clips and calls the model once per timestep for the whole
  batch (`T` calls instead of `N*T`). Measured on CPU with a tiny model:
  2x fewer forward calls, 4.05x wall-clock reduction for one epoch. Mixed
  precision (autocast + GradScaler) is additionally enabled by default on CUDA.
  See ARCHITECTURE.md section 7.1 and
  `tests/test_train.py::test_batched_loop_uses_fewer_model_calls`.

### Fixed (low severity)

- Removed dead imports with no functional effect: `psid8/scripts/dataset_stats.py`
  (`defaultdict`), `sentry/plots.py` (`json`, `os`), `sentryc/graph_builder.py`
  (`Iterable`), `sentryc/metrics.py` (`Iterable`), `sentryc/gnn_correlation.py`
  (`Any`), `tests/test_sentryc.py` (two dead imports in test bodies).

### Added

- `tests/test_psid8_scripts.py` (6 tests): `coco_to_yolo.py` both layouts, the
  loud-failure path for desynchronized images/labels, and a real integration
  test confirming the output is directly loadable by `VideoClipDataset`.
- `tests/test_train.py` (4 tests): gradient-flow correctness, loud failure on
  an all-background epoch, Ultralytics-version-shape compatibility, and the
  measured batching speedup.
- `sentry.data.collate_batched`, `sentry.train.extract_feat_list`,
  `sentry.train.build_criterion`, `sentry.train.run_stage_b`.

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
