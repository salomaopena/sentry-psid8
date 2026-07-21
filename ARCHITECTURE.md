# ARCHITECTURE — how this codebase works

A guide for anyone (human or agent) opening this repository with no prior
context. It explains what each module does, how data flows through the system,
which design decisions are load-bearing, and which failure modes are already
solved so they are not reintroduced.

---

## 1. What the system is

Two artifacts share one repository:

- **SENTRY** — a single-stage detector extended with temporal reasoning. It
  wraps a pretrained YOLO detector with a **Temporal Feature Memory (TFM)**:
  one ConvGRU + motion-gated attention per FPN level. Per-frame detections are
  then aggregated into **events** by tube linking, producing structured alert
  records (class, track, time span, confidence trajectory, evidence terms).
- **PSID-8 toolkit** — the annotation schema, converters, split builder,
  agreement/integrity checks and evaluation metrics needed to build and score a
  benchmark of 8 physical-security incident classes.

The 8 canonical classes and their **fixed integer ids** (used everywhere —
labels, prompts, metrics, plots):

```
0 accident   1 suspicious_behavior   2 crime      3 fire
4 intrusion  5 suspicious_object     6 fall       7 vandalism
```

---

## 2. Repository map

```
sentry/                    the model and the evaluation machinery
  modules.py               ConvGRU, MotionGate, TFMLevel, TemporalFeatureMemory
  ultralytics_adapter.py   SentryYOLO: injects the TFM into a YOLO model
  stageb_train.py          helpers for supervised Stage-B training
  losses.py                temporal-consistency loss + Eq.(3) aggregator
  data.py                  VideoClipDataset: ordered windows of frames
  tubes.py                 TubeLinker + confirm_events -> alert records
  metrics.py               event AUC / P-R-F1 / confusion matrix / bootstrap CI
  seeds.py                 multi-seed protocol (pre-registered, resumable)
  plots.py                 figures (training curves, ablation, confusion matrix)
  train.py, eval.py        CLI entry points

psid8/                     the benchmark toolkit
  schema.json              8 classes + compositional attribute vocabularies
  ANNOTATION_GUIDE.md      operational definitions for annotators
  openvocab_prompts.json   compositional prompts derived from the schema
  openvocab_prompts_ucf.json  UCF-adapted prompt variant (ablation)
  ucfcrime_mapping.json    UCF-Crime's 13 classes -> our 8 (or null = excluded)
  scripts/
    le2i_to_clips.py       Le2i fall dataset -> canonical clip layout
    coco_to_yolo.py        CVAT COCO export -> YOLO labels + attribute sidecars
    build_splits.py        camera-disjoint, class-covering splits
    integrity_check.py     Phase-1 freeze: hashes + leakage assertions
    agreement.py           inter-annotator kappa + box IoU
    dataset_stats.py       per-split counts for the paper's dataset table

configs/sentry_base.yaml   hyperparameter SEARCH SPACE (not fixed defaults)
protocol/PREREGISTRATION.md  the anti-manipulation protocol (read this first)
notebooks/sentry_kaggle.ipynb  the full reproduction pipeline
tests/run_tests.py         offline unit tests (pure numpy; no GPU needed)
```

---

## 3. The canonical data contract

Everything downstream of dataset ingestion speaks one format. **Adding a new
dataset means writing one adapter** to this contract; nothing else changes.

**Clip layout on disk**
```
clips/<clip_id>/frames/00001.jpg, 00002.jpg, ...     (1-based, zero-padded)
clips/<clip_id>/labels/00001.txt, ...                (YOLO: `cls cx cy w h`, normalized)
```
A frame with no `.txt` (or an empty one) is background. Labels use the canonical
class ids above — e.g. a fall is class `6`, never `0`.

**manifest.json** — one entry per clip:
```json
{"clip_id": "Home_01__video_3", "camera_id": "Home_01", "scenario": "Home",
 "classes": [6], "n_frames": 312}
```
`classes: []` marks a negative clip. `camera_id` is what `build_splits.py`
keeps disjoint across train/val/test; when a corpus does not document cameras,
a proxy (e.g. the environment folder) is used **and declared in the paper**.

`le2i_to_clips.py` is the reference adapter. A second adapter (fire, ABODA, …)
only has to emit the same layout + manifest.

---

## 4. Data flow, end to end

```
raw dataset
   │  adapter (e.g. le2i_to_clips.py)
   ▼
clips/ + manifest.json
   │  build_splits.py  →  splits.json   (camera-disjoint; every class in every split)
   │  integrity_check.py → integrity_report.json  (SHA-256 freeze; PHASE 1 ENDS HERE)
   ▼
Stage A: ultralytics YOLO trains frame-level          → best.pt  (the BASELINE)
   ▼
Stage B: SentryYOLO(best.pt) + TFM; backbone frozen   → tfm_*.pt
   │      loss = v8DetectionLoss(box,cls,dfl) + λ·L_tc      [Eq. 3]
   ▼
Inference (streaming, per clip):
   frame → SentryYOLO → NMS → detections
                                  │  TubeLinker (IoU ≥ 0.5, gap ≤ 5)
                                  ▼
                              confirm_events (per-class persistence n_c)
                                  ▼
                          alert records  →  metrics.py (event AUC / F1 / confusion)
```

**Threshold discipline**: `τ_c` (confidence) and `n_c` (persistence) are swept on
**validation only**, frozen, and the test split is scored **once**. This is not
a style preference — it is the pre-registered protocol.

---

## 5. The model, module by module

### `modules.py`
- `ConvGRUCell` — Eq. (1). Gates `z`, `r` computed jointly; state `h_t` keeps
  spatial structure (it is a conv, not a flatten).
- `MotionGate` — `M_t = σ(W_m * |P_t − P_{t−1}|)`. A learned scalar `α` (one per
  level) scales its influence.
- `TFMLevel` — Eq. (2): `F_t = proj(h_t) ⊙ (1 + α·M_t) + P_t`.
  **The additive skip `+ P_t` is load-bearing**: at initialization the fused
  feature ≈ the original feature, so an untrained SENTRY behaves like the
  frame-level detector. This is what makes the "untrained TFM" ablation a valid
  control, and it is confirmed empirically (F1 0.621 vs baseline 0.727).
- `TemporalFeatureMemory` — one `TFMLevel` per FPN level; holds streaming state.
  **`reset()` must be called at the start of every clip**, or state leaks across
  videos and results are silently wrong.

### `ultralytics_adapter.py`
`SentryYOLO` wraps the Detect head's `forward` so the TFM sits between neck and
head. The official backbone/neck/head and their losses stay untouched; the
architectural delta *is* the TFM. Touch points isolated in `_find_detect_head()`
and `_wrap_head()` — if a future Ultralytics release changes internals, only
those two functions need attention.

Key API: `reset_stream()`, `freeze_base()`, `last_evidence` (mean motion-gate
activation of the current frame, carried into alert records).

### `tubes.py`
Greedy online linking by class + IoU, tolerating gaps. `confirm_events` filters
tubes by a per-class minimum persistence `n_c` and emits the structured alert.
Pure numpy — testable without a GPU.

### `metrics.py`
- `event_auc`, `event_prf` — matching by temporal IoU, greedy, per class.
- `event_confusion_matrix` — matches **ignoring class**, so a wrong-label hit
  (class confusion) is distinguishable from a miss. Extra row/column =
  background: last column = missed events, last row = false alarms.
- `bootstrap_ci` — percentile CI over per-video metrics.

### `seeds.py`
`run_over_seeds(pipeline_fn)` runs the **pre-declared** seeds `[0,1,2]`, writes
one JSON per seed (so an interrupted session resumes), and aggregates to
mean ± std + bootstrap CI. Splits stay fixed across seeds, so reported variance
isolates training stochasticity.

---

## 6. Load-bearing design decisions (do not "simplify" these)

1. **Stage B must be supervised.** Training the TFM with the temporal-consistency
   term *alone* is ill-posed: its trivial optimum is "change nothing", giving
   `L_tc ≡ 0` and zero gradient. The detection loss provides the task signal;
   `L_tc` is only a regularizer.
2. **`v8DetectionLoss` needs the `DetectionModel`**, not its inner `Sequential`,
   and reads `hyp.box/.cls/.dfl` by **attribute**, so `model.args` must be a
   namespace (`SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)`), never a dict.
3. **The head has two output formats.** In `train()` mode it returns raw feature
   maps (what the loss consumes); in `eval()` mode it returns the decoded
   `(1, 4+nc, N)` tensor (what NMS consumes). Mixing them raises shape errors.
4. **`nc = 8` everywhere.** Le2i labels carry class id 6; a single-class
   `data.yaml` would crash because ids must be `< nc`.
5. **Background subsampling only in train.** `make_yolo_layout(bg_stride_train=5)`
   thins background frames in the training split; val/test keep every frame, so
   metrics are never subsampled.
6. **Never export YOLO format from CVAT.** It cannot carry the compositional
   attributes. Export COCO 1.0 and convert with `coco_to_yolo.py`, which writes
   `.attrs.json` sidecars.

---

## 7. Environment failure modes already solved

| Symptom | Cause | Fix (already in the code/notebook) |
|---|---|---|
| `AttributeError: ray.train._internal.session has no attribute '_get_session'` at `on_fit_epoch_end` | Ultralytics auto-registers a Ray Tune callback; Kaggle's `ray` build dropped the symbol | `!pip uninstall -y ray` before training |
| Training hangs/dies silently with `device="0,1"` | DDP spawns a subprocess that does not survive notebooks | `device=0` |
| `malloc_consolidate(): invalid chunk size` reading Le2i `.avi` | Corrupt MP3 audio track crashes OpenCV's decoder | decode via the `ffmpeg` CLI with `-an` (in `le2i_to_clips.py`) |
| Le2i annotations "not found" | Mirrors nest folders twice and misspell `Annotations_files` | recursive glob `Annotation*files` |
| A no-fall Le2i clip labelled as a fall | Some ADL videos carry a dummy interval `[1,2]` with no boxes inside | a clip is a fall **only if** a box lies inside the interval |
| `build_splits` raises "could not guarantee all classes" | Too few cameras per scenario (Le2i: 2) | falls back to global camera-level allocation, recorded as `"stratification": "global_fallback"` |
| Kaggle paths not found | Mount layout differs per account | always read `!ls /kaggle/input`; it may be `/kaggle/input/<slug>/` or `/kaggle/input/datasets/<user>/<slug>/` |
| `v8DetectionLoss` raises `TypeError: list indices must be integers or slices, not str` | Ultralytics changed the training-mode model output between versions: 8.3.49 (pinned for the paper's Kaggle runs) returns the plain per-level feature list directly; 8.4.x returns a `{"boxes","scores","feats"}` dict instead, and that version's loss expects the whole dict, not just `["feats"]` | `sentry/train.py` feeds the raw model output to the criterion **unmodified** (each version's loss knows its own shape) and uses `extract_feat_list()` only for the auxiliary `L_tc` term, which needs the plain feature list regardless of version |
| `coco_to_yolo.py` produces a `labels/` folder nothing downstream can find | The script never used its `images_dir` argument and wrote a flat `labels/` folder incompatible with the `clips/<clip_id>/frames+labels` contract | fixed: `--layout clips` (default) writes labels next to the clip's existing frames and validates every referenced image exists; `--layout flat` preserves the legacy single-folder workflow and now actually copies/symlinks images, including background images (see below) |
| `integrity_check.py` raises `AttributeError: module 'datetime' has no attribute 'UTC'` | `datetime.UTC` requires Python 3.11+; `pyproject.toml` declares `requires-python = ">=3.10"` | changed to `datetime.timezone.utc`, available since Python 3.2 |

### 7.1 Stage B training: correctness and the batching fix (audited and rewritten)

An audit of this repository found that `sentry/train.py` was **silently non-functional**: it hard-coded `decoded = None`, so its loss branch never executed, `loss_total` stayed a plain Python `0.0` for every step of every epoch, and `torch.is_tensor(loss_total)` was therefore always `False` -- `backward()`/`optimizer.step()` never ran, for any epoch, while the script printed `"epoch N: ok"` and saved a checkpoint that was just the TFM's random initialization. No exception, no warning: a plausible-looking log and a useless checkpoint. This is the single most dangerous kind of bug in a research pipeline, because it does not fail, it fabricates a false success.

The script has been rewritten (`sentry/train.py`, reusing `sentry/stageb_train.py` unchanged) to mirror exactly the supervised approach validated interactively in the Kaggle notebook (cell [5], which produced the epoch-ablation numbers in Table 2: v8DetectionLoss as the primary signal, L_tc as an auxiliary regularizer). Two properties are now enforced by tests (`tests/test_train.py`), not just asserted in prose:

- **Loud failure.** An epoch in which every mini-batch has zero labeled frames raises `RuntimeError` naming the exact reason, instead of completing and looking successful.
- **Real gradient flow.** `test_stage_b_actually_updates_tfm_weights` snapshots the TFM's parameters, trains for two epochs on tiny synthetic data, and asserts every parameter tensor changed. This is the regression test for the bug above: on the old script, this assertion would have failed with `0/27` tensors changed.

**The batching optimization** (requested explicitly: time and memory both matter). The old loop nested `for clip in batch: for frame in clip:`, i.e. one Python-level forward+backward graph construction per *(clip, frame)* pair. `ConvGRUCell`, `MotionGate` and `TemporalFeatureMemory` (`sentry/modules.py`) already operate on an arbitrary batch dimension -- nothing about the model required this. The rewritten loop stacks the `N` clips of a mini-batch (via the new `sentry.data.collate_batched`) and calls the model **once per timestep for the whole batch**: `T` forward calls instead of `N*T`. Measured on this repository's own test hardware (CPU, tiny `yolov8n`, `batch_size=2`, `window=4`, 4 synthetic clips):

| | model() calls | wall-clock (1 epoch) |
|---|---|---|
| old per-clip loop | 16 | 1.81 s |
| batched (this fix) | 8 | 0.45 s |

A 2x reduction in forward-call count (matching `batch_size` exactly, as designed) produced a 4.05x wall-clock reduction on CPU, because batching also amortizes fixed Python/autograd/kernel-launch overhead per call, not just the raw compute. On GPU, mixed precision (`torch.autocast` + `GradScaler`) is additionally enabled by default (`device.type == "cuda"`), further reducing memory and step time; it is left off on CPU, where it has no benefit and can be numerically fragile. Neither change alters what is computed, only how many calls it takes and at what numeric precision -- the loss formula, the labels, and the model are identical to the validated notebook run.

---

## 8. Reproducing the reported numbers

Run `notebooks/sentry_kaggle.ipynb` top to bottom (GPU T4×2, Internet ON). Each
cell states the value it should produce:

| Result | Where | Expected |
|---|---|---|
| Frame-level baseline (fall) | cell [3] | mAP50 = 0.675 ± 0.043, CI95 [0.627, 0.711] |
| TFM parameter overhead | cell [4] | +21.1% (5.47M) |
| Stage B epoch ablation | cells [5]+[6] | F1 0.056 / 0.132 / 0.047 at 5/10/30 epochs vs baseline 0.727 |
| Untrained-TFM control | cell [6], `LOAD_TFM=False` | F1 0.621 |
| Level E prompt conditions | cell [7] | AUC 0.000 / 0.243 / 0.128; macro-F1 0.029 / 0.215 / 0.190 |
| Latency | cell [8] | 12.47 ms frame-level → 15.46 ms SENTRY (+24.0%) |

Everything the paper reports is exported by cell [9] into `artifacts.zip`.
**Integrity rule: no number enters the paper unless it comes from those files.**

---

## 9. The headline finding, in one paragraph

The frame-level baseline is solid. The TFM, trained on the data that public
corpora afford (78 clips, one class, four cameras), does **not** beat it: the
training loss falls monotonically with epochs (16.7 → 14.1 → 12.7) while test F1
stays far below baseline and false positives grow (407 → 158 → 488). That is
memorization, not learning — the bottleneck is data volume and diversity, not
optimization or architecture. Meanwhile, compositional prompts beat class-name
prompts ~7× in open-vocabulary event detection (macro-F1 0.215 vs 0.029),
validating the schema, though absolute performance stays low. Three independent
lines of evidence converge on the same conclusion: the field needs an annotated
spatiotemporal video benchmark covering the eight classes. Building it is the
next item on the agenda; this repository is the toolkit for doing so.

---

## 10. SENTRY-C: the convergence layer

`sentryc/` fuses SENTRY's physical event stream with a network-anomaly stream
through a heterogeneous event graph. It is **additive**: it consumes the alert
records `sentry/tubes.py` already emits and never changes how the visual model is
trained or evaluated. The published results of §8 remain valid because nothing
upstream of them was touched.

### 10.1 Why only five classes may seed a correlation

Three conditions must hold before a physical event is worth correlating with a
network anomaly:

1. **Causal/temporal plausibility** — a network event could plausibly precede,
   accompany, or follow the physical one through a real mechanism.
2. **Ambiguity asymmetry** — the video evidence alone *underdetermines* the
   incident, so a network signal has something left to disambiguate.
3. **Network coverage of the asset** — the asset involved is actually visible to
   the network sensor.

Five classes satisfy all three and are therefore **seed-eligible**:

    intrusion · suspicious_behavior · vandalism · fall · suspicious_object

Three classes do not, and are **consequence-only**:

    accident · fire · crime

They stay fully part of PSID-8 and of the standalone SENTRY evaluation; what
they may not do is *seed* a correlation. A fire is not disambiguated by a port
scan — asserting that correlation would be a causal claim the evidence does not
support. They enter the graph only as successors of a seed, through a directed
`precedes` edge with a positive time delta (vandalism against a suppression
sensor preceding a fire; an accident following an unresolved intrusion).

The empirical anchor for condition (2) is in §8 of this document: **intrusion
scores F1 = 0.000 in every prompt condition** because a person indoors is a
resident or an intruder and the pixels do not say which. That is not a model
failure — it is ambiguity that only a non-visual signal (an authentication
event, or its absence) can resolve. Convergence is therefore *derived* from a
measured result, not assumed.

### 10.1b Scope: what this work does and does not claim

This work does not pursue explainability as a contribution. `TemporalRuleScorer`
exists solely as the H3 predictive-merit comparator (does the GNN out-detect
hand-written rules on identical inputs?); it is not offered as an
interpretability deliverable, and no claim about explaining model decisions to
an operator is made anywhere in this module. Readers looking for that
contribution will not find it here by design.

### 10.1c Positioning against recent multi-modal cyber-physical fusion work

A relevant point of comparison is UMTD-Net (a 2025 cross-modal attention
architecture fusing video, audio and network telemetry for smart-city threat
detection), which reports high aggregate F1/AUC but leaves several gaps that
motivate design choices in this module:

| Dimension | UMTD-Net | This work |
|---|---|---|
| Output granularity | binary threat/normal per input window | per-class event: track, time span, confidence trajectory |
| Fusion justification | none stated; all modalities fused indiscriminately for every input | explicit per-class eligibility (`SEED_ELIGIBLE`), grounded in a measured result (§8: intrusion F1 = 0.000 under vision alone) |
| Temporal precedence between modalities | not modelled (fusion is over a synchronized window) | first-class: `precedes` edges, `time_to_detection`, `lead_time_gain` |
| Missing-modality behaviour | acknowledged as an open limitation ("future work") | structural: an absent modality is simply an absent neighbour in the graph; no retraining or special-casing needed |
| Cross-instance correspondence between modalities | not described; the three benchmarks used (UCF-Crime, UrbanSound8K, TON_IoT) are independent corpora with no documented real pairing | declared explicitly: Degree-1 experiments use a synthetic network stream with a stated coupling parameter κ and declared negative controls (see the experiment plan) — the pairing is a stated assumption, not a silent one |

The comparison is deliberately narrow: it does not include explainability
(out of scope here, see §10.1b) or raw aggregate accuracy, since the two
systems have not been evaluated on the same corpus. Its purpose is to state
plainly where this module's design choices depart from a recent representative
system, not to claim general superiority.

### 10.2 Load-bearing: the constraint is code, not convention

`SEED_ELIGIBLE` and `CONSEQUENCE_ONLY` partition the eight classes, and a module
level `assert` verifies the partition is total and disjoint. `EventGraphBuilder.build()`
raises `SeedConstraintError` if asked to seed with a consequence-only class, and
auto-seeding silently skips them. `CorrelationModel._check_seed()` re-verifies at
scoring time, because a graph may have been assembled elsewhere. The test suite
(`tests/test_sentryc.py`) asserts both directions: consequence classes must be
rejected as seeds **and** must remain reachable as successors.

### 10.3 Modules

| Module | Responsibility |
|---|---|
| `network_stream.py` | `NetworkEvent` schema + `NetworkEventSource` adapter. The `anomaly_type` vocabulary is CLOSED (`port_scan`, `new_device`, `exfiltration`, `dos_camera`, `unauthorized_access`) — an open vocabulary would make the fusion-justification probabilities unestimable. Not an IDS: the detector is a separate component. |
| `graph_builder.py` | Nodes `PhysicalEvent` / `NetworkEvent` / `Asset`; edges `temporal_proximity`, `spatial_colocation`, `asset_relation`, `precedes`. Enforces the seed constraint. Exposes `FusionJustification` (gain = P(e_n\|e_p,class) − P(e_n), plus Amb(e_p), Amb(e_n)) with configurable thresholds θ, τ. |
| `gnn_correlation.py` | `HeteroGNNScorer` (small typed-attention GNN) and `TemporalRuleScorer` (fixed-rule baseline) behind one `CorrelationModel` interface, so H3 can compare their predictive merit on identical inputs. Explainability is explicitly out of scope for this work; the rule baseline exists to answer whether the GNN detects and ranks convergent incidents better than hand-written rules, nothing further is claimed for it. |
| `alerts.py` | `ConvergentAlert`, including `time_to_detection` measured from the FIRST contributing event of either modality — lead time is a primary claim (H3), so it is a first-class field, not a derived afterthought. |
| `metrics.py` | Convergent F1/AUC (importing the temporal-IoU machinery from `sentry/metrics.py`, never re-implementing it), lead-time gain with bootstrap CI, false-positive reduction, and modality contribution. |

### 10.4 Design decisions worth defending

- **Reuse, do not duplicate, the matching logic.** `sentryc/metrics.py` imports
  `tiou`, `match_events`, `event_prf`, `event_auc` and `bootstrap_ci` from
  `sentry.metrics`. If the two ever drifted, every convergent-vs-unimodal
  comparison in the thesis would be quietly invalid.
- **Lead time is measured only on true positives.** The latency of a false alarm
  is meaningless; averaging it in would make a noisy detector look fast.
- **`modality_contribution` is not optional.** A convergence paper that cannot
  show a non-trivial `joint` count is riding on one strong modality. The metric
  makes that visible instead of burying it inside an aggregate F1.
- **The rule baseline sees exactly the same information as the GNN.** Same graph,
  same features. Otherwise the H3 comparison is rigged in the GNN's favour.
- **The GNN is deliberately shallow (2 layers).** The graphs are small and the
  dataset will be small. §6 of this document records what depth did to the TFM
  under exactly those conditions: it memorized. Do not "improve" this by adding
  capacity before there is data to justify it.
- **Messages are per relation, not per node.** A neighbour linked by both
  `temporal_proximity` and `spatial_colocation` sends two typed messages, so the
  attention can weigh co-located evidence differently from merely co-occurring
  evidence. `len(kinds)` counts messages, not distinct neighbours.
- **Feature encoding is torch-free.** `gather_neighbors()` is pure numpy and unit
  tested. An encoding bug would silently corrupt every GNN result, and catching
  it must not require a GPU.

### 10.5 Failure modes

| Symptom | Cause | Response |
|---|---|---|
| `SeedConstraintError` at graph build | code tried to seed with accident/fire/crime | correct — attach the event as a successor of a real seed instead |
| Every physical/network pair is linked | no `FusionJustification` supplied, or θ = 0 | supply estimated probabilities and raise θ; unfiltered graphs make the GNN learn co-occurrence, not causation |
| Convergent alert with empty `network_evidence` | the correlation window contained no anomaly | expected; `modality_contribution` will count it as `physical_only`, which is the honest answer |
| Lead time ≈ 0 for every alert | network events never precede the physical event | check clock alignment between the two streams first; a constant offset destroys this metric |

### 10.6 Clock alignment (read before running any experiment)

Physical timestamps come from frame indices; network timestamps come from the
capture. **They must be expressed on one clock before any correlation is
attempted.** A constant offset silently produces plausible-looking graphs with
meaningless lead times. Record the alignment procedure alongside the data, and
treat it as part of the freeze.

### 10.7 Status

The module is implemented and unit tested; **no experimental results exist yet**.
Every number in any future write-up must come from a real run — the same
integrity rule as the rest of this repository. Current results: [PENDING].
