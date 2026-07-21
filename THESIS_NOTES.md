# Thesis Notes — Code Audit Session (keep updated)

This file exists so nothing discussed during the engineering audit gets lost
before the thesis is written. Add to it; do not let it go stale.

---

## 1. Scope of impact: does the audit invalidate Article 1's results?

**No.** This needs to be stated clearly and defended if a reviewer asks the
same question.

The critical bug found in this audit (`sentry/train.py` silently training
nothing) lived in a **standalone CLI script**. The numbers reported in the
paper (Table 2, the epoch ablation at 5/10/30 epochs) came from a **different,
already-working code path**: the Kaggle notebook's Stage B cell (`[5]`), which
had its own inline training loop, independent of `sentry/train.py`, and was
already using the correct supervised formulation (`v8DetectionLoss` +
auxiliary `L_tc`).

The evidence that this inline loop trained for real, not the reasoning alone:
the training loss (`det`) fell **monotonically** as epochs increased (16.7 →
14.1 → 12.7 at 5/10/30 epochs). That pattern cannot occur if
`backward()`/`optimizer.step()` never ran — it is the exact signature the
broken standalone script could **not** have produced (there, the loss would
have stayed at its initial value forever, as reproduced in
`tests/test_train.py`).

**Conclusion:** the reported overfitting (training loss falls while test F1
degrades) is a real finding about data scarcity, not an artifact of broken
code. Article 1's diagnosis stands. The bug is a latent defect in an
alternative entry point that was never used to produce a published number —
important to fix so nobody trips over it later, but it does not change what
has already been reported.

A second, narrower point: the `coco_to_yolo.py` layout fix also does not
affect Article 1. The Le2i pipeline (`le2i_to_clips.py`) writes directly to the
canonical `clips/<id>/frames+labels` layout without ever going through
`coco_to_yolo.py`. That fix matters for the **future** PSID-8 video benchmark
(CVAT → COCO → `coco_to_yolo.py`), not for anything already published.

---

## 2. YOLO / Ultralytics version compatibility — what is actually verified

Two independent axes of variation; only one is fixed and tested.

**Axis A — Ultralytics package version** (e.g. 8.3.49 pinned on Kaggle vs.
8.4.103 available in the audit sandbox). A real, verified incompatibility was
found here: the model's raw training-mode output changed shape between these
versions (a plain feature list vs. a `{"boxes","scores","feats"}` dict), and
`v8DetectionLoss`'s expected input shape changed accordingly. Fixed:
`sentry/train.py` now feeds the raw output to the criterion **unmodified**
(each version's loss parses its own shape) and uses `extract_feat_list()`
only for the auxiliary `L_tc` term. Covered by
`tests/test_train.py::test_extract_feat_list_handles_both_ultralytics_shapes`.

**Axis B — YOLO architecture generation** (v8, v9, v10, v11, v12/13, YOLO26).
**Not fully tested.** What is known, checked against Ultralytics' own
documentation as of mid-2026:

| Generation | Head | DFL | Compatible with current `build_criterion` (v8DetectionLoss)? |
|---|---|---|---|
| YOLOv8 (tested: `yolov8n`) | anchor-free `Detect`, decoupled | yes, reg_max=16 | **Verified working** (this audit) |
| YOLO11 | same `Detect` design as v8 (only backbone/neck blocks differ: C3k2 vs C2f) | yes, reg_max=16 | Likely compatible, **not tested** |
| YOLO26 | dual head (`Detect`, one-to-one NMS-free + one-to-many), "Progressive Loss"/`E2ELoss` | **removed entirely** | **Not compatible as-is.** `v8DetectionLoss` computes a DFL term that has no counterpart in YOLO26's output. Using YOLO26 as the base detector requires locating and using the loss class Ultralytics ships for it (name unconfirmed at time of writing) and adapting `build_criterion` accordingly. |

**Action before using anything other than YOLOv8/YOLO11 as the base
detector:** repeat the same live-verification procedure used in this audit
(instantiate the model, inspect `model(x)` in train mode, confirm the
criterion's expected input shape) before trusting any reported number.

---

## 3. Files touched in this audit (for merging into the working copy)

**Modified** (replace these exact files):
`psid8/scripts/coco_to_yolo.py`, `psid8/scripts/integrity_check.py`,
`sentry/aggregate.py`, `sentry/train.py`, `sentry/data.py`, `pyproject.toml`,
`sentry/__init__.py` (version bump only), `sentryc/graph_builder.py`,
`sentryc/metrics.py`, `sentryc/gnn_correlation.py`, `sentry/plots.py`,
`psid8/scripts/dataset_stats.py`, `tests/run_tests.py`, `tests/test_sentryc.py`,
`ARCHITECTURE.md`, `CHANGELOG.md`.

**Added** (new files): `tests/test_psid8_scripts.py`, `tests/test_train.py`.

**Untouched, confirmed by hash before/after**: `sentry/modules.py`,
`sentry/ultralytics_adapter.py`, `sentry/tubes.py`, `sentry/metrics.py`,
`sentry/seeds.py`, `sentry/stageb_train.py`,
`psid8/scripts/le2i_to_clips.py`, `psid8/scripts/build_splits.py`,
`psid8/scripts/agreement.py`, `psid8/scripts/curate_clips.py`,
`sentryc/network_stream.py`, `sentryc/alerts.py`.

**The Kaggle notebook (`notebooks/sentry_kaggle.ipynb`) was NOT modified in
this audit.** Its Stage B cell `[5]` has its own inline copy of the training
loop (predates and is independent of `sentry/train.py`); it is what produced
the published numbers and remains valid. It does **not** yet benefit from:
(a) the Ultralytics-version-shape tolerance (probably moot on Kaggle's pinned
8.3.49, where the plain-list shape holds), and (b) the batched-training
speedup (still material — see below). Pending action: rewrite cell `[5]` to
import `collate_batched`/`run_stage_b` from `sentry.train` instead of
reimplementing the loop inline, once validated on Kaggle hardware.

**Practical merge instructions:** the safest approach is to overwrite the
whole local clone with the contents of the delivered zip, then use `git diff`
against the working copy before committing, so the exact set of changes is
visible and reviewable rather than merged blindly.

---

## 4. Time/memory experiment: methodology to reproduce at real scale

The numbers already measured (2x fewer `model()` calls, 4.05x wall-clock
reduction for one epoch) were obtained on **toy data on a CPU sandbox**
(tiny `yolov8n`, 64x64 images, 4 synthetic clips, `batch_size=2`) — real,
reproducible, but **not** the numbers to cite in the paper. They exist only to
prove the mechanism (fewer forward calls -> less wall-clock) is real, not just
theoretical.

**To get paper-citable numbers**, repeat the same comparison on Kaggle, at the
paper's actual scale (Le2i clips, `imgsz=640`, `yolov8m`, the real train split,
`window=8`). Suggested protocol for Section V (or a new subsection on
implementation/computational cost):

1. Run one Stage B epoch with the **old** per-clip loop (available in git
   history / this document's file list) and record wall-clock + peak GPU
   memory (`torch.cuda.max_memory_allocated()`).
2. Run one Stage B epoch with the **new** batched loop
   (`sentry.train.run_stage_b` + `collate_batched`), same data, same seed,
   same `batch_size`. Record the same two numbers.
3. Report: forward-call count (exact, from the design: `T` vs `N*T`),
   wall-clock ratio, peak-memory ratio. State the batch size used, since the
   call-count reduction scales with it.
4. Confirm numerical equivalence: the two loops must produce statistically
   indistinguishable `det`/`tc` trajectories over an epoch (same loss formula,
   same labels, same model) — report this as a sanity check, since a
   legitimate speed claim requires showing correctness was preserved.

`tests/test_train.py::test_batched_loop_uses_fewer_model_calls` is the
template for step 1-2's call-counting instrumentation; adapt it to real data
by pointing `VideoClipDataset` at the real `clips_dir`/`splits.json` instead of
the synthetic fixture.

Suggested paper framing (Implementation / Computational Cost subsection):
*"Stage B batches clips of a mini-batch across the batch dimension at each
timestep rather than processing one clip at a time, reducing the number of
forward passes from N·T to T per mini-batch (N = batch size, T = window
length); at batch size [X] on [hardware], this reduced wall-clock time by
[Y]x and peak memory by [Z]% for one epoch, with no change to the loss
formulation or the reported metrics."*

---

## 5. Synthetic data: what exists and what does not

**What exists (code-correctness testing only, not research data):**
`tests/test_train.py::_build_synthetic_clips` generates uniform random-noise
images and one arbitrary fixed bounding box per clip. Its only purpose is to
prove gradients flow and the batching mechanism works; it has **no resemblance
to real surveillance footage** and must not be used as a basis for any
reported experiment or as a seed for data augmentation.

**What does NOT exist yet (research-grade synthetic data):** the
channel-attack injection generator discussed for the SENTRY-C /
physical-cyber convergence line (`sentryc/simulate.py` in the plan, not yet
implemented): extracting real clips from UCF-Crime/Le2i and injecting
declared, parameterized anomalies (freeze, replay, splice, dropout, blackout,
timestamp jitter) with a stated coupling parameter and explicit hard
negatives (packet loss, compression artifacts, genuinely static night scenes).
This is the next code artifact to build for that research line, not something
already delivered.

---

## 6. Open items / next decisions

- [ ] Decide whether to update the Kaggle notebook's cell `[5]` to use the
      batched loop (`collate_batched` + `run_stage_b`), and validate on real
      Kaggle hardware before trusting any new numbers from it.
- [ ] If a YOLO generation other than v8/11 is considered, re-verify the
      criterion construction first (see section 2's table).
- [ ] Re-run the batching comparison at paper scale on Kaggle for citable
      numbers (section 4's protocol).
- [ ] Decide on second annotator and begin Phase 0 calibration for the PSID-8
      video benchmark (unrelated to this audit, still the standing blocker for
      Article 2).
- [ ] `sentryc/simulate.py` (channel-attack synthetic generator) remains
      unbuilt; build only after the PSID-8 benchmark work is underway, per the
      agreed article ordering (Article 2 -> 3 -> 4).
