#!/usr/bin/env python3
"""SENTRY Stage B training: supervised temporal fine-tuning of the TFM.

Prerequisite (Stage A, frame-level, via the ultralytics CLI on Kaggle):
  yolo detect train data=data.yaml model=yolov8m.pt imgsz=640 epochs=100 seed=0

Rationale (documented after two earlier failed attempts): training the TFM
with the temporal-consistency term ALONE is ill-posed - its trivial optimum is
"change nothing", which pins the network at initialization (L_tc == 0 with
zero gradient). This script is driven by the detector's own supervised loss
(v8DetectionLoss: box + cls + dfl) over the labeled frames of each clip, with
L_tc as an auxiliary regularizer, exactly as validated interactively in the
Kaggle notebook (cell [5]), which produced the epoch-ablation numbers reported
in the paper (Table 2).

Two changes relative to the earlier skeleton version of this file, both
verified by the tests in tests/test_train.py:

1. CORRECTNESS. The earlier skeleton decoded the raw model output into a
   hardcoded `None` and never actually built a loss batch, so `loss_total`
   stayed a plain Python float `0.0` for every step of every epoch;
   `torch.is_tensor(loss_total)` was therefore always False and
   `backward()`/`optimizer.step()` NEVER ran, while the script still printed
   "epoch N: ok" and saved a checkpoint (which was just the TFM's random
   initialization). That failure was silent: no exception, no warning, a
   plausible-looking log and a useless checkpoint. This version computes a
   real loss and raises a clear, loud error if an entire epoch produces no
   optimizer step, instead of pretending training happened.

2. PERFORMANCE (time and memory). The earlier loop nested "for clip in batch:
   for frame in clip", i.e. one Python-level forward+backward graph
   construction per (clip, frame) pair. `ConvGRUCell`, `MotionGate` and
   `TemporalFeatureMemory` (sentry/modules.py) already operate on an arbitrary
   batch dimension, so this version stacks the N clips of a mini-batch and
   calls the model ONCE PER TIMESTEP for the whole batch: T forward calls
   instead of N*T. On CUDA, mixed precision (autocast + GradScaler) is enabled
   automatically, reducing both memory and step time further; it is left off
   on CPU, where it offers no benefit and can be numerically fragile.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from types import SimpleNamespace

import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def build_criterion(det_model, box=7.5, cls=0.5, dfl=1.5):
    """Construct v8DetectionLoss with the hyperparameters it reads.

    v8DetectionLoss needs the DetectionModel itself (not its inner Sequential)
    and reads `hyp.box` / `hyp.cls` / `hyp.dfl` by ATTRIBUTE access, so
    `model.args` must be a namespace, not a dict. See ARCHITECTURE.md section 6
    for why this specific construction is required.
    """
    from ultralytics.utils.loss import v8DetectionLoss
    det_model.args = SimpleNamespace(box=box, cls=cls, dfl=dfl)
    criterion = v8DetectionLoss(det_model)
    if isinstance(getattr(criterion, "hyp", None), dict):
        criterion.hyp = SimpleNamespace(**criterion.hyp)
    return criterion


def extract_feat_list(raw):
    """Extract the plain list of per-level feature-map tensors from a model's
    raw training-mode output, regardless of which Ultralytics generation
    produced it.

    Ultralytics changed this shape between versions actually observed in this
    project: the version pinned on Kaggle for the paper's runs (8.3.49)
    returns the feature-map list directly; a newer version (8.4.x, seen in
    this repository's CI/test environment) returns a dict
    `{"boxes":..., "scores":..., "feats": [P3, P4, P5]}` instead, and its
    `v8DetectionLoss` expects to be handed that whole dict (not just the
    feature list). This helper only concerns the AUXILIARY L_tc term, which
    always needs the plain per-level feature list; the detection-loss call
    itself is fed `raw` unmodified in `run_stage_b`, so each Ultralytics
    version gets the input shape its own loss implementation expects.
    """
    if isinstance(raw, dict):
        if "feats" in raw:
            return list(raw["feats"])
        raise KeyError(
            f"model output is a dict without a 'feats' key (keys: {list(raw)}); "
            "extract_feat_list needs updating for this Ultralytics version."
        )
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]


def run_stage_b(model, criterion, dataloader, epochs: int, lr: float,
                lambda_tc: float, device, use_amp: bool = False,
                log_every: int = 50, log_fn=print):
    """The batched Stage B training loop, torch-dependent but otherwise
    identical in structure regardless of GPU/CPU (AMP is only used on CUDA).

    `model` is a SentryYOLO instance already in `.train()` mode with the base
    detector frozen (`model.freeze_base()`), so only the TFM's parameters
    receive gradients. `dataloader` must be built with
    `sentry.data.collate_batched`.

    Returns the per-epoch history: list of {"epoch", "det", "tc"} dicts, in
    the same format `sentry.plots.parse_stageb_log` expects, so the same
    plotting utilities apply to a real run's output as to a pasted log.
    """
    import torch
    from sentry.stageb_train import build_loss_batch, temporal_consistency_feats

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    try:
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    except TypeError:
        # older torch: torch.amp.GradScaler / torch.cuda.amp.GradScaler took no
        # positional device argument
        scaler = torch.amp.GradScaler(enabled=use_amp)
    history = []

    for epoch in range(epochs):
        tot_det, tot_tc, n_steps, n_batches = 0.0, 0.0, 0, 0

        for batch in dataloader:
            images = batch["images"].to(device)          # (N, T, 3, H, W)
            labels_by_t = batch["labels_by_t"]            # [T][N]
            T = images.shape[1]

            model.reset_stream()                          # resets all N states at once
            prev_feats = None
            loss_det = torch.zeros((), device=device)
            loss_tc = torch.zeros((), device=device)
            any_signal = False

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                for t in range(T):
                    x_t = images[:, t]                     # (N, 3, H, W): ONE call for the batch
                    raw = model(x_t)

                    lb = build_loss_batch(labels_by_t[t], images.shape[-1], device)
                    if lb["cls"].numel() > 0:
                        # `raw` is fed to the criterion UNMODIFIED: different
                        # Ultralytics versions expect different shapes here
                        # (plain feature list vs a {"boxes","scores","feats"}
                        # dict), and each version's own v8DetectionLoss knows
                        # how to parse its own model's output.
                        ld, _ = criterion(raw, lb)
                        loss_det = loss_det + ld.sum()
                        any_signal = True

                    feat_list = extract_feat_list(raw)
                    m = model.last_evidence.get("motion_gate_mean", 0.0)
                    tc = temporal_consistency_feats(feat_list, prev_feats, motion=m)
                    if tc is not None:
                        loss_tc = loss_tc + tc
                    prev_feats = [f.detach() for f in feat_list]

                loss = loss_det + lambda_tc * loss_tc

            n_batches += 1
            if any_signal and loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tot_det += float(loss_det.detach())
                tot_tc += float(loss_tc.detach())
                n_steps += 1
            # else: this window carried no labeled frame at any timestep (can
            # happen with event_only=False); correctly skipped, not silently
            # counted as a successful step.

            if n_steps and n_steps % log_every == 0:
                log_fn(f"  ep{epoch} b{n_steps}/{len(dataloader)} | "
                       f"det={tot_det/n_steps:.3f} tc={tot_tc/n_steps:.4f}")

        if n_steps == 0:
            raise RuntimeError(
                f"epoch {epoch}: every batch in this epoch had zero labeled "
                f"frames, so no optimizer step ever ran. This dataset/window "
                f"configuration teaches the TFM nothing; check `event_only` and "
                f"the label files under <clips_dir>/<clip_id>/labels/. "
                f"(This is exactly the failure the earlier version of this "
                f"script suffered from silently: it must not be allowed to "
                f"pass as success.)"
            )

        mean_det, mean_tc = tot_det / n_steps, tot_tc / n_steps
        log_fn(f"epoch {epoch}: det={mean_det:.3f} | tc={mean_tc:.4f} "
              f"({n_steps}/{n_batches} batches had signal)")
        history.append({"epoch": epoch, "det": mean_det, "tc": mean_tc})

    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="Stage A best.pt")
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hidden-ch", type=int, default=128)
    ap.add_argument("--lambdas", type=float, nargs=4, default=[7.5, 0.5, 1.5, 1.0],
                    help="box, cls, dfl, tc coefficients (Eq. 3)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="clips processed together per timestep (see module docstring)")
    ap.add_argument("--event-only", action="store_true", default=True,
                    help="keep only windows with >=1 labeled frame (denser supervision)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="'cuda', 'cpu', or omit to auto-detect")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision even on CUDA")
    ap.add_argument("--out", default="runs/sentry_temporal")
    args = ap.parse_args()
    set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from sentry.ultralytics_adapter import SentryYOLO
    from sentry.data import VideoClipDataset, collate_batched

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    split_ids = json.load(open(args.splits))["splits"]["train"]
    ds = VideoClipDataset(args.clips_dir, split_ids, window=args.window,
                          stride=args.window, event_only=args.event_only)
    if len(ds) == 0:
        raise RuntimeError(
            f"VideoClipDataset found 0 windows under {args.clips_dir} for the "
            f"train split. Check that labels exist at <clip_id>/labels/*.txt "
            f"(see psid8/scripts/coco_to_yolo.py --layout clips)."
        )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=collate_batched, num_workers=args.num_workers,
                    drop_last=True)
    print(f"event windows: {len(ds)} | batches/epoch: {len(dl)} "
          f"(batch_size={args.batch_size}) | device={device} | amp={use_amp}")

    base = YOLO(args.weights)
    model = SentryYOLO(base, hidden_ch=args.hidden_ch).to(device)
    model(torch.zeros(1, 3, 640, 640, device=device))  # materialize the TFM
    model.train()
    model.freeze_base()
    for p in model.tfm.parameters():
        p.requires_grad_(True)

    det_model = model.base
    box, cls, dfl, lambda_tc = args.lambdas
    criterion = build_criterion(det_model, box=box, cls=cls, dfl=dfl)
    print(f"criterion OK | nc={det_model.model[-1].nc} | hyp.box={criterion.hyp.box}")

    history = run_stage_b(model, criterion, dl, epochs=args.epochs, lr=args.lr,
                          lambda_tc=lambda_tc, device=device, use_amp=use_amp)

    os.makedirs(args.out, exist_ok=True)
    torch.save({"tfm": model.tfm.state_dict(), "args": vars(args), "history": history},
              f"{args.out}/tfm_last.pt")
    json.dump(history, open(f"{args.out}/history.json", "w"), indent=1)
    print(f"saved {args.out}/tfm_last.pt and {args.out}/history.json")


if __name__ == "__main__":
    main()
