"""SENTRY event metrics (paper Table IV-C) - pure numpy, offline-testable.

- tIoU: intersection/union of temporal event intervals
- predicted-vs-GT event matching: same class, tIoU >= thr, greedy
- event-level AUC: ROC over predicted-event scores (positives = matched with
  GT; negatives = unmatched) + missed GT events as FN
- event_prf: event-level P/R/F1 per class
Spatial metrics (mAP@50, mAP@50:95) come from the detector backend
(Ultralytics val) and are not reimplemented here.
"""
from __future__ import annotations
import numpy as np


def tiou(a_start, a_end, b_start, b_end) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = (a_end - a_start + 1) + (b_end - b_start + 1) - inter
    return inter / union if union > 0 else 0.0


def match_events(preds: list[dict], gts: list[dict], tiou_thr: float = 0.5):
    """preds/gts: dicts with class_id, t_start, t_end (+ confidence in preds).
    Returns (pairs [(pi, gi, tiou)], unmatched_preds, unmatched_gts)."""
    cands = []
    for pi, p in enumerate(preds):
        for gi, g in enumerate(gts):
            if p["class_id"] != g["class_id"]:
                continue
            v = tiou(p["t_start"], p["t_end"], g["t_start"], g["t_end"])
            if v >= tiou_thr:
                cands.append((v, pi, gi))
    used_p, used_g, pairs = set(), set(), []
    for v, pi, gi in sorted(cands, reverse=True):
        if pi in used_p or gi in used_g:
            continue
        pairs.append((pi, gi, v)); used_p.add(pi); used_g.add(gi)
    fp = [i for i in range(len(preds)) if i not in used_p]
    fn = [i for i in range(len(gts)) if i not in used_g]
    return pairs, fp, fn


def event_auc(preds: list[dict], gts: list[dict], tiou_thr: float = 0.5) -> float:
    """Event-level ROC AUC. TP = matched pred; FP = unmatched pred;
    FNs enter as positives with score 0 (missed events)."""
    pairs, fp, fn = match_events(preds, gts, tiou_thr)
    y, s = [], []
    for pi, _, _ in pairs:
        y.append(1); s.append(preds[pi]["confidence"])
    for pi in fp:
        y.append(0); s.append(preds[pi]["confidence"])
    for _ in fn:
        y.append(1); s.append(0.0)
    y, s = np.array(y), np.array(s)
    if y.min() == y.max():
        return float("nan")  # undefined without both classes
    order = np.argsort(-s)
    y = y[order]
    tps = np.cumsum(y); fps = np.cumsum(1 - y)
    tpr = tps / tps[-1]; fpr = fps / fps[-1]
    return float(np.trapezoid(tpr, fpr))


def event_prf(preds, gts, tiou_thr: float = 0.5, n_classes: int = 8):
    out = {}
    for c in range(n_classes):
        pc = [p for p in preds if p["class_id"] == c]
        gc = [g for g in gts if g["class_id"] == c]
        pairs, fp, fn = match_events(pc, gc, tiou_thr)
        tp = len(pairs)
        prec = tp / (tp + len(fp)) if tp + len(fp) else 0.0
        rec = tp / (tp + len(fn)) if tp + len(fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[c] = {"P": prec, "R": rec, "F1": f1, "tp": tp, "fp": len(fp), "fn": len(fn)}
    return out


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI over per-video metrics (Sec. IV-C)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def streams_per_gpu(fps_measured: float, stream_fps: float = 25.0) -> int:
    """Operational reading of Sec. V-F."""
    return int(fps_measured // stream_fps)
