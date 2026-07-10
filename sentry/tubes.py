"""SENTRY event aggregation (paper Sec. III-E) - pure numpy, offline-testable.

Pipeline: per-frame detections -> greedy tube linking (IoU >= iou_thr,
gap <= max_gap) -> class-specific persistence confirmation (n_c) ->
structured alert record:
  {class; track; t_start-t_end; confidence trajectory; evidence terms}
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / (ua + 1e-9))


@dataclass
class Tube:
    class_id: int
    frames: list = field(default_factory=list)       # time indices t
    boxes: list = field(default_factory=list)        # xyxy per frame
    scores: list = field(default_factory=list)       # confidence per frame
    evidence: list = field(default_factory=list)     # optional dicts per frame

    @property
    def t_start(self): return self.frames[0]
    @property
    def t_end(self): return self.frames[-1]
    @property
    def length(self): return len(self.frames)
    @property
    def score(self): return float(np.mean(self.scores))


class TubeLinker:
    """Online greedy linking. Call update(t, detections) frame by frame.
    detections: list of dicts {bbox:[x1,y1,x2,y2], class_id:int, score:float,
                               evidence: optional dict}."""

    def __init__(self, iou_thr: float = 0.5, max_gap: int = 5):
        self.iou_thr = iou_thr
        self.max_gap = max_gap
        self.active: list[Tube] = []
        self.finished: list[Tube] = []

    def update(self, t: int, detections: list[dict]):
        # close tubes whose gap expired
        still = []
        for tb in self.active:
            (self.finished if t - tb.t_end > self.max_gap else still).append(tb)
        self.active = still
        # greedy matching by (same class, highest IoU)
        cands = []
        for di, d in enumerate(detections):
            for ti, tb in enumerate(self.active):
                if tb.class_id != d["class_id"]:
                    continue
                v = iou_xyxy(tb.boxes[-1], d["bbox"])
                if v >= self.iou_thr:
                    cands.append((v, ti, di))
        used_t, used_d = set(), set()
        for v, ti, di in sorted(cands, reverse=True):
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti); used_d.add(di)
            tb, d = self.active[ti], detections[di]
            tb.frames.append(t); tb.boxes.append(list(d["bbox"]))
            tb.scores.append(float(d["score"])); tb.evidence.append(d.get("evidence", {}))
        # unmatched detections start new tubes
        for di, d in enumerate(detections):
            if di in used_d:
                continue
            self.active.append(Tube(d["class_id"], [t], [list(d["bbox"])],
                                    [float(d["score"])], [d.get("evidence", {})]))

    def finalize(self):
        self.finished.extend(self.active)
        self.active = []
        return self.finished


# Minimum persistence n_c per class (frames @25 fps) - INITIAL values of the
# search space; final values are tuned on VALIDATION (Sec. III-E).
DEFAULT_MIN_PERSISTENCE = {0: 3, 1: 50, 2: 5, 3: 2, 4: 3, 5: 375, 6: 3, 7: 8}


def confirm_events(tubes: list[Tube], min_persistence: dict | None = None) -> list[dict]:
    """Filter tubes by n_c and emit structured alert records."""
    n_c = {**DEFAULT_MIN_PERSISTENCE, **(min_persistence or {})}
    alerts = []
    for tb in tubes:
        if tb.length < n_c.get(tb.class_id, 3):
            continue
        ev_keys = set().union(*[e.keys() for e in tb.evidence]) if tb.evidence else set()
        ev_mean = {k: float(np.mean([e[k] for e in tb.evidence if k in e]))
                   for k in ev_keys}
        alerts.append({
            "class_id": tb.class_id,
            "t_start": tb.t_start,
            "t_end": tb.t_end,
            "track": tb.boxes,
            "confidence_trajectory": [round(s, 4) for s in tb.scores],
            "confidence": round(tb.score, 4),
            "evidence_terms": ev_mean,   # e.g., tube-mean z and M gates
        })
    return alerts
