"""SENTRY Ultralytics adapter (recommended path for the paper numbers).

Strategy: load a YOLO (v8/11) pre-trained frame-level on the target corpus and
insert the TemporalFeatureMemory right BEFORE the Detect head by wrapping the
head's forward. The official backbone+neck+head (and their L_box, L_cls, L_dfl
losses) stay intact; the architectural delta is exactly the one in Sec. III.

Requires: ultralytics >= 8.2 (pin the exact version on Kaggle and record it in
Table IV-B). If the internal API changes, the touch points are isolated in
_find_detect_head() and _wrap_head().

Two-stage training recipe (fits 2x T4):
  Stage A - frame-level: train a standard YOLO on the corpus (ultralytics train).
  Stage B - temporal: freeze backbone+neck, train the TFM (+ optional head)
            with ordered clips (data.VideoClipDataset) and SentryLoss (L_tc on).
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .modules import TemporalFeatureMemory
from .tubes import TubeLinker, confirm_events


def _find_detect_head(yolo_model: nn.Module):
    """Locate the Detect module (last module of the DetectionModel)."""
    seq = yolo_model.model.model if hasattr(yolo_model, "model") else yolo_model.model
    head = seq[-1]
    if head.__class__.__name__ not in {"Detect", "DetectV10", "Pose", "Segment"}:
        raise RuntimeError(f"Unexpected head: {head.__class__.__name__}. "
                           "Adjust _find_detect_head for your ultralytics version.")
    return head


class SentryYOLO(nn.Module):
    """Wraps an Ultralytics model with the TFM.

    Usage (Kaggle):
        from ultralytics import YOLO
        base = YOLO("runs/frame_level/weights/best.pt")   # Stage A
        sentry = SentryYOLO(base, hidden_ch=128)
        sentry.reset_stream()                              # per clip/stream
        for frame in clip: out = sentry(frame_tensor)
    """

    def __init__(self, base_yolo, hidden_ch: int = 128, gate_k: int = 3,
                 alpha_init: float = 0.5):
        super().__init__()
        self.base = base_yolo.model if hasattr(base_yolo, "model") else base_yolo
        self.head = _find_detect_head(base_yolo)
        pyr_chs = [m.in_channels if hasattr(m, "in_channels") else None
                   for m in getattr(self.head, "cv2", [])]
        if not pyr_chs or pyr_chs[0] is None:
            pyr_chs = None                       # fallback: infer on first pass
        self.tfm = None
        self._pyr_chs = pyr_chs
        self._hidden_ch, self._gate_k, self._alpha0 = hidden_ch, gate_k, alpha_init
        self.last_evidence = {}
        self._wrap_head()

    # ---- integration ----
    def _wrap_head(self):
        head = self.head
        original_forward = head.forward
        parent = self

        def wrapped(x):
            # x: list of pyramid features [P3, P4, P5]
            if parent.tfm is None:
                chs = [t.shape[1] for t in x]
                parent.tfm = TemporalFeatureMemory(
                    tuple(chs), parent._hidden_ch, parent._gate_k, parent._alpha0
                ).to(x[0].device)
            fused, ev = parent.tfm(x)
            parent.last_evidence = {k: float(v.detach()) if hasattr(v, "detach") else float(v)
                                    for k, v in ev.items()}
            return original_forward(fused)

        head.forward = wrapped

    # ---- API ----
    def reset_stream(self):
        if self.tfm is not None:
            self.tfm.reset()

    def freeze_base(self):
        for p in self.base.parameters():
            p.requires_grad_(False)
        if self.tfm is not None:
            for p in self.tfm.parameters():
                p.requires_grad_(True)

    def forward(self, x):
        return self.base(x)


@torch.no_grad()
def run_clip_to_alerts(sentry: SentryYOLO, frames, conf_per_class: dict,
                       iou_link: float = 0.5, max_gap: int = 5,
                       min_persistence: dict | None = None, nms_fn=None):
    """Streaming inference of one clip -> alert records (Sec. III-E).

    frames: iterable of (1,3,H,W) tensors in temporal order.
    conf_per_class: tau_c thresholds selected ON VALIDATION.
    nms_fn: function (raw_pred) -> list of dicts {bbox, class_id, score}; on
            Kaggle use ultralytics.utils.ops.non_max_suppression and convert.
    """
    sentry.eval()
    sentry.reset_stream()
    linker = TubeLinker(iou_thr=iou_link, max_gap=max_gap)
    for t, frame in enumerate(frames):
        raw = sentry(frame)
        dets = nms_fn(raw) if nms_fn else []
        dets = [d for d in dets if d["score"] >= conf_per_class.get(d["class_id"], 0.25)]
        for d in dets:
            d.setdefault("evidence", {}).update(sentry.last_evidence)
        linker.update(t, dets)
    return confirm_events(linker.finalize(), min_persistence)
