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


def decode_eval_output(raw, end2end: bool, conf: float = 0.03, iou: float = 0.7,
                       max_det: int = 100):
    """Decode a model's EVAL-mode (inference) output into a list of detection
    dicts `{"bbox": [x1,y1,x2,y2], "score": float, "class_id": int}`,
    regardless of which YOLO generation produced it.

    `end2end` MUST be passed explicitly (e.g. `sentry.base.end2end`) rather
    than inferred from `raw`'s type or shape: in the Ultralytics version this
    was verified against, BOTH end2end (YOLO26) and non-end2end (YOLOv8/11)
    models return the same `(tensor, dict)` tuple shape in eval mode, so
    neither the Python type nor a shape heuristic reliably distinguishes them
    -- only the model's own `end2end` attribute does (the same flag
    `DetectionModel.init_criterion()` uses to pick the training loss, and the
    same flag Ultralytics' own `DetectionPredictor.postprocess` passes to
    `non_max_suppression`). This function mirrors that official code path
    instead of re-deriving the distinction independently.
    """
    from ultralytics.utils.nms import non_max_suppression
    preds = raw[0] if isinstance(raw, tuple) else raw
    dets = non_max_suppression(preds, conf, iou, max_det=max_det, end2end=end2end)[0]
    dets = dets.detach().cpu().numpy()
    return [{"bbox": r[:4].tolist(), "score": float(r[4]), "class_id": int(r[5])}
            for r in dets]


@torch.no_grad()
def run_clip_to_alerts(sentry: SentryYOLO, frames, conf_per_class: dict,
                       iou_link: float = 0.5, max_gap: int = 5,
                       min_persistence: dict | None = None, nms_fn=None):
    """Streaming inference of one clip -> alert records (Sec. III-E).

    frames: iterable of (1,3,H,W) tensors in temporal order.
    conf_per_class: tau_c thresholds selected ON VALIDATION.
    nms_fn: function (raw_pred) -> list of dicts {bbox, class_id, score};
            defaults to `decode_eval_output` with `end2end` read from
            `sentry.base.end2end`, which already handles both YOLOv8/11
            (standard NMS) and YOLO26 (end-to-end NMS) via Ultralytics' own
            `non_max_suppression(..., end2end=...)`.
    """
    sentry.eval()
    sentry.reset_stream()
    linker = TubeLinker(iou_thr=iou_link, max_gap=max_gap)
    if nms_fn is None:
        is_e2e = getattr(sentry.base, "end2end", False)
        nms_fn = lambda raw: decode_eval_output(raw, end2end=is_e2e)
    for t, frame in enumerate(frames):
        raw = sentry(frame)
        dets = nms_fn(raw)
        dets = [d for d in dets if d["score"] >= conf_per_class.get(d["class_id"], 0.25)]
        for d in dets:
            d.setdefault("evidence", {}).update(sentry.last_evidence)
        linker.update(t, dets)
    return confirm_events(linker.finalize(), min_persistence)
