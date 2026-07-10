"""SENTRY losses (paper Sec. III-D, Eq. 3).

L_total = l1*L_box(CIoU) + l2*L_cls(BCE) + l3*L_dfl + l4*L_tc

L_box, L_cls and L_dfl come from the underlying detector (on the Ultralytics
path they are the unchanged v8DetectionLoss). This module implements the new
term, L_tc, and the aggregator. L_tc matches detections across consecutive
frames by IoU >= 0.5 and penalizes the L1 variation of the scores
(anti-flicker), directly optimizing the event-level metrics.
"""
from __future__ import annotations
import torch


def pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    tl = torch.max(a[:, None, :2], b[None, :, :2])
    br = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (br - tl).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def temporal_consistency_loss(boxes_t, scores_t, boxes_prev, scores_prev,
                              iou_thr: float = 0.5) -> torch.Tensor:
    """L_tc = mean |s_t,i - s_{t-1},j| over IoU-matched pairs.

    boxes_*: (N,4) xyxy in the same reference frame; scores_*: (N,K) sigmoids.
    Greedy IoU matching (no gradient through the matching; gradient flows
    through the scores of both frames).
    """
    if boxes_t.numel() == 0 or boxes_prev.numel() == 0:
        return scores_t.new_zeros(())
    with torch.no_grad():
        iou = pairwise_iou(boxes_t, boxes_prev)
        vals, idx = iou.max(dim=1)
        keep = vals >= iou_thr
    if keep.sum() == 0:
        return scores_t.new_zeros(())
    return (scores_t[keep] - scores_prev[idx[keep]]).abs().mean()


class SentryLoss(torch.nn.Module):
    """Eq. (3) aggregator. `base_loss` is the underlying detector loss
    (returns L_box, L_cls, L_dfl combined or separate, backend-dependent)."""

    def __init__(self, base_loss, lambdas=(7.5, 0.5, 1.5, 1.0)):
        super().__init__()
        self.base_loss = base_loss
        self.l1, self.l2, self.l3, self.l4 = lambdas

    def forward(self, preds_seq, targets_seq, decoded_seq=None):
        """preds_seq/targets_seq: per-frame lists of the clip (temporal order).
        decoded_seq: per-frame list of (boxes, scores) for L_tc."""
        total, parts = 0.0, {"box": 0.0, "cls": 0.0, "dfl": 0.0, "tc": 0.0}
        for p, t in zip(preds_seq, targets_seq):
            lb, lc, ld = self.base_loss(p, t)
            total = total + self.l1 * lb + self.l2 * lc + self.l3 * ld
            parts["box"] += float(lb); parts["cls"] += float(lc); parts["dfl"] += float(ld)
        if decoded_seq is not None:
            for (b1, s1), (b0, s0) in zip(decoded_seq[1:], decoded_seq[:-1]):
                ltc = temporal_consistency_loss(b1, s1, b0, s0)
                total = total + self.l4 * ltc
                parts["tc"] += float(ltc)
        return total, parts
