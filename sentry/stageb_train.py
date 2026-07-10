"""Stage B supervised training helpers (paper Sec. III-D).

Rationale (documented after two failed L_tc-only attempts): training the TFM with
the temporal-consistency term ALONE is ill-posed - its trivial optimum is "change
nothing", which pins the network at initialization (L_tc == 0 with zero gradient).
Stage B must therefore be driven by the detector's supervised loss
(v8DetectionLoss: box+cls+dfl) over the labeled frames of each clip, with L_tc as
an auxiliary regularizer on the raw head features between consecutive frames.

This module builds the `batch` dict that v8DetectionLoss expects and computes a
lightweight temporal-consistency term on the training-format head output.
"""
from __future__ import annotations
import torch


def build_loss_batch(labels_per_frame, imgsz, device):
    """Assemble the v8DetectionLoss batch dict for a stack of frames.

    labels_per_frame: list (len = n_frames) of (k_i, 5) tensors [cls, cx, cy, w, h]
                      normalized in [0,1]; empty tensor for background frames.
    Returns dict with batch_idx (M,), cls (M,1), bboxes (M,4) - M = total boxes.
    """
    idx, cls, box = [], [], []
    for fi, lab in enumerate(labels_per_frame):
        if lab is None or lab.numel() == 0:
            continue
        lab = lab.to(device)
        n = lab.shape[0]
        idx.append(torch.full((n,), fi, dtype=torch.long, device=device))
        cls.append(lab[:, 0:1])
        box.append(lab[:, 1:5])
    if not idx:
        return {"batch_idx": torch.zeros(0, dtype=torch.long, device=device),
                "cls": torch.zeros(0, 1, device=device),
                "bboxes": torch.zeros(0, 4, device=device)}
    return {"batch_idx": torch.cat(idx),
            "cls": torch.cat(cls),
            "bboxes": torch.cat(box)}


def temporal_consistency_feats(feats_t, feats_prev, motion=None):
    """L1 consistency on the raw multi-scale head features between frames.

    feats_*: list of (1, C, H, W) tensors (training-format head output, one per
             FPN level). Compares only where content persists; if `motion`
             (scalar in [0,1]) is high, the penalty is down-weighted so that
             genuine change (e.g., the fall transition) is not punished.
    """
    if feats_prev is None:
        return None
    total, n = 0.0, 0
    w = 1.0 if motion is None else float(1.0 - min(max(motion, 0.0), 1.0))
    for ft, fp in zip(feats_t, feats_prev):
        total = total + (ft - fp).abs().mean()
        n += 1
    return (total / max(n, 1)) * w


def stack_frame_labels(clip_labels, n_frames):
    """clip_labels: list per frame of (k,5) numpy/tensor arrays -> list of tensors."""
    out = []
    for i in range(n_frames):
        lab = clip_labels[i] if i < len(clip_labels) else None
        if lab is None:
            out.append(torch.zeros(0, 5))
        else:
            t = lab if torch.is_tensor(lab) else torch.as_tensor(lab, dtype=torch.float32)
            out.append(t.reshape(-1, 5) if t.numel() else torch.zeros(0, 5))
    return out
