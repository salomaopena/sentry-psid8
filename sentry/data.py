"""Clip dataset for SENTRY temporal training.

Serves ordered windows of W frames per clip (batch = clips; within a clip the
temporal order is preserved for the TFM and for L_tc). Labels in YOLO layout
with .attrs.json sidecars produced by coco_to_yolo.py.
"""
from __future__ import annotations
import glob
import os
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    import cv2
    _TORCH = True
except Exception:                                    # allows offline import
    _TORCH = False
    Dataset = object


class VideoClipDataset(Dataset):
    def __init__(self, clips_dir: str, split_ids: list[str], window: int = 8,
                 stride: int = 4, imgsz: int = 640, event_only: bool = False):
        """event_only=True keeps ONLY windows containing >=1 labeled frame
        (option (a)): denser supervision signal, faster Stage B."""
        self.window, self.stride, self.imgsz = window, stride, imgsz
        self.samples = []                            # (clip_id, [frame_paths])
        for cid in split_ids:
            frames = sorted(glob.glob(os.path.join(clips_dir, cid, "frames", "*.jpg")))
            for s in range(0, max(1, len(frames) - window + 1), stride):
                w = frames[s:s + window]
                if len(w) != window:
                    continue
                if event_only:
                    has_label = any(
                        os.path.exists(fp.replace("/frames/", "/labels/")
                                       .rsplit(".", 1)[0] + ".txt")
                        and os.path.getsize(fp.replace("/frames/", "/labels/")
                                            .rsplit(".", 1)[0] + ".txt") > 0
                        for fp in w)
                    if not has_label:
                        continue
                self.samples.append((cid, w))

    def __len__(self):
        return len(self.samples)

    def _load_labels(self, frame_path):
        lp = frame_path.replace("/frames/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if not os.path.exists(lp):
            return np.zeros((0, 5), dtype=np.float32)
        rows = [l.split() for l in open(lp) if l.strip()]
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), np.float32)

    def __getitem__(self, i):
        assert _TORCH, "torch/cv2 are required at training time"
        cid, frame_paths = self.samples[i]
        imgs, labels = [], []
        for fp in frame_paths:
            im = cv2.imread(fp)
            im = cv2.resize(im, (self.imgsz, self.imgsz))
            imgs.append(torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255)
            labels.append(torch.from_numpy(self._load_labels(fp)))
        return {"clip_id": cid, "images": torch.stack(imgs), "labels": labels}


def collate_clips(batch):
    return batch                                     # list of clips; loop handles it


def collate_batched(batch):
    """Stack N same-window-length clip samples for batched Stage B training.

    Unlike `collate_clips` (which returns the list as-is, for a per-clip
    Python loop), this groups the images of all N clips at each timestep into
    one (N,3,H,W) tensor, so the model is called once per timestep for the
    WHOLE mini-batch instead of once per (clip, timestep) pair. This is what
    `sentry/train.py` uses; `ConvGRUCell`/`MotionGate`/`TemporalFeatureMemory`
    already operate on an arbitrary batch dimension (see sentry/modules.py), so
    no change to the model is needed, only to how the loop feeds it.

    Every sample in `batch` must share the same window length T, which holds
    for any single VideoClipDataset instance (fixed `window` argument).

    Returns:
      images:     (N, T, 3, H, W) tensor
      labels_by_t: list of length T; labels_by_t[t] is a list of N per-clip
                   (k,5) label tensors for timestep t (k varies per clip/frame)
      clip_ids:   list of N clip identifiers, for logging
    """
    assert _TORCH, "torch is required to collate a training batch"
    t_lens = {b["images"].shape[0] for b in batch}
    assert len(t_lens) == 1, (
        f"collate_batched requires a single fixed window length across the "
        f"batch, got {t_lens}. Build the DataLoader from one VideoClipDataset "
        f"(one fixed `window`) at a time."
    )
    T = t_lens.pop()
    images = torch.stack([b["images"] for b in batch])         # (N, T, 3, H, W)
    labels_by_t = [[b["labels"][t] for b in batch] for t in range(T)]
    clip_ids = [b["clip_id"] for b in batch]
    return {"images": images, "labels_by_t": labels_by_t, "clip_ids": clip_ids}
