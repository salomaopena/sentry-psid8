#!/usr/bin/env python3
"""SENTRY Stage B training: temporal fine-tuning with ordered clips.

Prerequisite (Stage A, frame-level, via the ultralytics CLI on Kaggle):
  yolo detect train data=data.yaml model=yolov8m.pt imgsz=640 epochs=100 seed=0

This script freezes backbone+neck and trains the TFM (+ optional head) with
SentryLoss (L_tc on). Functional skeleton: the [ADAPT] points depend on the
pinned ultralytics version (prediction decoding for L_tc; a concrete
implementation ships in the Kaggle notebook, cell [5]).
"""
import argparse
import json
import random

import numpy as np


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Stage A best.pt")
    ap.add_argument("--clips-dir", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--hidden-ch", type=int, default=128)
    ap.add_argument("--lambdas", type=float, nargs=4, default=[7.5, 0.5, 1.5, 1.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/sentry_temporal")
    args = ap.parse_args()
    set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from sentry.ultralytics_adapter import SentryYOLO
    from sentry.losses import temporal_consistency_loss
    from sentry.data import VideoClipDataset, collate_clips

    split_ids = json.load(open(args.splits))["splits"]["train"]
    ds = VideoClipDataset(args.clips_dir, split_ids, window=args.window)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_clips,
                    num_workers=2)

    base = YOLO(args.weights)
    model = SentryYOLO(base, hidden_ch=args.hidden_ch).cuda()
    model(torch.zeros(1, 3, 640, 640).cuda())          # materialize the TFM
    model.freeze_base()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr)
    l4 = args.lambdas[3]

    for epoch in range(args.epochs):
        for batch in dl:
            opt.zero_grad()
            loss_total = 0.0
            for clip in batch:
                model.reset_stream()
                prev_decoded = None
                for f in range(clip["images"].shape[0]):
                    img = clip["images"][f:f + 1].cuda()
                    raw = model(img)
                    # [ADAPT] decode raw -> (boxes xyxy, scores KxN) with the
                    # pinned ultralytics version; see notebook cell [5] for a
                    # concrete differentiable decoding via NMS index gathering.
                    decoded = None  # (boxes, scores)
                    if decoded is not None and prev_decoded is not None:
                        loss_total = loss_total + l4 * temporal_consistency_loss(
                            decoded[0], decoded[1], prev_decoded[0], prev_decoded[1])
                    prev_decoded = decoded
            if torch.is_tensor(loss_total):
                loss_total.backward()
                opt.step()
        print(f"epoch {epoch}: ok")
    torch.save({"tfm": model.tfm.state_dict(), "args": vars(args)},
               f"{args.out}/tfm_last.pt")


if __name__ == "__main__":
    main()
