#!/usr/bin/env python3
"""PSID-8 inter-annotator agreement.

Input: two JSON files (annotator A and B) with the format:
  {frame_key: [{"bbox": [x1,y1,x2,y2], "class_ids": [ints]}, ...], ...}
over the SAME frame sample (the double-annotated sample, >= 20%).

Output:
  - Cohen's kappa per class (class presence/absence in the frame)
  - mean IoU of matched boxes (greedy IoU matching, same frame,
    non-empty class intersection)

Quality gates: read from psid8/schema.json's quality_gates field via
psid8.schema (never hardcoded here as a second copy); the active thresholds
are printed at the start of every run.
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import cohen_kappa_score

# Runs as a direct CLI script (locally and on Kaggle), where Python puts only
# this file's own directory on sys.path, not the repo root -- so the repo
# root is added explicitly here, before importing psid8.schema.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from psid8.schema import CLASS_NAMES, MIN_KAPPA_PER_CLASS, MIN_MEAN_BOX_IOU

N_CLASSES = len(CLASS_NAMES)  # single source of truth; see psid8/schema.py


def frame_class_vector(anns, n_classes=N_CLASSES):
    v = np.zeros(n_classes, dtype=int)
    for a in anns:
        for c in a["class_ids"]:
            v[c] = 1
    return v


def iou(b1, b2):
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)


def greedy_match_ious(anns_a, anns_b):
    pairs = []
    used_b = set()
    cand = []
    for i, a in enumerate(anns_a):
        for j, b in enumerate(anns_b):
            if set(a["class_ids"]) & set(b["class_ids"]):
                cand.append((iou(a["bbox"], b["bbox"]), i, j))
    for v, i, j in sorted(cand, reverse=True):
        if j in used_b or any(p[0] == i for p in pairs):
            continue
        pairs.append((i, j, v))
        used_b.add(j)
    return [v for _, _, v in pairs]


def compute(ann_a, ann_b, n_classes=N_CLASSES):
    keys = sorted(set(ann_a) & set(ann_b))
    if not keys:
        raise ValueError("No frames in common between the two annotators.")
    A = np.stack([frame_class_vector(ann_a[k], n_classes) for k in keys])
    B = np.stack([frame_class_vector(ann_b[k], n_classes) for k in keys])
    kappas = {}
    for c in range(n_classes):
        if A[:, c].sum() == 0 and B[:, c].sum() == 0:
            kappas[c] = None  # class absent from the sample
        elif (A[:, c] == B[:, c]).all():
            kappas[c] = 1.0
        else:
            kappas[c] = float(cohen_kappa_score(A[:, c], B[:, c]))
    ious = []
    for k in keys:
        ious.extend(greedy_match_ious(ann_a[k], ann_b[k]))
    return kappas, (float(np.mean(ious)) if ious else None), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotator_a")
    ap.add_argument("annotator_b")
    ap.add_argument("--kappa-min", type=float, default=MIN_KAPPA_PER_CLASS)
    ap.add_argument("--iou-min", type=float, default=MIN_MEAN_BOX_IOU)
    args = ap.parse_args()
    print(f"quality gates (from psid8/schema.json): "
          f"kappa >= {args.kappa_min}, mean IoU >= {args.iou_min}")
    ka, miou, n = compute(json.load(open(args.annotator_a)), json.load(open(args.annotator_b)))
    print(f"Frames in common: {n}")
    ok = True
    for c, v in ka.items():
        status = "-" if v is None else ("OK" if v >= args.kappa_min else "FAIL")
        ok &= v is None or v >= args.kappa_min
        print(f"  class {c}: kappa = {'n/a' if v is None else f'{v:.3f}'} [{status}]")
    print(f"Mean IoU of matched boxes: {miou if miou is None else f'{miou:.3f}'}")
    ok &= (miou is None) or (miou >= args.iou_min)
    print("QUALITY GATE:", "PASS" if ok else "FAIL - revise the guide and retrain annotators")


if __name__ == "__main__":
    main()
