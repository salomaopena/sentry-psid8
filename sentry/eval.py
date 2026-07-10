#!/usr/bin/env python3
"""SENTRY evaluation.

Spatial (mAP@50, mAP@50:95): use the backend's `yolo detect val` with the
wrapper loaded - those numbers come from the official validator, not from here.

Event level (AUC, tIoU, P/R/F1) + latency/FPS: this script, over the alert
records produced by run_clip_to_alerts and the event ground truth.
"""
import argparse
import json
from sentry.metrics import event_auc, event_prf, bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True, help="JSON: {clip_id: [alerts]}")
    ap.add_argument("--gt-events", required=True, help="JSON: {clip_id: [GT events]}")
    ap.add_argument("--tiou", type=float, default=0.5)
    args = ap.parse_args()
    alerts = json.load(open(args.alerts))
    gts = json.load(open(args.gt_events))
    per_video_auc = []
    all_p, all_g = [], []
    for cid, g in gts.items():
        p = alerts.get(cid, [])
        all_p += p; all_g += g
        a = event_auc(p, g, args.tiou)
        if a == a:                                   # skip NaN
            per_video_auc.append(a)
    mean, lo, hi = bootstrap_ci(per_video_auc)
    print(f"Event AUC (per-video mean): {mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
    prf = event_prf(all_p, all_g, args.tiou)
    for c, m in prf.items():
        print(f"  class {c}: P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} "
              f"(tp={m['tp']} fp={m['fp']} fn={m['fn']})")


if __name__ == "__main__":
    main()
