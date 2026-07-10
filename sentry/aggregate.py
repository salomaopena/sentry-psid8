#!/usr/bin/env python3
"""Aggregate per-seed metric files into paper-ready numbers.

Input: one JSON per seed, named metrics_seed{K}.json, each a flat dict of
metric name -> float (or nested one level, e.g. {"fall": {"mAP50": ...}}).
Output: mean +/- std per metric, plus a 95% bootstrap CI over the seed means,
formatted exactly as Table II expects: "12.3 ± 0.4 [11.8, 12.9]".

Usage:
  python -m sentry.aggregate runs/metrics_seed*.json --out aggregated.json
"""
import argparse
import glob
import json
import numpy as np


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float)) and v == v:
            out[key] = float(v)
    return out


def aggregate(files, n_boot=1000, seed=0):
    runs = [flatten(json.load(open(f))) for f in files]
    keys = sorted(set.intersection(*[set(r) for r in runs]))
    rng = np.random.default_rng(seed)
    table = {}
    for k in keys:
        v = np.array([r[k] for r in runs])
        boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        table[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "ci95": [float(lo), float(hi)], "n_seeds": len(v),
                    "per_seed": v.tolist(),
                    "formatted": f"{v.mean():.1f} \u00b1 {v.std(ddof=1) if len(v) > 1 else 0:.1f} "
                                 f"[{lo:.1f}, {hi:.1f}]"}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="metrics_seed*.json (glob ok)")
    ap.add_argument("--out", default="aggregated.json")
    args = ap.parse_args()
    files = sorted(sum([glob.glob(f) for f in args.files], []))
    assert len(files) >= 2, f"Need >=2 seed files, got {files}. Single-seed results are anecdotal."
    table = aggregate(files)
    json.dump(table, open(args.out, "w"), indent=1)
    print(f"Aggregated {len(files)} seeds -> {args.out}")
    for k, v in table.items():
        print(f"  {k:40s} {v['formatted']}")


if __name__ == "__main__":
    main()
