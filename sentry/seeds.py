"""Multi-seed protocol utilities (paper Sec. IV-C/IV-D).

Community standard: single-seed results are not sufficient for comparative
claims. This module (i) seeds every RNG source deterministically, (ii) runs a
pipeline over a PRE-DECLARED seed list, and (iii) aggregates per-seed metric
dicts into mean +/- std with percentile-bootstrap CIs.

Design decisions (state them in the paper):
- Seeds are fixed and pre-registered (default [0, 1, 2]) BEFORE any result is
  seen - this rules out seed-hacking.
- Data splits stay FIXED across seeds (build_splits uses its own seed): the
  reported variance isolates training stochasticity (init, batch order,
  augmentation), not split variance.
"""
from __future__ import annotations
import json
import os
import random

import numpy as np

DECLARED_SEEDS = [0, 1, 2]          # pre-registered; change ONLY before Phase 2


def set_all_seeds(seed: int):
    """Seed python, numpy, torch (CPU+CUDA) and make cuDNN deterministic."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def dataloader_seeding(seed: int):
    """Return (generator, worker_init_fn) so DataLoader workers are seeded too:
        DataLoader(..., generator=g, worker_init_fn=winit)
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)

    def winit(worker_id):
        s = seed + worker_id
        np.random.seed(s)
        random.seed(s)

    return g, winit


def run_over_seeds(pipeline_fn, seeds=None, out_dir="runs/seeds"):
    """Run `pipeline_fn(seed) -> dict of scalar metrics` for each seed, saving
    one JSON per seed and an aggregate JSON. Resumable: seeds whose JSON
    already exists are skipped (Kaggle-session friendly)."""
    seeds = list(seeds or DECLARED_SEEDS)
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for s in seeds:
        path = os.path.join(out_dir, f"metrics_seed{s}.json")
        if os.path.exists(path):
            print(f"[seed {s}] found existing {path} - skipping (resume mode)")
            results.append(json.load(open(path)))
            continue
        print(f"[seed {s}] running...")
        set_all_seeds(s)
        m = pipeline_fn(s)
        m["seed"] = s
        json.dump(m, open(path, "w"), indent=1)
        results.append(m)
    agg = aggregate(results)
    json.dump(agg, open(os.path.join(out_dir, "aggregate.json"), "w"), indent=1)
    print("Aggregate:", json.dumps(agg, indent=1))
    return results, agg


def aggregate(per_seed: list[dict], n_boot: int = 1000, alpha: float = 0.05):
    """mean +/- std + bootstrap CI per numeric key across seeds."""
    rng = np.random.default_rng(0)
    keys = [k for k in per_seed[0] if k != "seed"
            and isinstance(per_seed[0][k], (int, float))]
    out = {"n_seeds": len(per_seed), "seeds": [m.get("seed") for m in per_seed]}
    for k in keys:
        v = np.array([m[k] for m in per_seed], dtype=float)
        boots = [rng.choice(v, size=len(v), replace=True).mean()
                 for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  "ci95": [float(lo), float(hi)], "per_seed": v.tolist()}
    return out
