#!/usr/bin/env python3
"""Aggregate per-seed metric files into paper-ready numbers.

This is a thin CLI wrapper. The numeric core (mean, std, 95% bootstrap CI) is
NOT reimplemented here: it is imported from `sentry.seeds.aggregate`, which is
also what `run_over_seeds` uses internally. Before this fix, this file had its
own separate bootstrap implementation with a different output schema than
`sentry/seeds.py`, and the two could silently drift apart. There is now exactly
one place that computes the bootstrap CI over seeds.

Input: one JSON per seed, named metrics_seed{K}.json, each a flat dict of
metric name -> float (or nested one level, e.g. {"fall": {"mAP50": ...}}).
Output: same numeric fields as `sentry.seeds.aggregate` (mean, std, ci95,
per_seed), plus a human-readable "formatted" string for quick reading, e.g.
"12.3 +/- 0.4 [11.8, 12.9]".

Usage:
  python -m sentry.aggregate runs/metrics_seed*.json --out aggregated.json
"""
import argparse
import glob
import json

from sentry.seeds import aggregate as _aggregate_core


def flatten(d: dict, prefix: str = "") -> dict:
    """Flatten a (possibly one-level nested) metrics dict into scalar leaves,
    e.g. {"fall": {"mAP50": 0.7}} -> {"fall/mAP50": 0.7}."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "/"))
        elif isinstance(v, (int, float)) and v == v:   # v == v excludes NaN
            out[key] = float(v)
    return out


def aggregate(files: list[str], n_boot: int = 1000, seed: int = 0) -> dict:
    """Load, flatten and aggregate metric files.

    Delegates the numeric computation to `sentry.seeds.aggregate` so that the
    CLI path and the `run_over_seeds` path always agree on the same numbers for
    the same input. Adds a "formatted" display string per metric on top.
    """
    runs = [flatten(json.load(open(f))) for f in files]
    # "seed" may appear as a scalar field in the source JSON (a seed id, not a
    # metric); exclude it from the metric set, matching sentry.seeds.aggregate's
    # own convention.
    common_keys = sorted(set.intersection(*[set(r) for r in runs]) - {"seed"})
    # sentry.seeds.aggregate expects a list of dicts with a "seed" key and
    # numeric leaves; the seed id here is just the file's position, since
    # file-based runs do not necessarily carry their own seed identifier.
    per_seed = [{**{k: r[k] for k in common_keys}, "seed": i}
                for i, r in enumerate(runs)]
    core = _aggregate_core(per_seed, n_boot=n_boot)

    table = {}
    for k in common_keys:
        entry = dict(core[k])   # copy: mean, std, ci95, per_seed
        lo, hi = entry["ci95"]
        entry["n_seeds"] = core["n_seeds"]
        entry["formatted"] = (f"{entry['mean']:.1f} \u00b1 {entry['std']:.1f} "
                              f"[{lo:.1f}, {hi:.1f}]")
        table[k] = entry
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
