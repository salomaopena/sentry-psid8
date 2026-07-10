#!/usr/bin/env python3
"""Build PSID-8 splits: scenario-stratified, camera-disjoint.

Input: a JSON manifest listing clips:
  [{"clip_id": str, "camera_id": str, "scenario": str,
    "classes": [ids...], "n_frames": int}, ...]
Negative clips have "classes": [].

Guarantees (verified; aborts if violated):
  1. No camera appears in more than one split (no leakage).
  2. Every class present in the manifest appears in ALL splits.
  3. Approximate 70/15/15 proportions per scenario WHEN POSSIBLE; corpora with
     too few cameras per scenario (e.g., Le2i: 2 scenarios x 2 cameras) fall
     back to GLOBAL camera-level allocation. The fallback keeps guarantees 1-2,
     is recorded in the output JSON ("stratification" field), and must be
     declared in the paper.

Usage:
  python build_splits.py manifest.json --out splits.json --seed 0
"""
import argparse
import json
import random
from collections import defaultdict

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def group_by_camera(clips):
    cams = defaultdict(list)
    for c in clips:
        cams[c["camera_id"]].append(c)
    return cams


def camera_profile(clip_list):
    classes = set()
    frames = 0
    for c in clip_list:
        classes.update(c["classes"])
        frames += c["n_frames"]
    return classes, frames


def _coverage_ok(assign, cams, all_classes, ratios):
    cov = {s: set() for s in ratios}
    counts = {s: 0 for s in ratios}
    for cam_id, split in assign.items():
        cls, _ = camera_profile(cams[cam_id])
        cov[split].update(cls)
        counts[split] += 1
    return (all(counts[s] > 0 for s in ratios)
            and all(all_classes <= cov[s] for s in ratios))


def _finish(clips, assign, ratios):
    splits = {s: [] for s in ratios}
    for c in clips:
        splits[assign[c["camera_id"]]].append(c["clip_id"])
    return splits


def build_splits(clips, seed=0, ratios=RATIOS, max_tries=2000):
    rng = random.Random(seed)
    cams = group_by_camera(clips)
    all_classes = set()
    for c in clips:
        all_classes.update(c["classes"])

    by_scenario = defaultdict(list)
    for cam_id, cl in cams.items():
        by_scenario[cl[0]["scenario"]].append(cam_id)

    # ---- mode 1: scenario-stratified allocation ----
    for attempt in range(max_tries):
        assign = {}
        for scenario, cam_ids in by_scenario.items():
            ids = cam_ids[:]
            rng.shuffle(ids)
            n = len(ids)
            n_tr = max(1, round(n * ratios["train"])) if n >= 3 else max(1, n - 2)
            n_va = max(1, round(n * ratios["val"])) if n >= 3 else (1 if n >= 2 else 0)
            for i, cid in enumerate(ids):
                if i < n_tr:
                    assign[cid] = "train"
                elif i < n_tr + n_va:
                    assign[cid] = "val"
                else:
                    assign[cid] = "test"
        if _coverage_ok(assign, cams, all_classes, ratios):
            return _finish(clips, assign, ratios), assign, ("scenario", attempt)

    # ---- mode 2: GLOBAL camera-level fallback (few cameras per scenario) ----
    cam_list = sorted(cams)
    n = len(cam_list)
    if n < len(ratios):
        raise RuntimeError(
            f"Only {n} distinct cameras in the manifest - camera-disjoint "
            f"{'/'.join(ratios)} splits are impossible. Merge splits or add sources.")
    n_te = max(1, round(n * ratios["test"]))
    n_va = max(1, round(n * ratios["val"]))
    n_tr = n - n_va - n_te
    for attempt in range(max_tries):
        ids = cam_list[:]
        rng.shuffle(ids)
        assign = {}
        for i, cid in enumerate(ids):
            assign[cid] = ("train" if i < n_tr
                           else "val" if i < n_tr + n_va else "test")
        if _coverage_ok(assign, cams, all_classes, ratios):
            print("WARNING: scenario stratification infeasible (too few cameras "
                  "per scenario); using GLOBAL camera-level allocation. "
                  "Declare this in the paper.")
            return _finish(clips, assign, ratios), assign, ("global_fallback", attempt)
    raise RuntimeError(
        "Could not guarantee all classes in all splits with camera disjunction "
        f"after {max_tries} tries in both modes. Collect more cameras for the "
        "rare classes (report: run dataset_stats.py).")


def verify(clips, splits):
    cam_split = {}
    id2clip = {c["clip_id"]: c for c in clips}
    for split, clip_ids in splits.items():
        for cid in clip_ids:
            cam = id2clip[cid]["camera_id"]
            if cam in cam_split and cam_split[cam] != split:
                raise AssertionError(f"LEAKAGE: camera {cam} in {cam_split[cam]} and {split}")
            cam_split[cam] = split
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--out", default="splits.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    clips = json.load(open(args.manifest))
    splits, assign, (mode, tries) = build_splits(clips, seed=args.seed)
    verify(clips, splits)
    sizes = {s: len(v) for s, v in splits.items()}
    json.dump({"seed": args.seed, "stratification": mode, "splits": splits,
               "camera_assignment": assign},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"OK (mode={mode}, attempt {tries}). Sizes: {sizes}. "
          f"Cameras: {assign}. Saved to {args.out}")


if __name__ == "__main__":
    main()
