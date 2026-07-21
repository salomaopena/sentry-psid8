#!/usr/bin/env python3
"""PSID-8 statistics per split: clips, frames, instances per class,
co-occurrence and negative ratio. Produces the basis of the paper's
dataset table."""
import json, sys
from collections import Counter
import itertools

def main(manifest_path, splits_path):
    clips = {c["clip_id"]: c for c in json.load(open(manifest_path))}
    splits = json.load(open(splits_path))["splits"]
    for split, ids in splits.items():
        n_frames = sum(clips[i]["n_frames"] for i in ids)
        cls = Counter(); co = Counter(); neg = 0
        for i in ids:
            cc = clips[i]["classes"]
            if not cc: neg += 1
            cls.update(cc)
            for a, b in itertools.combinations(sorted(set(cc)), 2):
                co[(a, b)] += 1
        print(f"\n== {split}: {len(ids)} clips, {n_frames} frames, "
              f"{neg} negatives ({neg/max(len(ids),1):.0%})")
        for c in range(8):
            print(f"  class {c}: {cls.get(c, 0)} clips")
        if co:
            print("  co-occurrences:", dict(co))

if __name__ == "__main__":
    main(*sys.argv[1:3])
