#!/usr/bin/env python3
"""Pre-experiment integrity checks (Phase 1 of the protocol):
1. SHA256 of splits.json and the manifest (freezes the dataset);
2. camera disjunction across splits;
3. presence of every class in every split.
Writes integrity_report.json - commit it BEFORE any training."""
import hashlib, json, sys, datetime

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def main(manifest_path, splits_path):
    clips = {c["clip_id"]: c for c in json.load(open(manifest_path))}
    data = json.load(open(splits_path)); splits = data["splits"]
    cam_split = {}
    for s, ids in splits.items():
        for i in ids:
            cam = clips[i]["camera_id"]
            assert cam_split.setdefault(cam, s) == s, f"LEAKAGE: camera {cam}"
    classes_per_split = {s: set().union(*[set(clips[i]["classes"]) for i in ids]) or set()
                         for s, ids in splits.items()}
    all_classes = set().union(*classes_per_split.values())
    for s, cs in classes_per_split.items():
        missing = all_classes - cs
        assert not missing, f"Split {s} missing classes {missing}"
    report = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "manifest_sha256": sha256(manifest_path),
              "splits_sha256": sha256(splits_path),
              "checks": ["camera_disjoint", "class_coverage"], "status": "PASS"}
    json.dump(report, open("integrity_report.json", "w"), indent=2)
    print("PASS - integrity_report.json written. Freeze this commit (Phase 1).")

if __name__ == "__main__":
    main(*sys.argv[1:3])
