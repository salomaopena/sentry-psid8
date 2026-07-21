#!/usr/bin/env python3
"""Curate PSID-8 video clips from public source corpora.

The benchmark is built by RE-ANNOTATING clips extracted from existing public
datasets, not by filming new footage. This script performs the extraction and,
crucially, records the metadata that everything downstream depends on:

  * `source` / `source_id` — provenance, so the corpus is auditable and so we
    distribute ANNOTATIONS rather than redistributing someone else's video;
  * `camera_id` / `scene_id` — what `build_splits.py` keeps disjoint across
    train/val/test. Getting this wrong silently inflates every reported metric,
    because the same camera would appear on both sides of the split;
  * `intended_class` — the class the clip is a CANDIDATE for, taken from the
    source label. It is a hint for the annotator, never a label: the annotator
    may find no event at all, or a different one. Treating a source label as
    ground truth would import that corpus's noise into ours.

Usage
-----
    # 1. plan: what would be extracted, without touching any video
    python curate_clips.py plan --config curation.yaml --out plan.json

    # 2. review plan.json by hand, then extract
    python curate_clips.py extract --plan plan.json --out /data/psid8_raw

    # 3. the extracted tree is ready for CVAT upload and for the manifest
    python curate_clips.py manifest --root /data/psid8_raw --out manifest.json

Clip length defaults to 10 s, which is what the annotation budget assumes
(~8 min of annotation per clip; see CALIBRATION_PROTOCOL.md).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
from collections import Counter, defaultdict

CLASSES = ["accident", "suspicious_behavior", "crime", "fire",
           "intrusion", "suspicious_object", "fall", "vandalism"]

# Source-label -> candidate PSID-8 class. A hint for the annotator, NOT a label.
UCF_HINTS = {
    "Assault": "crime", "Fighting": "crime", "Robbery": "crime",
    "Shooting": "crime", "Stealing": "crime", "Shoplifting": "crime",
    "Abuse": "crime", "Burglary": "intrusion", "RoadAccidents": "accident",
    "Vandalism": "vandalism", "Arson": "fire",
    # Arrest and Explosion map to no PSID-8 class and are deliberately excluded.
}


def sha256(path: str, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(limit):
            h.update(chunk)
    return h.hexdigest()


def probe_duration(path: str) -> float:
    """Duration in seconds via ffprobe (ffmpeg is already a dependency)."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nk=1:nw=1", path],
        capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


# --------------------------------------------------------------------------
# PLAN
# --------------------------------------------------------------------------
def plan_ucf(root: str, temporal_ann: str, per_class: int, clip_s: float,
             seed: int) -> list[dict]:
    """UCF-Crime: its temporal annotation tells us WHERE the event is, which is
    what makes re-annotation affordable — the annotator does not have to search
    a 10-minute video for a 4-second event."""
    rng = random.Random(seed)
    index = {os.path.basename(p): p
             for p in glob.glob(f"{root}/**/*.mp4", recursive=True)}
    by_class = defaultdict(list)

    for line in open(temporal_ann):
        p = line.split()
        if len(p) < 6:
            continue
        vid, src_cls = p[0], p[1]
        hint = UCF_HINTS.get(src_cls)
        if hint is None or vid not in index:
            continue
        s1, e1 = int(p[2]), int(p[3])
        if s1 < 0:
            continue                      # no annotated event window
        by_class[hint].append({
            "source": "UCF-Crime", "source_id": vid, "source_class": src_cls,
            "path": index[vid], "event_frames": [s1, e1],
        })

    out = []
    for cls, items in by_class.items():
        rng.shuffle(items)
        for it in items[:per_class]:
            # centre the clip on the annotated event; the annotator refines the
            # exact boundaries per the guide (§3)
            fps = 30.0                    # UCF-Crime is ~30fps; refined at extract
            mid = (it["event_frames"][0] + it["event_frames"][1]) / 2 / fps
            start = max(0.0, mid - clip_s / 2)
            out.append({**it, "intended_class": cls,
                        "clip_start_s": round(start, 2), "clip_len_s": clip_s,
                        # one video = one camera in UCF-Crime; no camera metadata
                        # exists, so the video IS the camera (declared limitation)
                        "camera_id": f"ucf::{os.path.splitext(it['source_id'])[0]}",
                        "scene_id": f"ucf::{it['source_class']}"})
    return out


def plan_le2i(root: str, per_class: int, clip_s: float, seed: int) -> list[dict]:
    """Le2i: fall. Real camera metadata exists (the environment folder), which is
    why Le2i is the one source where camera-disjoint splits are meaningful."""
    rng = random.Random(seed)
    vids = glob.glob(f"{root}/**/Videos/*.avi", recursive=True)
    rng.shuffle(vids)
    out = []
    for v in vids[:per_class]:
        env = os.path.basename(os.path.dirname(os.path.dirname(v)))
        out.append({
            "source": "Le2i", "source_id": os.path.basename(v),
            "source_class": "fall", "path": v, "intended_class": "fall",
            "clip_start_s": 0.0, "clip_len_s": clip_s,
            "camera_id": f"le2i::{env}", "scene_id": f"le2i::{env.rsplit('_',1)[0]}",
        })
    return out


def plan_generic(root: str, cls: str, per_class: int, clip_s: float,
                 seed: int, source_name: str) -> list[dict]:
    """Any folder of videos assigned to one candidate class (e.g. ABODA for
    suspicious_object, or a CC-licensed collection for suspicious_behavior)."""
    rng = random.Random(seed)
    vids = sorted(glob.glob(f"{root}/**/*.mp4", recursive=True)) + \
           sorted(glob.glob(f"{root}/**/*.avi", recursive=True))
    rng.shuffle(vids)
    out = []
    for v in vids[:per_class]:
        stem = os.path.splitext(os.path.basename(v))[0]
        out.append({
            "source": source_name, "source_id": os.path.basename(v),
            "source_class": cls, "path": v, "intended_class": cls,
            "clip_start_s": 0.0, "clip_len_s": clip_s,
            "camera_id": f"{source_name.lower()}::{stem}",
            "scene_id": f"{source_name.lower()}::{cls}",
        })
    return out


def cmd_plan(args):
    cfg = json.load(open(args.config)) if args.config.endswith(".json") else _yaml(args.config)
    plan, seed = [], int(cfg.get("seed", 0))
    clip_s = float(cfg.get("clip_len_s", 10))
    per_class = int(cfg.get("positives_per_class", 40))

    if "ucf" in cfg:
        plan += plan_ucf(cfg["ucf"]["root"], cfg["ucf"]["temporal_annotation"],
                         per_class, clip_s, seed)
    if "le2i" in cfg:
        plan += plan_le2i(cfg["le2i"]["root"], per_class, clip_s, seed)
    for extra in cfg.get("generic", []):
        plan += plan_generic(extra["root"], extra["class"], per_class, clip_s,
                             seed, extra["name"])

    counts = Counter(c["intended_class"] for c in plan)
    cams = len({c["camera_id"] for c in plan})
    print(f"planned clips: {len(plan)} across {cams} distinct cameras")
    for cls in CLASSES:
        n = counts.get(cls, 0)
        flag = "" if n >= per_class else f"   <-- SHORT of {per_class}"
        print(f"  {cls:22s} {n:3d}{flag}")
    missing = [c for c in CLASSES if counts.get(c, 0) == 0]
    if missing:
        print(f"\nNO SOURCE for: {missing}")
        print("These classes need a source, or must be declared absent in v1 of "
              "the benchmark. Do not ship a class with a handful of clips and "
              "call it covered.")
    json.dump({"seed": seed, "clip_len_s": clip_s, "clips": plan},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out} — REVIEW IT BY HAND before extracting")


# --------------------------------------------------------------------------
# EXTRACT
# --------------------------------------------------------------------------
def cmd_extract(args):
    plan = json.load(open(args.plan))
    os.makedirs(args.out, exist_ok=True)
    written, failed = 0, []

    for i, c in enumerate(plan["clips"]):
        clip_id = f"{c['intended_class']}__{c['source'].lower()}__{i:04d}"
        cdir = os.path.join(args.out, clip_id)
        os.makedirs(os.path.join(cdir, "frames"), exist_ok=True)

        # -an: source audio is irrelevant and, in some corpora, corrupt enough to
        # crash decoders (see ARCHITECTURE.md section 7)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-ss", str(c["clip_start_s"]), "-i", c["path"],
               "-t", str(c["clip_len_s"]), "-an", "-vsync", "0",
               "-start_number", "1", "-q:v", "2",
               os.path.join(cdir, "frames", "%05d.jpg")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        n_frames = len(glob.glob(os.path.join(cdir, "frames", "*.jpg")))
        if r.returncode != 0 or n_frames == 0:
            failed.append((clip_id, r.stderr.strip()[:120]))
            shutil.rmtree(cdir, ignore_errors=True)
            continue

        json.dump({**{k: v for k, v in c.items() if k != "path"},
                   "clip_id": clip_id, "n_frames": n_frames,
                   "source_sha256": sha256(c["path"])[:16],
                   # filled by the annotator, not by this script:
                   "classes": [], "hard_negative": [], "zones": [],
                   "annotation_status": "pending"},
                  open(os.path.join(cdir, "meta.json"), "w"), indent=1)
        written += 1
        if written % 25 == 0:
            print(f"  extracted {written}/{len(plan['clips'])}", flush=True)

    print(f"\nextracted {written} clips into {args.out}")
    if failed:
        print(f"FAILED {len(failed)}:")
        for cid, err in failed[:10]:
            print(f"  {cid}: {err}")


# --------------------------------------------------------------------------
# MANIFEST
# --------------------------------------------------------------------------
def cmd_manifest(args):
    """Build the manifest AFTER annotation, from each clip's meta.json.

    `classes` is written by the annotator (empty = negative clip). This is the
    file `build_splits.py` and `integrity_check.py` consume.
    """
    rows, pending = [], 0
    for meta_path in sorted(glob.glob(f"{args.root}/*/meta.json")):
        m = json.load(open(meta_path))
        if m.get("annotation_status") != "done":
            pending += 1
            continue
        rows.append({
            "clip_id": m["clip_id"], "camera_id": m["camera_id"],
            "scenario": m["scene_id"], "classes": m["classes"],
            "n_frames": m["n_frames"], "source": m["source"],
            "hard_negative": m.get("hard_negative", []),
        })
    json.dump(rows, open(args.out, "w"), indent=1)

    pos = sum(1 for r in rows if r["classes"])
    hard = sum(1 for r in rows if r["hard_negative"])
    print(f"manifest: {len(rows)} annotated clips "
          f"({pos} positive, {len(rows)-pos} negative, of which {hard} hard negatives)")
    if pending:
        print(f"  {pending} clips still pending annotation (excluded)")
    per_cls = Counter(c for r in rows for c in r["classes"])
    for cls in CLASSES:
        print(f"  {cls:22s} {per_cls.get(CLASSES.index(cls), 0):3d} clips")
    print(f"\nwrote {args.out} -> feed it to build_splits.py")


def _yaml(path):
    import yaml
    return yaml.safe_load(open(path))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="decide what to extract (touches no video)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="plan.json")
    p.set_defaults(func=cmd_plan)

    e = sub.add_parser("extract", help="cut the clips listed in a reviewed plan")
    e.add_argument("--plan", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_extract)

    m = sub.add_parser("manifest", help="build the manifest from annotated clips")
    m.add_argument("--root", required=True)
    m.add_argument("--out", default="manifest.json")
    m.set_defaults(func=cmd_manifest)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
