#!/usr/bin/env python3
"""Extract frames from the UCF-Crime TEST videos, scoped to the classes that
map onto the PSID-8 taxonomy (psid8/ucfcrime_mapping.json).

Why this script exists: the official Kaggle mirror ships full videos (not
pre-sampled frames), organized as:
    Anomaly-Videos-Part-1..4/<Category>/<video>_x264.mp4
    Normal_Videos_for_Event_Recognition/, Testing_Normal_Videos/, ...
    Temporal_Anomaly_Annotation_for_Testing_Videos.txt
The temporal annotation file ALREADY scopes the official test split (video
name, category, up to two anomaly windows) - Sultani et al.'s note. That means
we do not need to touch UCF_Crimes-Train-Test-Split at all: it is simpler and
more robust to (i) parse the annotation file, (ii) keep only rows whose
category maps to one of our 8 classes, (iii) locate each video by filename
under the four Anomaly-Videos-Part-* folders, and (iv) extract frames at a
stride, writing <out>/<video_stem>/NNNNN.jpg.

Time-boxed execution: use --limit-per-class to cap videos per mapped class
(e.g., 10) so a 3-day schedule stays realistic; declare the resulting subset
size explicitly in the paper if used.

Usage:
  python ucf_extract_test_frames.py \\
      --root /kaggle/input/ucf-crime-dataset \\
      --annotation /kaggle/input/ucf-crime-dataset/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \\
      --mapping psid8/ucfcrime_mapping.json \\
      --out /kaggle/working/ucf_test_frames \\
      --fps-stride 5 --limit-per-class 10
"""
import argparse
import glob
import json
import os


def parse_annotation(path):
    """Each row: video_name category start1 end1 start2 end2 (Sultani et al.
    convention; -1 means no second instance). Returns list of dicts."""
    rows = []
    for line in open(path):
        parts = line.split()
        if len(parts) < 6:
            continue
        video, category = parts[0], parts[1]
        nums = [int(x) for x in parts[2:6]]
        rows.append({"video": video, "category": category, "windows": nums})
    return rows


def find_video(search_roots, video_name):
    """video_name may or may not include an extension; search recursively."""
    stem = os.path.splitext(video_name)[0]
    for root in search_roots:
        hits = glob.glob(os.path.join(root, "**", stem + ".*"), recursive=True)
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dataset root containing Anomaly-Videos-Part-1..4 etc.")
    ap.add_argument("--annotation", required=True,
                    help="path to Temporal_Anomaly_Annotation_for_Testing_Videos.txt")
    ap.add_argument("--mapping", required=True,
                    help="path to psid8/ucfcrime_mapping.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps-stride", type=int, default=5,
                    help="keep 1 of every N frames (declare this in the paper)")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="0 = no limit; else cap videos per mapped class "
                        "(time-boxed runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="only report what would be extracted, no I/O")
    args = ap.parse_args()

    import cv2

    mapping = json.load(open(args.mapping))
    rows = parse_annotation(args.annotation)
    search_roots = sorted(glob.glob(os.path.join(args.root, "Anomaly-Videos-Part-*")))
    assert search_roots, f"No Anomaly-Videos-Part-* folders found under {args.root}"

    per_class_count = {}
    manifest = []
    for r in rows:
        tax = mapping.get(r["category"])
        if tax is None:
            continue  # Arrest/Explosion/Normal or anything unmapped: skipped
        n = per_class_count.get(tax, 0)
        if args.limit_per_class and n >= args.limit_per_class:
            continue
        video_path = find_video(search_roots, r["video"])
        if video_path is None:
            print(f"  WARNING: video not found for {r['video']} ({r['category']}); skipped.")
            continue
        per_class_count[tax] = n + 1
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(args.out, stem)
        manifest.append({"video": r["video"], "category": r["category"],
                         "taxonomy_class": tax, "windows": r["windows"],
                         "video_path": video_path, "frames_dir": out_dir})
        if args.dry_run:
            continue
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        f_idx, kept = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            f_idx += 1
            if (f_idx - 1) % args.fps_stride:
                continue
            cv2.imwrite(os.path.join(out_dir, f"{f_idx:05d}.jpg"), frame)
            kept += 1
        cap.release()
        print(f"  {stem} [{tax}]: {kept} frames extracted (stride={args.fps_stride})")

    os.makedirs(args.out, exist_ok=True)
    json.dump(manifest, open(os.path.join(args.out, "extraction_manifest.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Videos selected: {len(manifest)}")
    print("Per mapped class:", per_class_count)
    print(f"Manifest: {os.path.join(args.out, 'extraction_manifest.json')}")
    print("Feed CFG['ucf_frames'] = this --out path and CFG['ucf_temporal'] = "
          "--annotation path into notebook cell [6b].")


if __name__ == "__main__":
    main()
