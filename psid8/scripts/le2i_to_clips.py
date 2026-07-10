#!/usr/bin/env python3
"""Convert the Le2i Fall Detection dataset (Kaggle: tuyenldvn/falldataset-imvia)
into the SENTRY clips layout: clips/<clip_id>/{frames,labels}.

Le2i structure (per environment folder, e.g. Coffee_room_01):
  Videos/video (i).avi
  Annotation_files/video (i).txt
Per the official README, each annotation file contains: the frame number of the
beginning of the fall, the frame number of the end of the fall, then, for each
frame, the bounding box. MIRRORS DIFFER in the per-frame column order, so this
script supports two formats and a --preview mode that renders boxes on sample
frames for VISUAL VERIFICATION before converting anything:

  --bbox-format corners    per-frame line: frame,label,x1,y1,x2,y2   (default)
  --bbox-format center_hw  per-frame line: frame,label,h,w,cx,cy

Run --preview FIRST, inspect preview_*.jpg, and only then convert. If boxes look
transposed or offset, switch the format flag.

Labels are written in YOLO format with class id 6 (fall) ONLY for frames inside
the [fall_start, fall_end] interval; frames with a person outside that interval
get no label (background), which encodes the "lying without transition is not a
fall" rule at the frame level. Fall-free videos (start=end=0) become negative
clips. A manifest entry per clip is emitted for build_splits.py, with
camera_id = environment folder (the closest available proxy for a camera).

Usage:
  python le2i_to_clips.py --root /kaggle/input/falldataset-imvia \
      --out /kaggle/working/le2i_clips [--preview 6] [--bbox-format corners]
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess

# Quiet harmless FFmpeg audio warnings and avoid threaded-decode heap issues
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


def parse_annotation(path, bbox_format):
    lines = [l.strip() for l in open(path) if l.strip()]
    fall_start, fall_end = int(lines[0].split(",")[0]), int(lines[1].split(",")[0])
    boxes = {}
    for l in lines[2:]:
        parts = [int(float(x)) for x in re.split(r"[,\s]+", l) if x != ""]
        if len(parts) < 6:
            continue
        f = parts[0]
        if bbox_format == "corners":
            _, x1, y1, x2, y2 = parts[1:6]
        else:  # center_hw: frame,label,h,w,cx,cy
            _, h, w, cx, cy = parts[1:6]
            x1, y1, x2, y2 = cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2
        if x2 > x1 and y2 > y1:
            boxes[f] = (x1, y1, x2, y2)
    return fall_start, fall_end, boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox-format", choices=["corners", "center_hw"],
                    default="corners")
    ap.add_argument("--preview", type=int, default=0,
                    help="render N sample frames with boxes and exit")
    ap.add_argument("--fps-keep", type=int, default=1,
                    help="keep 1 of every N frames (1 = all)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="clip_ids to skip (e.g. corrupted videos)")
    args = ap.parse_args()

    import cv2   # used ONLY on JPEGs (safe); video decoding is done by ffmpeg CLI
    assert shutil.which("ffmpeg"), "ffmpeg binary not found on PATH"

    def ffmpeg_extract_all(video, out_dir, q=2):
        """Extract every frame as out_dir/%05d.jpg (1-based, matches the
        1-based frame indices of the Le2i annotations). Audio is dropped with
        -an, which sidesteps the corrupt MP3 streams that crash OpenCV."""
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-i", video, "-an", "-vsync", "0", "-start_number", "1",
               "-q:v", str(q), os.path.join(out_dir, "%05d.jpg")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip()[:300])
        return len(glob.glob(os.path.join(out_dir, "*.jpg")))

    def ffmpeg_extract_one(video, frame_idx_1based, out_jpg):
        """Extract a single frame via ffmpeg (select is 0-based)."""
        n0 = max(0, frame_idx_1based - 1)
        cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
               "-i", video, "-an", "-vf", f"select=eq(n\\,{n0})",
               "-vframes", "1", "-vsync", "0", "-y", out_jpg]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(out_jpg)

    manifest = []
    skipped = []
    previews_left = args.preview
    # Tolerant discovery: mirrors vary in nesting depth (Env/Env/...) and in the
    # folder name ("Annotation_files" vs "Annotations_files").
    ann_files = sorted(glob.glob(os.path.join(args.root, "**", "Annotation*files", "*.txt"),
                                 recursive=True))
    assert ann_files, "No Annotation*_files folders found under --root; check the dataset path."
    print(f"Found {len(ann_files)} annotated videos.")

    for ann in ann_files:
        env = os.path.basename(os.path.dirname(os.path.dirname(ann)))
        vid_name = os.path.splitext(os.path.basename(ann))[0]
        env_dir = os.path.dirname(os.path.dirname(ann))
        video_path = os.path.join(env_dir, "Videos", vid_name + ".avi")
        if not os.path.exists(video_path):
            cands = (glob.glob(os.path.join(env_dir, "Video*", vid_name + ".*"))
                     or glob.glob(os.path.join(env_dir, "**", vid_name + ".avi"),
                                  recursive=True))
            if not cands:
                print(f"  WARNING: video missing for {ann}; skipped.")
                continue
            video_path = cands[0]

        fall_start, fall_end, boxes = parse_annotation(ann, args.bbox_format)
        # Le2i convention quirk: no-fall (ADL) videos may carry a dummy interval
        # (e.g., [1,2] or [0,0]) instead of a real one. A clip counts as a fall
        # ONLY if at least one annotated box lies inside the interval.
        has_fall = fall_end > 0 and any(fall_start <= f <= fall_end for f in boxes)
        clip_id = f"{env}__{vid_name}".replace(" ", "_").replace("(", "").replace(")", "")
        if fall_end > 0 and not has_fall:
            print(f"  note: {clip_id}: dummy fall interval [{fall_start},{fall_end}] "
                  f"with no boxes inside -> treated as NEGATIVE (ADL)")
        if args.skip and clip_id in args.skip:
            print(f"  {clip_id}: skipped by --skip"); skipped.append(clip_id); continue
        print(f"  processing {clip_id} ...", flush=True)

        if args.preview and previews_left > 0:
            if not has_fall:
                continue                      # preview only on true fall videos
            mid = (fall_start + fall_end) // 2
            outp = f"preview_{clip_id}_f{mid}.jpg"
            ok = ffmpeg_extract_one(video_path, mid, outp)
            if ok and mid in boxes:
                frame = cv2.imread(outp)
                x1, y1, x2, y2 = boxes[mid]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.imwrite(outp, frame)
                print(f"  preview written: {outp}  (verify the box covers the person)")
                previews_left -= 1
            elif not ok:
                print(f"  WARNING: ffmpeg failed on {clip_id}; add to --skip if it repeats")
            if previews_left == 0:
                print("Preview done. Inspect the images, pick --bbox-format, then rerun without --preview.")
                return
            continue

        frames_dir = os.path.join(args.out, "clips", clip_id, "frames")
        labels_dir = os.path.join(args.out, "clips", clip_id, "labels")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        try:
            n_frames = ffmpeg_extract_all(video_path, frames_dir)
        except Exception as e:
            print(f"  WARNING: ffmpeg failed on {clip_id}: {e}")
            skipped.append(clip_id)
            shutil.rmtree(os.path.dirname(frames_dir), ignore_errors=True)
            continue
        if n_frames == 0:
            print(f"  WARNING: 0 frames extracted from {clip_id}; skipping")
            skipped.append(clip_id)
            shutil.rmtree(os.path.dirname(frames_dir), ignore_errors=True)
            continue
        first = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[0]
        H, W = cv2.imread(first).shape[:2]
        kept = 0
        for fp in sorted(glob.glob(os.path.join(frames_dir, "*.jpg"))):
            f_idx = int(os.path.splitext(os.path.basename(fp))[0])
            if (f_idx - 1) % args.fps_keep:
                os.remove(fp)
                continue
            stem = f"{f_idx:05d}"
            in_fall = has_fall and fall_start <= f_idx <= fall_end
            if in_fall and f_idx in boxes:
                x1, y1, x2, y2 = boxes[f_idx]
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                open(os.path.join(labels_dir, stem + ".txt"), "w").write(
                    f"6 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            kept += 1
        manifest.append({"clip_id": clip_id, "camera_id": env, "scenario": env.rsplit("_", 1)[0],
                         "classes": [6] if has_fall else [], "n_frames": kept})
        print(f"  {clip_id}: {kept} frames, fall=[{fall_start},{fall_end}]")

    os.makedirs(args.out, exist_ok=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    n_pos = sum(1 for m in manifest if m["classes"])
    print(f"\nDone: {len(manifest)} clips ({n_pos} falls, {len(manifest) - n_pos} negatives).")
    if skipped:
        print(f"Skipped {len(skipped)} clips (report in the paper): {skipped}")
    print(f"Manifest: {os.path.join(args.out, 'manifest.json')} -> feed it to build_splits.py")


if __name__ == "__main__":
    main()
