#!/usr/bin/env python3
"""Convert a COCO export (CVAT) to a PSID-8 YOLO/Ultralytics layout.

- Boxes become normalized `class cx cy w h` lines (one line per class of the
  box, replicating the box for each class: documented multi-label convention).
- Preserves event_id and compositional attributes in a per-image JSON sidecar
  (same stem, .attrs.json), because the YOLO format cannot carry them and
  SENTRY uses them for tubes/evaluation.

Two output layouts are supported, because two different workflows produce
frames differently:

  --layout clips (default): the canonical video-clip contract used by the main
    SENTRY pipeline (see ARCHITECTURE.md section 3): frames already exist on
    disk at `<clips_root>/<clip_id>/frames/*.jpg` (written by curate_clips.py
    or le2i_to_clips.py). This mode writes labels to
    `<clips_root>/<clip_id>/labels/*.txt` + `.attrs.json`, next to those
    frames, and VALIDATES that every image referenced by the COCO export
    actually exists on disk at that path (catching a desynchronized dataset
    immediately, instead of silently producing labels nothing can find).

  --layout flat: the original single-folder image-pilot workflow (PSID-8-Im):
    `images_dir` holds loose images with no clip structure. This mode copies
    (or symlinks) those images into `<out_dir>/images/` and writes labels into
    `<out_dir>/labels/`, mirroring the standard Ultralytics images/+labels/
    pair. This is the legacy behavior, preserved for that workflow.

Before this fix, `images_dir` was accepted as a parameter but never read: no
image was ever copied or validated, and (for the clips layout in particular)
the output directory structure did not match what the rest of the pipeline
(sentry/data.py::VideoClipDataset, psid8/scripts/build_splits.py) expects,
so a converted dataset could silently have zero locatable labels.

Usage:
  python coco_to_yolo.py coco.json images_dir out_dir --layout clips [--clip-id ID] [--copy]
  python coco_to_yolo.py coco.json images_dir out_dir --layout flat [--copy]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict


def _load_annotations(coco_path: str):
    """Shared parsing: returns (images_by_id, annotations_by_image_id)."""
    coco = json.load(open(coco_path))
    imgs = {im["id"]: im for im in coco["images"]}
    per_img = defaultdict(list)
    for a in coco["annotations"]:
        per_img[a["image_id"]].append(a)
    return imgs, per_img


def _yolo_lines_and_attrs(im: dict, anns: list[dict]):
    """Build the YOLO label lines and the .attrs.json sidecar payload for one
    image's annotations. Shared by both layouts so the two never diverge in
    how a box is encoded."""
    W, H = im["width"], im["height"]
    lines, attrs = [], []
    for a in anns:
        x, y, w, h = a["bbox"]
        cx, cy = (x + w / 2) / W, (y + h / 2) / H
        class_ids = a.get("attributes", {}).get("class_ids") or [a["category_id"]]
        for c in class_ids:
            lines.append(f"{c} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        attrs.append({"bbox": [x, y, x + w, y + h],
                      "class_ids": class_ids,
                      "event_id": a.get("attributes", {}).get("event_id"),
                      "attributes": a.get("attributes", {})})
    return lines, attrs


def _place_image(src: str, dst: str, copy: bool):
    """Symlink (default, fast, no media duplication) or copy an image file."""
    if os.path.exists(dst) or os.path.islink(dst):
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def convert_clip(coco_path: str, frames_dir: str, clips_root: str,
                 clip_id: str | None = None, copy: bool = False) -> str:
    """--layout clips: write labels next to a clip's existing frames.

    `frames_dir` is the directory of already-extracted frames for ONE clip
    (e.g. `/data/psid8_raw/<clip_id>/frames`). `clip_id` is inferred from the
    parent folder name when not given. Every image the COCO export references
    is checked against `frames_dir`; a missing image raises immediately rather
    than silently producing an unusable dataset.
    """
    if clip_id is None:
        parent = os.path.dirname(os.path.normpath(frames_dir))
        clip_id = os.path.basename(parent)

    imgs, per_img = _load_annotations(coco_path)
    labels_dir = os.path.join(clips_root, clip_id, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    missing = []
    for img_id, anns in per_img.items():
        im = imgs[img_id]
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
        src_image = os.path.join(frames_dir, os.path.basename(im["file_name"]))
        if not os.path.exists(src_image):
            missing.append(src_image)
            continue
        lines, attrs = _yolo_lines_and_attrs(im, anns)
        open(os.path.join(labels_dir, stem + ".txt"), "w").write("\n".join(lines))
        json.dump(attrs, open(os.path.join(labels_dir, stem + ".attrs.json"), "w"),
                  ensure_ascii=False)

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by the COCO export were not "
            f"found under {frames_dir}: {missing[:5]}"
            + (" (and more)" if len(missing) > 5 else "")
            + ". Labels and frames are out of sync; fix this before training."
        )

    print(f"[clips] {clip_id}: converted {len(per_img)} images "
          f"-> {labels_dir}")
    return labels_dir


def convert_flat(coco_path: str, images_dir: str, out_dir: str,
                 copy: bool = False) -> str:
    """--layout flat: legacy single-folder image-pilot workflow.

    Copies or symlinks the referenced images into `<out_dir>/images/` and
    writes labels into `<out_dir>/labels/`, mirroring the Ultralytics
    images/+labels/ pair. Every image referenced by the COCO export is
    validated against `images_dir`; missing images raise immediately.
    """
    imgs, per_img = _load_annotations(coco_path)
    images_out = os.path.join(out_dir, "images")
    labels_out = os.path.join(out_dir, "labels")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    # Iterate over ALL images declared in the COCO export, not only the ones
    # that have at least one annotation. `per_img` (built from `annotations`)
    # never contains an image with zero boxes, so an earlier version of this
    # loop (iterating `per_img.items()`) silently dropped every background
    # image: hard negatives (images with a deliberate "nothing here" label)
    # never got copied into `images_out` at all, which is data loss for
    # exactly the hard-negative images the annotation guide asks to collect.
    missing, placed = [], 0
    for img_id, im in imgs.items():
        anns = per_img.get(img_id, [])
        fname = os.path.basename(im["file_name"])
        stem = os.path.splitext(fname)[0]
        src_image = os.path.join(images_dir, fname)
        if not os.path.exists(src_image):
            missing.append(src_image)
            continue
        _place_image(src_image, os.path.join(images_out, fname), copy)
        placed += 1
        if anns:
            lines, attrs = _yolo_lines_and_attrs(im, anns)
            open(os.path.join(labels_out, stem + ".txt"), "w").write("\n".join(lines))
            json.dump(attrs, open(os.path.join(labels_out, stem + ".attrs.json"), "w"),
                      ensure_ascii=False)
        # else: background image, correctly copied with no label file

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by the COCO export were not "
            f"found under {images_dir}: {missing[:5]}"
            + (" (and more)" if len(missing) > 5 else "")
            + ". Images and labels would be out of sync; fix this before training."
        )

    print(f"[flat] converted {placed} images -> {images_out} + {labels_out}")
    return out_dir


def main(coco_path: str, images_dir: str, out_dir: str,
        layout: str = "clips", clip_id: str | None = None,
        copy: bool = False):
    """Backward-compatible entry point (kept as `main(coco, images_dir, out)`
    for any existing positional-argument caller). `layout` defaults to
    "clips", the layout the active video pipeline consumes; pass
    layout="flat" for the legacy single-folder image-pilot workflow."""
    if layout == "clips":
        return convert_clip(coco_path, images_dir, out_dir, clip_id=clip_id, copy=copy)
    elif layout == "flat":
        return convert_flat(coco_path, images_dir, out_dir, copy=copy)
    raise ValueError(f"unknown layout {layout!r}; use 'clips' or 'flat'")


def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coco_path")
    ap.add_argument("images_dir", help="clips layout: the clip's frames dir; "
                                       "flat layout: the loose-images dir")
    ap.add_argument("out_dir", help="clips layout: the clips root; "
                                    "flat layout: the output dataset dir")
    ap.add_argument("--layout", choices=["clips", "flat"], default="clips")
    ap.add_argument("--clip-id", default=None,
                    help="clips layout only; inferred from images_dir's parent "
                         "folder name when omitted")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (flat layout only)")
    args = ap.parse_args()
    main(args.coco_path, args.images_dir, args.out_dir,
         layout=args.layout, clip_id=args.clip_id, copy=args.copy)


if __name__ == "__main__":
    _cli()
