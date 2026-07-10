#!/usr/bin/env python3
"""Convert a COCO export (CVAT) to the PSID-8 YOLO/Ultralytics layout.

- Boxes become normalized `class cx cy w h` lines (one line per class of the
  box, replicating the box for each class: documented multi-label convention).
- Preserves event_id and compositional attributes in a per-image JSON sidecar
  (same stem, .attrs.json), because the YOLO format cannot carry them and
  SENTRY uses them for tubes/evaluation.

Usage: python coco_to_yolo.py coco.json images_dir out_dir
"""
import json, os, sys
from collections import defaultdict

def main(coco_path, images_dir, out_dir):
    coco = json.load(open(coco_path))
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)
    imgs = {im["id"]: im for im in coco["images"]}
    per_img = defaultdict(list)
    for a in coco["annotations"]:
        per_img[a["image_id"]].append(a)
    for img_id, anns in per_img.items():
        im = imgs[img_id]
        W, H = im["width"], im["height"]
        stem = os.path.splitext(os.path.basename(im["file_name"]))[0]
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
        open(os.path.join(out_dir, "labels", stem + ".txt"), "w").write("\n".join(lines))
        json.dump(attrs, open(os.path.join(out_dir, "labels", stem + ".attrs.json"), "w"),
                  ensure_ascii=False)
    print(f"Converted {len(per_img)} images into {out_dir}/labels")

if __name__ == "__main__":
    main(*sys.argv[1:4])
