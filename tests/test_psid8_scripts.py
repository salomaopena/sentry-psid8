#!/usr/bin/env python3
"""Tests for psid8/scripts/coco_to_yolo.py (both layouts) and, where relevant,
the interaction with sentry/data.py::VideoClipDataset and
psid8/scripts/integrity_check.py.

Uses only dependencies already declared in requirements.txt/pyproject.toml
(numpy, opencv-python); no new dependency is introduced for testing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "psid8", "scripts", "coco_to_yolo.py")


def _load_module():
    """Import coco_to_yolo.py by path (it lives outside any package's __init__
    tree, as a standalone CLI script)."""
    spec = importlib.util.spec_from_file_location("coco_to_yolo", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_jpg(path: str, size=(48, 64)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.imwrite(path, img)


def test_coco_to_yolo_clips_layout_happy_path():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as d:
        clip_id = "bench11"
        frames_dir = os.path.join(d, "clips_root", clip_id, "frames")
        for i in (1, 2):
            _write_jpg(os.path.join(frames_dir, f"{i:05d}.jpg"))

        coco = {
            "images": [{"id": 1, "file_name": "00001.jpg", "width": 64, "height": 48},
                      {"id": 2, "file_name": "00002.jpg", "width": 64, "height": 48}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 6, "bbox": [10, 10, 20, 20],
                 "attributes": {"class_ids": [6], "event_id": "ev1"}},
            ],
        }
        coco_path = os.path.join(d, "instances_default.json")
        json.dump(coco, open(coco_path, "w"))

        labels_dir = mod.main(coco_path, frames_dir, os.path.join(d, "clips_root"),
                              layout="clips")

        assert labels_dir == os.path.join(d, "clips_root", clip_id, "labels")
        assert os.path.exists(os.path.join(labels_dir, "00001.txt"))
        assert os.path.exists(os.path.join(labels_dir, "00001.attrs.json"))
        # frame 2 has no annotation: no label files for it, and that's correct
        # (an unlabeled frame is background, not an error)
        assert not os.path.exists(os.path.join(labels_dir, "00002.txt"))

        line = open(os.path.join(labels_dir, "00001.txt")).read().strip()
        cls, cx, cy, w, h = line.split()
        assert cls == "6"
        assert abs(float(cx) - 0.3125) < 1e-6
        assert abs(float(cy) - 0.4167) < 1e-3

        attrs = json.load(open(os.path.join(labels_dir, "00001.attrs.json")))
        assert attrs[0]["event_id"] == "ev1"
        assert attrs[0]["class_ids"] == [6]
        print("test_coco_to_yolo_clips_layout_happy_path OK")


def test_coco_to_yolo_clips_layout_infers_clip_id():
    """clip_id, when omitted, must be the frames directory's parent folder
    name, matching curate_clips.py's own naming (clip_id/frames/*.jpg)."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as d:
        frames_dir = os.path.join(d, "clips_root", "intrusion__ucf__0007", "frames")
        _write_jpg(os.path.join(frames_dir, "00001.jpg"))
        coco = {"images": [{"id": 1, "file_name": "00001.jpg", "width": 64, "height": 48}],
                "annotations": []}
        coco_path = os.path.join(d, "coco.json")
        json.dump(coco, open(coco_path, "w"))

        labels_dir = mod.main(coco_path, frames_dir, os.path.join(d, "clips_root"))
        assert os.path.basename(os.path.dirname(labels_dir)) == "intrusion__ucf__0007"
        print("test_coco_to_yolo_clips_layout_infers_clip_id OK")


def test_coco_to_yolo_clips_layout_missing_image_raises():
    """A COCO export referencing a frame that does not exist on disk must fail
    loudly. Before the fix, coco_to_yolo.py never touched `images_dir` at all,
    so this desynchronization was invisible until training silently saw zero
    labeled frames."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as d:
        frames_dir = os.path.join(d, "clips_root", "clipX", "frames")
        os.makedirs(frames_dir, exist_ok=True)   # deliberately empty: no frames written
        coco = {"images": [{"id": 1, "file_name": "00099.jpg", "width": 64, "height": 48}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 6,
                                 "bbox": [1, 1, 2, 2], "attributes": {}}]}
        coco_path = os.path.join(d, "coco.json")
        json.dump(coco, open(coco_path, "w"))

        try:
            mod.main(coco_path, frames_dir, os.path.join(d, "clips_root"))
        except FileNotFoundError as e:
            assert "00099.jpg" in str(e)
        else:
            raise AssertionError("a missing referenced frame must raise, not silently skip")
        print("test_coco_to_yolo_clips_layout_missing_image_raises OK")


def test_coco_to_yolo_flat_layout_legacy_workflow():
    """--layout flat: the original loose-images (PSID-8-Im pilot) workflow.
    Images are placed into out_dir/images/ and labels into out_dir/labels/,
    the Ultralytics-standard pair; images_dir is genuinely used here."""
    mod = _load_module()
    with tempfile.TemporaryDirectory() as d:
        images_dir = os.path.join(d, "loose_images")
        for name in ("imgA.jpg", "imgB.jpg"):
            _write_jpg(os.path.join(images_dir, name), size=(80, 100))
        coco = {
            "images": [{"id": 1, "file_name": "imgA.jpg", "width": 100, "height": 80},
                      {"id": 2, "file_name": "imgB.jpg", "width": 100, "height": 80}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 3, "bbox": [5, 5, 10, 10],
                 "attributes": {}},
            ],
        }
        coco_path = os.path.join(d, "coco.json")
        json.dump(coco, open(coco_path, "w"))
        out_dir = os.path.join(d, "out")

        mod.main(coco_path, images_dir, out_dir, layout="flat")

        assert os.path.exists(os.path.join(out_dir, "images", "imgA.jpg"))
        assert os.path.exists(os.path.join(out_dir, "images", "imgB.jpg"))
        assert os.path.exists(os.path.join(out_dir, "labels", "imgA.txt"))
        # imgB has no annotation: no label file, correctly absent
        assert not os.path.exists(os.path.join(out_dir, "labels", "imgB.txt"))
        print("test_coco_to_yolo_flat_layout_legacy_workflow OK")


def test_coco_to_yolo_flat_layout_copy_flag_makes_real_files():
    mod = _load_module()
    with tempfile.TemporaryDirectory() as d:
        images_dir = os.path.join(d, "loose_images")
        _write_jpg(os.path.join(images_dir, "imgA.jpg"))
        coco = {"images": [{"id": 1, "file_name": "imgA.jpg", "width": 64, "height": 48}],
                "annotations": []}
        coco_path = os.path.join(d, "coco.json")
        json.dump(coco, open(coco_path, "w"))
        out_dir = os.path.join(d, "out")

        mod.main(coco_path, images_dir, out_dir, layout="flat", copy=True)
        placed = os.path.join(out_dir, "images", "imgA.jpg")
        assert os.path.exists(placed) and not os.path.islink(placed)
        print("test_coco_to_yolo_flat_layout_copy_flag_makes_real_files OK")


def test_coco_to_yolo_output_is_consumable_by_video_clip_dataset():
    """The real integration test: the fixed script's output must be directly
    loadable by sentry.data.VideoClipDataset. This is the exact failure mode
    the audit flagged (images/labels desynchronized from the clip contract)."""
    try:
        import torch
    except ImportError:
        print("test_coco_to_yolo_output_is_consumable_by_video_clip_dataset "
              "SKIPPED (torch not installed)")
        return
    print(f"  (torch {torch.__version__} available; running the real integration test)")

    from sentry.data import VideoClipDataset

    mod = _load_module()
    with tempfile.TemporaryDirectory() as d:
        clips_root = os.path.join(d, "clips_root")
        frames_dir = os.path.join(clips_root, "clipA", "frames")
        for i in (1, 2):
            _write_jpg(os.path.join(frames_dir, f"{i:05d}.jpg"))
        coco = {
            "images": [{"id": 1, "file_name": "00001.jpg", "width": 64, "height": 48},
                      {"id": 2, "file_name": "00002.jpg", "width": 64, "height": 48}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 6, "bbox": [10, 10, 20, 20],
                 "attributes": {"class_ids": [6]}},
            ],
        }
        coco_path = os.path.join(d, "coco.json")
        json.dump(coco, open(coco_path, "w"))
        mod.main(coco_path, frames_dir, clips_root, layout="clips")

        ds = VideoClipDataset(clips_root, ["clipA"], window=2, stride=1)
        assert len(ds) > 0, ("VideoClipDataset found no samples: the clips-layout "
                             "output is not aligned with what it expects")
        item = ds[0]
        assert item["clip_id"] == "clipA"
        assert item["images"].shape[0] == 2
        assert item["labels"][0].shape[0] == 1
        print("test_coco_to_yolo_output_is_consumable_by_video_clip_dataset OK")


if __name__ == "__main__":
    test_coco_to_yolo_clips_layout_happy_path()
    test_coco_to_yolo_clips_layout_infers_clip_id()
    test_coco_to_yolo_clips_layout_missing_image_raises()
    test_coco_to_yolo_flat_layout_legacy_workflow()
    test_coco_to_yolo_flat_layout_copy_flag_makes_real_files()
    test_coco_to_yolo_output_is_consumable_by_video_clip_dataset()
    print("\nALL psid8/scripts/coco_to_yolo.py TESTS PASSED")
