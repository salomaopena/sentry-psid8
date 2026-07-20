# Kaggle workflow (2x T4)

```bash
pip install "ultralytics==8.<pin>" && python -c "import ultralytics, torch; print(ultralytics.__version__, torch.__version__)"
# 0) datasets mounted under /kaggle/input (Le2i fall, fire videos, UCF-Crime test, ABODA)
python psid8/scripts/le2i_to_clips.py --help      # convert Le2i -> clips layout (run --preview first!)
python psid8/scripts/build_splits.py manifest.json --out splits.json --seed 0
python psid8/scripts/integrity_check.py manifest.json splits.json   # Phase 1

# 1) Stage A - frame-level (also produces the YOLOv8/YOLO11 baselines of Table II)
yolo detect train data=data.yaml model=yolov8m.pt imgsz=640 epochs=100 seed=0 device=0,1

# 2) Stage B - temporal
python -m sentry.train --weights runs/detect/train/weights/best.pt \
  --clips-dir data/clips --splits splits.json --window 8 --seed 0

# 3) Per-class thresholds on validation + streaming inference -> alerts + event eval
python -m sentry.eval --alerts alerts_val.json --gt-events gt_val.json

# 4) Latency/FPS: torch.cuda.Event on 1x T4, batch=1, half=True,
#    50 warm-up iterations, then mean over 500 frames.
```text
Notes: (i) pin and record ALL versions; (ii) export W&B logs into the repo;
(iii) the test split is only touched in Phase 5 of PREREGISTRATION.md.
