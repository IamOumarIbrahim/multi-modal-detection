# Model checkpoints

Weights (`*.pt`) are not tracked in git.

- `yolo11n.pt`: Ultralytics COCO-pretrained YOLO11n (2.6M parameters, 6.5 GFLOPs at 640×640). Both detectors start from it. Ultralytics downloads it automatically on first use.
- Trained detectors are written to `runs/detect/yolo11n_{rgb,thermal}_60e/weights/best.pt`.

```bash
python scripts/train_full_detectors.py --smoke   # 2-epoch check, times one epoch
python scripts/train_full_detectors.py           # full run, about 3 h on an RTX 4060
python scripts/extract_confidences.py            # rebuild the confidence cache from the new weights
```
