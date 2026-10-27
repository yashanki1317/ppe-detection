"""Evaluate trained weights. Example:
python evaluate.py --weights runs/ppe_baseline/train/weights/best.pt --data /path/to/data.yaml --split test
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="PPE-Detection-with-Gloves-9/data.yaml")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()
    weights, data = Path(a.weights).expanduser().resolve(), Path(a.data).expanduser().resolve()
    if not weights.exists(): raise FileNotFoundError(f"Weights not found: {weights}")
    if not data.exists(): raise FileNotFoundError(f"data.yaml not found: {data}")
    metrics = YOLO(str(weights)).val(data=str(data), split=a.split, imgsz=a.imgsz, batch=a.batch)
    print(f"Split: {a.split}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP@50: {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Per-class AP@50: {metrics.box.ap50}")

if __name__ == "__main__":
    main()
