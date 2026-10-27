"""Train YOLOv8n on the PPE dataset. Example:
python train.py --data /path/to/PPE-Detection-with-Gloves-9/data.yaml
"""
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="PPE-Detection-with-Gloves-9/data.yaml")
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/ppe_baseline")
    p.add_argument("--name", default="train")
    a = p.parse_args()
    data = Path(a.data).expanduser().resolve()
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}. Download/extract the dataset and pass its correct path with --data.")
    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.project, name=a.name)
    print("Training complete. Check the run folder for weights/best.pt and weights/last.pt.")

if __name__ == "__main__":
    main()
