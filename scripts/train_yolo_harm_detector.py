"""
Reflections — YOLOv11n Custom Training Pipeline
================================================

Trains a YOLOv11-nano model to detect harmful visual content categories:
  - adult_nudity, explicit_body_part, underwear, bikini
  - blood, gore, dead_body
  - knife, gun, rifle, explosion
  - fire, drug, alcohol, cigarette

Uses Ultralytics API for training + exports to ONNX/TFLite for mobile deployment.

Usage:
    pip install ultralytics
    python train_yolo_harm_detector.py --data data/harm_dataset.yaml --epochs 100

Dataset YAML format (harm_dataset.yaml):
    path: ./data/harm_dataset
    train: images/train
    val: images/val
    test: images/test
    names:
      0: person
      1: adult_nudity
      2: underwear
      3: bikini
      4: explicit_body_part
      5: blood
      6: gore
      7: dead_body
      8: knife
      9: gun
      10: rifle
      11: explosion
      12: fire
      13: drug
      14: alcohol
      15: cigarette
"""

import argparse
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Reflections — YOLOv11n Harm Detector Training")
    parser.add_argument("--data", type=str, default="data/harm_dataset.yaml",
                        help="Path to dataset YAML config")
    parser.add_argument("--model", type=str, default="yolo11n.pt",
                        help="Base YOLO model (yolo11n.pt for nano)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Image size (320 for mobile speed)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="GPU device (0, 1, or cpu)")
    parser.add_argument("--project", type=str, default="runs/harm_detector", help="Output project dir")
    parser.add_argument("--name", type=str, default="yolo11n_harm", help="Run name")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--export-tflite", action="store_true", help="Export to TFLite after training")
    return parser.parse_args()


def create_sample_dataset_yaml(yaml_path):
    """Creates a sample dataset YAML if it doesn't exist."""
    content = """# Reflections Harm Detection Dataset
# Place your annotated images in the directory structure below.

path: ./data/harm_dataset
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: adult_nudity
  2: underwear
  3: bikini
  4: explicit_body_part
  5: blood
  6: gore
  7: dead_body
  8: knife
  9: gun
  10: rifle
  11: explosion
  12: fire
  13: drug
  14: alcohol
  15: cigarette

# Annotation format: YOLO (one .txt per image)
# Each line: class_id center_x center_y width height (all normalized 0-1)
"""
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"[Reflections] Created sample dataset YAML: {yaml_path}")
    print("[Reflections] Please add your annotated images before training.")


def train(args):
    """Train YOLOv11n on harm detection dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[Reflections] Installing ultralytics...")
        os.system("pip install ultralytics")
        from ultralytics import YOLO

    # Check if dataset YAML exists
    if not os.path.exists(args.data):
        print(f"[Reflections] Dataset YAML not found: {args.data}")
        create_sample_dataset_yaml(args.data)
        print("\n[Reflections] IMPORTANT: You need to prepare your dataset before training.")
        print("  1. Collect images for each harmful category")
        print("  2. Annotate them using Roboflow or CVAT (YOLO format)")
        print("  3. Place images in data/harm_dataset/images/train/ and /val/")
        print("  4. Place labels in data/harm_dataset/labels/train/ and /val/")
        print("  5. Run this script again")
        return None

    print(f"\n{'='*60}")
    print(f"  Reflections — YOLOv11n Harm Detector Training")
    print(f"{'='*60}")
    print(f"  Base Model:  {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Image Size:  {args.imgsz}px")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch Size:  {args.batch}")
    print(f"  Device:      {args.device}")
    print(f"{'='*60}\n")

    # Load pretrained YOLOv11n
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Optimization settings for mobile deployment
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,       # Final learning rate factor (cosine annealing)
        weight_decay=0.01,
        warmup_epochs=3,
        # Data augmentation (simulates real-world conditions)
        hsv_h=0.015,    # Hue augmentation
        hsv_s=0.7,      # Saturation augmentation
        hsv_v=0.4,      # Value augmentation
        degrees=10.0,   # Rotation
        translate=0.1,   # Translation
        scale=0.5,       # Scale
        flipud=0.1,      # Vertical flip (some content may be rotated)
        fliplr=0.5,      # Horizontal flip
        mosaic=1.0,      # Mosaic augmentation
        mixup=0.1,       # Mixup augmentation
        # Training configs
        patience=15,     # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        amp=True,        # Mixed precision training
        plots=True,
        verbose=True,
    )

    print(f"\n[Reflections] Training complete!")
    print(f"[Reflections] Best model: {args.project}/{args.name}/weights/best.pt")

    return model


def export_model(model, args):
    """Export trained model to ONNX and/or TFLite for deployment."""
    best_path = f"{args.project}/{args.name}/weights/best.pt"
    
    if not os.path.exists(best_path):
        print(f"[Reflections] Best weights not found: {best_path}")
        return

    from ultralytics import YOLO
    best_model = YOLO(best_path)

    if args.export_onnx:
        print("\n[Reflections] Exporting to ONNX...")
        best_model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=True,
            dynamic=False,
            opset=14
        )
        print("[Reflections] [OK] ONNX export complete")

    if args.export_tflite:
        print("\n[Reflections] Exporting to TensorFlow Lite (INT8 quantized)...")
        best_model.export(
            format="tflite",
            imgsz=args.imgsz,
            int8=True
        )
        print("[Reflections] [OK] TFLite INT8 export complete (ready for Android NNAPI)")


def validate(args):
    """Run validation on the trained model."""
    best_path = f"{args.project}/{args.name}/weights/best.pt"
    if not os.path.exists(best_path):
        print(f"[SafeScreen] No trained model found at {best_path}")
        return

    from ultralytics import YOLO
    model = YOLO(best_path)
    
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    print(f"\n{'='*60}")
    print(f"  Validation Results")
    print(f"{'='*60}")
    print(f"  mAP@0.5:      {results.box.map50:.4f}")
    print(f"  mAP@0.5:0.95:  {results.box.map:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    model = train(args)
    if model:
        export_model(model, args)
        validate(args)
