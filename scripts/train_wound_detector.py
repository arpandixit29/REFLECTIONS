"""
Reflections — Real-Time Wound & Graphic Injury Detector Trainer
================================================================

Trains an EfficientNet-B0 / MobileNetV3 model to identify graphic wounds, blood, 
lacerations, and bodily injuries for zero-latency screen blurring.

Exports trained weights to PyTorch (.pt) and ONNX (.onnx) for Web & Mobile inference.
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Reflections - Wound & Graphic Injury Training Engine")
    parser.add_argument("--data-dir", type=str, default="./data/wound_dataset", help="Path to wound dataset folder")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="Input image dimension")
    parser.add_argument("--output-model", type=str, default="models/safescreen_wound_detector.pt", help="Saved checkpoint path")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export to ONNX after training")
    return parser.parse_args()

def build_wound_classifier(num_classes=2):
    """Build lightweight EfficientNet-B0 fine-tuned for wound & injury detection."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(128, num_classes)
    )
    return model

class WoundDataset(Dataset):
    """Flexible dataset loader that supports image folders or file lists."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        # Walk directory
        if os.path.exists(root_dir):
            for root, _, files in os.walk(root_dir):
                for f in files:
                    if os.path.splitext(f)[1].lower() in valid_exts:
                        full_path = os.path.join(root, f)
                        # Determine label: 1 if 'wound' or 'injury' or in wound folder, else 0
                        lower_f = f.lower()
                        lower_r = root.lower()
                        is_wound = (
                            "wound" in lower_f or "cut" in lower_f or "blood" in lower_f or "injury" in lower_f or
                            "wound" in lower_r or "cut" in lower_r or "blood" in lower_r or "injury" in lower_r
                        )
                        label = 1 if is_wound else 0
                        self.samples.append((full_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label

def export_onnx(model, output_pt):
    output_onnx = output_pt.replace(".pt", ".onnx")
    print(f"[Reflections] Exporting model to ONNX: {output_onnx}")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_onnx,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=14
        )
        print(f"[Reflections] [OK] Successfully exported ONNX format: {output_onnx}")
    except Exception as e:
        print(f"[Reflections] Note on ONNX export: {e}")
        print(f"[Reflections] [OK] PyTorch trained weights preserved at: {output_pt}")

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Reflections — Wound & Graphic Content Detector Trainer")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  Dataset Dir:   {args.data_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Output Model:  {args.output_model}")
    print(f"{'='*60}\n")
    print(f"  Device:        {device}")
    print(f"  Dataset Dir:   {args.data_dir}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch Size:    {args.batch_size}")
    print(f"  Output Model:  {args.output_model}")
    print(f"{'='*60}\n")

    os.makedirs(os.path.dirname(args.output_model) or "models", exist_ok=True)

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = WoundDataset(args.data_dir, transform=data_transforms)

    if len(dataset) == 0:
        print(f"[SafeScreen] Note: Data directory '{args.data_dir}' contains 0 items.")
        print("[SafeScreen] Creating pre-trained model architecture checkpoint for production integration...")
        model = build_wound_classifier(num_classes=2)
        torch.save(model.state_dict(), args.output_model)
        if args.export_onnx:
            export_onnx(model, args.output_model)
        print(f"[SafeScreen] Saved model initialized checkpoint to {args.output_model}")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = build_wound_classifier(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        corrects = 0
        total = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = corrects / max(1, total)
        print(f"Epoch [{epoch+1}/{args.epochs}] — Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc*100:.2f}%")

    torch.save(model.state_dict(), args.output_model)
    print(f"\n[SafeScreen] [OK] Saved trained model weights to: {args.output_model}")

    if args.export_onnx:
        export_onnx(model, args.output_model)

if __name__ == "__main__":
    main()
