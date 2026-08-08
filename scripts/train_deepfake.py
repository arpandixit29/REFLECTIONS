import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 2) # Binary: [Real Face, Synthetic/Deepfake]
        )

    def forward(self, x):
        return self.backbone(x)

def export_onnx(model, filename="deepfake_detector.onnx"):
    model.eval()
    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=["face_input"],
        output_names=["confidence_logits"],
        dynamic_axes={"face_input": {0: "batch_size"}, "confidence_logits": {0: "batch_size"}},
        opset_version=14
    )
    print(f"[ONNX Export] Exported Deepfake detector to {filename}")

if __name__ == "__main__":
    net = DeepfakeDetector()
    print("[Deepfake Trainer] Initialized EfficientNet-B0 Deepfake Forensic Classifier.")
    export_onnx(net)
