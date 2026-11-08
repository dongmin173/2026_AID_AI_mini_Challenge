import sys
import torch
import torchvision
from model import build_model

print("Python:", sys.executable)
print("Python version:", sys.version.split()[0])
print("torch:", torch.__version__, "torchvision:", torchvision.__version__)
print("CUDA build:", torch.version.cuda, "CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("Use the Python executable from your working GPU venv before training.")
print("GPU:", torch.cuda.get_device_name(0))
net = build_model("regnet32", pretrained=False).cuda().eval()
print("RegNetY-3.2GF binary classifier parameters:", sum(p.numel() for p in net.parameters()))
with torch.inference_mode():
    y = net(torch.zeros(1, 3, 224, 224, device="cuda"))
assert y.shape == (1, 2) and torch.isfinite(y).all()
print("GPU forward pass OK. This check did not download weights.")
