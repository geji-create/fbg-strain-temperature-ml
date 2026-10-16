from __future__ import annotations

import torch


print(f"PyTorch: {torch.__version__}")
print(f"CUDA build: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU was not detected by PyTorch.")

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")

matrix = torch.randn(1024, 1024, device=device)
result = matrix @ matrix.T
torch.cuda.synchronize()
print(f"Matrix test: shape={tuple(result.shape)}, finite={torch.isfinite(result).all().item()}")

