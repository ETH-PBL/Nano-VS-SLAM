from src.kp2dtiny.models.kp2dtiny import tiny_factory
import torch
model = tiny_factory("S_A", 28, v3=True).cpu()

model.eval()
model.training = False


x = torch.randn(1, 3, 120, 160)
out = model(x)
print(out.keys())
print("Sanity check passed.")