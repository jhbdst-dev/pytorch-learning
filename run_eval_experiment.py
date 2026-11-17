import torch

from models.model import SimpleModel

model = SimpleModel()

x = torch.tensor([[6.0]], dtype=torch.float32)

# 실험 1: eval()만 사용
model. eval()

y_pred1 = model(x)

print("실험 1 :", y_pred1.requires_grad)

# 실험 2: eval()과 no_grad() 함께 사용
with torch.no_grad():
    y_pred2 = model(x)

print("실험 2:", y_pred2.requires_grad)