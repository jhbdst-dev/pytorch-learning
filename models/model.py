import torch
from torch import nn

class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


if __name__ == "__main__":
    model = SimpleModel()

    x = torch.tensor(
        [[1], [2], [3], [4], [5]],
        dtype=torch.float32
    )

    y_pred = model(x)

    assert y_pred.shape == (5, 1)
    
    print("예측값 :", y_pred)
    print("출력 Shape :", y_pred.shape)