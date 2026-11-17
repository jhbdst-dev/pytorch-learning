import torch

from data.dataloader import MyDataset
from data.dataloader import create_dataloader
from models.model import SimpleModel

x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = 2 * x

dataset = MyDataset(x, y)
dataloader = create_dataloader(
    dataset,
    batch_size=2,
    shuffle=False,
    drop_last=False
)
model = SimpleModel()

for batch_x, batch_y in dataloader:
    y_pred = model(batch_x)

    print("입력 데이터 :", batch_x.shape)
    print("예측값 :", y_pred.shape)
