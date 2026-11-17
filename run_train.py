import torch

from data.dataset import MyDataset
from data.dataloader import create_dataloader
from models.model import SimpleModel
from losses.loss import create_loss_function
from optimizers.optimizer import create_optimizer
from training.train import train_one_epoch


x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
y = 2 * x

dataset = MyDataset(x, y)
dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)

model = SimpleModel()
criterion = create_loss_function()
optimizer = create_optimizer(model)

for epoch in range(10):
    avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)

    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")