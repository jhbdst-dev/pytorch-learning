import torch

from data.dataset import MyDataset
from data.dataloader import create_dataloader
from models.model import SimpleModel
from losses.loss import create_loss_function
from optimizers.optimizer import create_optimizer
from training.train import train_one_epoch


def test_train_loss_decreases():
    torch.manual_seed(42)

    x = torch.tensor(
        [[1], [2], [3], [4], [5]],
        dtype=torch.float32
    )
    y = 2 * x

    # TODO 1: Dataset 생성
    dataset = MyDataset(x, y)

    # TODO 2: DataLoader 생성
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    # TODO 3: Model 생성
    model = SimpleModel()

    # TODO 4: Loss Function 생성
    criterion = create_loss_function()
    
    # TODO 5: Optimizer 생성
    optimizer = create_optimizer(model)

    # TODO 6: 학습 전 Loss 측정
    model.eval()

    with torch.no_grad():
        initial_pred = model(x)
        initial_loss = criterion(initial_pred, y).item()

    print("초기 예측값 :", initial_pred)
    print("초기 Loss :", initial_loss)

    # TODO 7: 반복 학습
    for epoch in range(10):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer)

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    # TODO 8: 학습 후 Loss 측정

    model.eval()

    with torch.no_grad():
        final_pred = model(x)
        final_loss = criterion(final_pred, y).item()

    print("최종 예측값 :", final_pred)
    print("최종 Loss :", final_loss)

    assert final_loss < initial_loss