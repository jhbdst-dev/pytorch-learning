import torch

from data.dataset import MyDataset
from data.dataloader import create_dataloader
from models.model import SimpleModel
from losses.loss import create_loss_function
from optimizers.optimizer import create_optimizer
from training.train import train_one_epoch
from evaluation.evaluate import evaluate
from evaluation.predict import predict


def main():
    torch.manual_seed(42)

    # 1: 데이터 생성
    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=torch.float32
    )
    y = 2 * x

    # 2: Dataset / DataLoader 생성
    dataset = MyDataset(x, y)
    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    # 3: Model / Loss / Optimizer 생성
    model = SimpleModel()
    criterion = create_loss_function()
    optimizer = create_optimizer(model)

    # 4: Epoch 반복 학습
    epochs = 200

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer)
        print(epoch + 1, train_loss)

    # 5: 평가
    eval_x = torch.tensor(
        [[6.0], [7.0], [8.0]],
        dtype=torch.float32
    )

    eval_y = 2 * eval_x

    eval_dataset = MyDataset(eval_x, eval_y)
    eval_loader = create_dataloader(
        eval_dataset,
        batch_size=2,
        shuffle=False
    )

    eval_loss = evaluate(
        model,
        eval_loader,
        criterion
    )

    print("Evaluation Loss:", eval_loss)

    # 6: 새로운 데이터 예측
    new_x = torch.tensor(
        [[9.0]],
        dtype=torch.float32
    )

    prediction = predict(
        model,
        new_x
    )

    print("Prediction:", prediction)

    # 7-1: prediction의 Shape 검증
    assert prediction.shape == (1, 1)

    # 7-2: prediction의 dtype 검증
    assert prediction.dtype == torch.float32

    # 7-3: 입력 데이터와 정답 데이터의 Shape 일치 검증
    assert x.shape == y.shape

    # 7-4: 입력 데이터와 정답 데이터의 dtype 일치 검증
    assert x.dtype == y.dtype

    # 평가 손실 및 예측 결과가 유한한지 검증
    assert torch.isfinite(torch.as_tensor(eval_loss)).all().item()
    assert torch.isfinite(prediction).all().item()

    return {
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "prediction": prediction
    }

if __name__ == "__main__":
    main()