import torch

from data.dataset import MyDataset
from data.dataloader import create_dataloader
from models.model import SimpleModel
from losses.loss import create_loss_function
from optimizers.optimizer import create_optimizer
from training.train import train_one_epoch
from evaluation.evaluate import evaluate
from main import main


def test_training_loss_decreases():
    # 동일한 데이터로 학습을 반복하면 손실이 감소하는지 검증한다.
    torch.manual_seed(42)

    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=torch.float32
    )
    y = 2 * x

    dataset = MyDataset(x, y)

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    model = SimpleModel()
    criterion = create_loss_function()
    optimizer = create_optimizer(model)

    first_loss = train_one_epoch(
        model,
        dataloader,
        criterion,
        optimizer
    )

    for epoch in range(199):
        last_loss = train_one_epoch(model, dataloader, criterion, optimizer)

    assert first_loss > last_loss


def test_evaluation_after_training():
    # 학습에 사용하지 않은 입력 6~8에 대한 평가 손실을 검증한다.
    torch.manual_seed(42)

    # 학습 데이터: x=1~5, y=2x
    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0]],
        dtype=torch.float32
    )
    y = 2 * x

    train_dataset = MyDataset(x, y)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=2,
        shuffle=False
    )

    model = SimpleModel()
    criterion = create_loss_function()
    optimizer = create_optimizer(model)

    # 학습 단계에서만 Optimizer 사용
    for epoch in range(200):
        train_one_epoch(model, train_loader, criterion, optimizer)
        
    # 평가 데이터: 학습에 사용하지 않은 x=6~8
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

    # evaluate()는 파라미터를 수정하지 않으므로 Optimizer가 필요 없음
    eval_loss = torch.as_tensor(evaluate(model, eval_loader, criterion))

    assert eval_loss.ndim == 0
    assert torch.isfinite(eval_loss).item()
    assert 0 <= eval_loss.item() < 0.1


def test_main_integration():
    # 학습 -> 별도 데이터 평가 -> 새 데이터 예측의 전체 실행을 검증한다.
    result = main()

    train_loss = torch.as_tensor(result["train_loss"])
    eval_loss = torch.as_tensor(result["eval_loss"])
    prediction = result["prediction"]

    assert train_loss.ndim == 0
    assert eval_loss.ndim == 0
    assert torch.isfinite(train_loss).item()
    assert torch.isfinite(eval_loss).item()
    assert train_loss.item() >= 0
    assert eval_loss.item() >= 0

    assert prediction.shape == (1, 1)
    assert prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all().item()

    # 새 입력 9.0의 정답은 18.0: 예측이 정답에 가까운지 확인
    assert torch.allclose(prediction, torch.tensor([[18.0]]), atol=1.0, rtol=0.0)