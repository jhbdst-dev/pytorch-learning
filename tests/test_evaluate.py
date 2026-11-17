import torch
import pytest

from evaluation.evaluate import evaluate
from models.model import SimpleModel
from data.dataset import MyDataset
from data.dataloader import create_dataloader
from losses.loss import create_loss_function


def test_evaluate_returns_average_loss():

    x = torch.tensor(
        [[1.0], [2.0], [3.0]],
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

    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)

    criterion = create_loss_function()

    actual_loss = evaluate(model, dataloader, criterion)

    expected_loss = 14 / 3

    assert actual_loss == pytest.approx(expected_loss)


def test_evaluate_calculates_average_loss():
    torch.manual_seed(42)

    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]],
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

    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)

    criterion = create_loss_function()

    actual_loss = evaluate(model, dataloader, criterion)

    assert actual_loss == pytest.approx(7.5)


def test_evaluate_does_not_update_weights():
    torch.manual_seed(42)
    
    x = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0]],
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

    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)

    criterion = create_loss_function()

    weight_before = model.linear.weight.detach().clone()
    bias_before = model.linear.bias.detach().clone()
    
    actual_loss = evaluate(model, dataloader, criterion)

    weight_after = model.linear.weight.detach().clone()
    bias_after = model.linear.bias.detach().clone()

    assert torch.equal(weight_before, weight_after)
    assert torch.equal(bias_before, bias_after)


def test_evaluate_handles_unequal_batch_sizes():
    
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

    with torch.no_grad():
        model.linear.weight.fill_(1.0)
        model.linear.bias.fill_(0.0)

    criterion = create_loss_function()

    actual_loss = evaluate(model, dataloader, criterion)

    expected_loss = 11.0

    assert actual_loss == expected_loss

def test_evaluate_raises_on_empty_dataloader():
    x = torch.empty((0, 1), dtype=torch.float32)
    y = 2 * x

    dataset = MyDataset(x, y)

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    model = SimpleModel()
    criterion = create_loss_function()

    with pytest.raises(ValueError):
        evaluate(model, dataloader, criterion)

    