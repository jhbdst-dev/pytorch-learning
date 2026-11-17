import torch

from models.model import SimpleModel
from evaluation.predict import predict


def test_predict_returns_expected_value():
    model = SimpleModel()

    with torch.no_grad():
        model.linear.weight.fill_(2.0)
        model.linear.bias.fill_(0.0)

    x = torch.tensor([[6.0]])

    actual_pred = predict(model, x)
    expected_pred = torch.tensor([[12.0]])

    assert torch.equal(actual_pred, expected_pred)
    assert actual_pred.shape == (1, 1)
    assert actual_pred.requires_grad is False


def test_predict_handles_multiple_inputs():
    model = SimpleModel()

    with torch.no_grad():
        model.linear.weight.fill_(2.0)
        model.linear.bias.fill_(0.0)

    x = torch.tensor(
        [[6.0], [7.0], [8.0]],
        dtype=torch.float32
    )

    actual_pred = predict(model, x)
    expected_pred = torch.tensor([[12.0], [14.0], [16.0]])

    assert torch.equal(actual_pred, expected_pred)
    assert actual_pred.shape == (3, 1)
    assert actual_pred.requires_grad is False