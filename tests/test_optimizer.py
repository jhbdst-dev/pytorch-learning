from models.model import SimpleModel
from optimizers.optimizer import create_optimizer


def test_optimizer():
    model = SimpleModel()
    optimizer = create_optimizer(model, lr=0.01)

    optimizer.param_groups[0]["lr"]

    params = optimizer.param_groups[0]["params"]

    assert optimizer.param_groups[0]["lr"] == 0.01
    assert len(params) == 2

    assert params[0] is model.linear.weight
    assert params[1] is model.linear.bias