import torch
from models.model import SimpleModel
from data.dataset import MyDataset
from data.dataloader import create_dataloader

def test_model_output_shape():
    model = SimpleModel()

    x = torch.tensor(
        [[1], [2], [3], [4], [5]],
        dtype=torch.float32
    )

    y_pred = model(x)

    assert y_pred.shape == (5, 1)


def test_model_parameters():
    model = SimpleModel()

    weight = model.linear.weight
    bias = model.linear.bias

    assert weight.shape == (1,1)
    assert bias.shape == (1,)


def test_model_dataloader():

    x = torch.tensor([[1], [2], [3], [4], [5]], dtype=torch.float32)
    y = 2 * x

    dataset = MyDataset(x, y)
    model = SimpleModel()

    dataloader = create_dataloader(
        dataset,
        batch_size=2
    )

    batch_count = 0

    for batch_x, batch_y in dataloader:
        y_pred = model(batch_x)

        assert y_pred.shape == batch_x.shape

        batch_count += 1

    assert batch_count == 3