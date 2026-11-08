import torch
from data.dataset import MyDataset


def test_dataset():
    a = [[1], [2], [3], [4], [5]]
    x = torch.tensor(a, dtype=torch.float32)
    y = 2 * x

    dataset = MyDataset(x, y)

    assert len(dataset) == 5
    assert x.shape == y.shape
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert torch.equal(x * 2, y)

    input_data, target_data = dataset[2]

    assert torch.equal(input_data, x[2])
    assert torch.equal(target_data, y[2])