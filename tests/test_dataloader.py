import torch
from torch.utils.data import DataLoader
from data.dataset import MyDataset
from data.dataloader import create_dataloader

def test_dataloader():
    a = [[1], [2], [3], [4], [5]]
    x = torch.tensor(a, dtype=torch.float32)
    y = 2 * x

    dataset = MyDataset(x, y)

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False
    )

    assert len(dataloader) == 3

    iterator = iter(dataloader)
    inputs, targets = next(iterator)

    assert inputs.shape == (2, 1)
    assert targets.shape == (2, 1)

    assert torch.equal(inputs * 2, targets)


def test_drop_last():
    a = [[1], [2], [3], [4], [5]]
    x = torch.tensor(a, dtype=torch.float32)
    y = 2 * x

    dataset = MyDataset(x, y)

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        drop_last=True
    )

    assert len(dataset) == 5
    assert len(dataloader) == 2

    total_count = 0

    for inputs, targets in dataloader:
        total_count += len(inputs)

    assert total_count == 4


def test_shuffle():
    a = [[1], [2], [3], [4], [5]]
    x = torch.tensor(a, dtype=torch.float32)
    y = 2 * x

    dataset = MyDataset(x, y)

    dataloader = create_dataloader(
        dataset,
        batch_size=2,
        shuffle=True
    )

    all_inputs = []

    for inputs, targets in dataloader:
        assert torch.equal(inputs * 2, targets)
        all_inputs.append(inputs)

    combined_inputs = torch.cat(all_inputs, dim=0)
    sorted_inputs = torch.sort(combined_inputs, dim=0).values

    assert torch.equal(sorted_inputs, x)