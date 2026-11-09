import torch
from torch.utils.data import DataLoader
from data.dataset import MyDataset


def create_dataloader(dataset, batch_size=2, shuffle=False, drop_last=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    return dataloader


if __name__ == "__main__":
    
    a = [[1], [2], [3], [4], [5]]
    x = torch.tensor(a, dtype=torch.float32)
    y = 2 * x
    
    dataset = MyDataset(x, y)
    
    dataloader = create_dataloader(dataset, batch_size=2, shuffle=False)    
    
    for inputs, targets in dataloader:
        print(inputs, targets)
