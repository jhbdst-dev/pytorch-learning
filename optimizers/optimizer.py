import torch


def create_optimizer(model, lr=0.01):
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr
    )
    return optimizer