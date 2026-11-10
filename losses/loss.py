from torch import nn


def create_loss_function():
    criterion = nn.MSELoss()
    return criterion