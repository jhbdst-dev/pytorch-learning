import torch

from losses.loss import create_loss_function


def test_mse_loss():
    
    y_pred = torch.tensor([[1.0], [3.0], [5.0]])
    batch_y = torch.tensor([[2.0], [4.0], [6.0]])

    criterion = create_loss_function()
    loss = criterion(y_pred, batch_y)

    assert loss.item() == 1.0