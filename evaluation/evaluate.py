import torch

def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    num_samples = 0

    with torch.inference_mode():
        for batch_x, batch_y in dataloader:
            y_pred = model(batch_x)
            loss = criterion(y_pred, batch_y)

            batch_size = batch_x.shape[0]

            total_loss += loss.item() * batch_size
            num_samples += batch_size

    if num_samples == 0:
        raise ValueError("Evaluation DataLoader is empty")

    abg_loss = total_loss / num_samples

    return abg_loss