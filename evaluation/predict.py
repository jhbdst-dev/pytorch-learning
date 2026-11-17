import torch


def predict(model, x):
    model.eval()

    with torch.inference_mode():
        y_pred = model(x)

    return y_pred