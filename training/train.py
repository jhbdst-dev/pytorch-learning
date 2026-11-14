def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0.0

    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()

        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)

        total_loss += loss.item() * batch_x.size(0)

        loss.backward()

        optimizer.step()

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss