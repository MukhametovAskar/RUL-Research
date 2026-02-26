import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    criterion = torch.nn.MSELoss()

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
