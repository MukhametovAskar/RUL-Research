import torch
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, device, scaler, criterion, max_grad_norm=float('inf')):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(loader, desc="Training"):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            preds = model(xb)
            loss = criterion(preds, yb)  
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)
