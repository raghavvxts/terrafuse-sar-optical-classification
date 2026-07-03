import torch
from tqdm import tqdm


def train_one_epoch(model, loader, criterion, optimizer, device, fusion=False):
    """
    Runs one full training epoch.

    Args:
        fusion: if True, expects batches of (sar_img, opt_img, label);
                if False, expects batches of (image, label).

    Returns:
        Average cross-entropy loss over the epoch.
    """
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="  Train", leave=False):
        if fusion:
            sar_imgs, opt_imgs, labels = batch
            sar_imgs = sar_imgs.to(device)
            opt_imgs = opt_imgs.to(device)
            labels   = labels.to(device)
            optimizer.zero_grad()
            outputs  = model(sar_imgs, opt_imgs)
        else:
            images, labels = batch
            images  = images.to(device)
            labels  = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, fusion=False):
    """
    Evaluates model accuracy on the given loader.

    Returns:
        Fraction of correctly classified samples (float in [0, 1]).
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Eval ", leave=False):
            if fusion:
                sar_imgs, opt_imgs, labels = batch
                sar_imgs = sar_imgs.to(device)
                opt_imgs = opt_imgs.to(device)
                labels   = labels.to(device)
                outputs  = model(sar_imgs, opt_imgs)
            else:
                images, labels = batch
                images  = images.to(device)
                labels  = labels.to(device)
                outputs = model(images)

            preds    = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total
