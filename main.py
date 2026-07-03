import os
import torch
import torch.nn as nn
from model import SAR_model, Optical_model, Fusion_model
from utils import (
    get_sar_dataloaders,
    get_optical_dataloaders,
    get_fusion_dataloaders,
    DEVICE,
    LR,
    EPOCHS
)
from train import train_one_epoch, evaluate


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save_checkpoint(path, model, optimizer, epoch, best_val_acc):
    torch.save({
        "epoch":           epoch,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc":    best_val_acc,
    }, path)


def _load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["epoch"] + 1, ckpt["best_val_acc"]


# ── SAR training ───────────────────────────────────────────────────────────────

def sar_main():
    print("\n" + "="*60)
    print("  SAR-only training")
    print("="*60)

    ckpt_path = "best_model.pth"
    model     = SAR_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch  = 0
    best_val_acc = 0.0

    if os.path.exists(ckpt_path):
        start_epoch, best_val_acc = _load_checkpoint(
            ckpt_path, model, optimizer, DEVICE)
        print(f"[SAR] Resumed from epoch {start_epoch} | "
              f"best_val_acc={best_val_acc:.4f}")

    train_loader, val_loader, test_loader = get_sar_dataloaders()

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)

        print(f"[SAR] Epoch {epoch+1:02d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_acc)
            print(f"       ✓ Checkpoint saved (best val_acc={best_val_acc:.4f})")

    test_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n[SAR] Final test accuracy : {test_acc*100:.2f}%")
    return test_acc


# ── Optical training ───────────────────────────────────────────────────────────

def optical_main():
    print("\n" + "="*60)
    print("  Optical-only training")
    print("="*60)

    ckpt_path = "best_model_optical.pth"
    model     = Optical_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch  = 0
    best_val_acc = 0.0

    if os.path.exists(ckpt_path):
        start_epoch, best_val_acc = _load_checkpoint(
            ckpt_path, model, optimizer, DEVICE)
        print(f"[Optical] Resumed from epoch {start_epoch} | "
              f"best_val_acc={best_val_acc:.4f}")

    train_loader, val_loader, test_loader = get_optical_dataloaders()

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)

        print(f"[Optical] Epoch {epoch+1:02d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_acc)
            print(f"          ✓ Checkpoint saved (best val_acc={best_val_acc:.4f})")

    test_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n[Optical] Final test accuracy : {test_acc*100:.2f}%")
    return test_acc


# ── Fusion training ────────────────────────────────────────────────────────────

def fusion_main(pretrained=False):
    """
    Args:
        pretrained: if True, warm-start both CNN branches from the
                    single-modal checkpoints before training begins.
                    The fusion head is always randomly initialised.
    """
    print("\n" + "="*60)
    print("  Late feature-level fusion training  (SAR + Optical)")
    print("="*60)

    ckpt_path = "best_model_fusion.pth"
    model     = Fusion_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch  = 0
    best_val_acc = 0.0

    # Warm-start only if no fusion checkpoint exists yet
    if pretrained and not os.path.exists(ckpt_path):
        model.load_pretrained_branches(
            sar_ckpt_path="best_model.pth",
            opt_ckpt_path="best_model_optical.pth",
            device=DEVICE,
        )

    if os.path.exists(ckpt_path):
        start_epoch, best_val_acc = _load_checkpoint(
            ckpt_path, model, optimizer, DEVICE)
        print(f"[Fusion] Resumed from epoch {start_epoch} | "
              f"best_val_acc={best_val_acc:.4f}")

    train_loader, val_loader, test_loader = get_fusion_dataloaders()

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, fusion=True)
        val_acc = evaluate(model, val_loader, DEVICE, fusion=True)

        print(f"[Fusion] Epoch {epoch+1:02d}/{EPOCHS}  "
              f"loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(ckpt_path, model, optimizer, epoch, best_val_acc)
            print(f"         ✓ Checkpoint saved (best val_acc={best_val_acc:.4f})")

    test_acc = evaluate(model, test_loader, DEVICE, fusion=True)
    print(f"\n[Fusion] Final test accuracy : {test_acc*100:.2f}%")

    # ── Final comparison table ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RESULTS SUMMARY (Based on Current Run)")
    print("="*60)
    print(f"  Fusion        : {test_acc*100:.2f}%")
    print("="*60)
    print("Note: Original experimental results reported in README (98.38% SAR,")
    print("98.08% Optical) were obtained on the legacy notebook splits.")
    return test_acc


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # sar_main()
    # optical_main()
    fusion_main(pretrained=True)
