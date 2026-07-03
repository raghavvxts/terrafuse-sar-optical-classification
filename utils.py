import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from dataset import build_canonical_manifest, SAR_Dataset, Optical_Dataset, Fusion_Dataset

# ── Global config ──────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
SEED        = 42          # fixed seed → reproducible train/val/test splits

# Configurable dataset path
BASE_DIR = os.environ.get(
    "TERRAFUSE_DATA_DIR", 
    "/kaggle/input/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain/v_2"
)

CLASS_NAMES = ["agri", "barrenland", "grassland", "urban"]

# ── Transforms ─────────────────────────────────────────────────────────────────
SAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

SAR_EVAL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

OPTICAL_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

OPTICAL_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Transforms for Fusion (no automatic flip to allow manual sync)
FUSION_SAR_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

FUSION_OPTICAL_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Internal split helper ──────────────────────────────────────────────────────
def _split_indices(total, seed=SEED):
    """
    Returns (train_idx, val_idx, test_idx) for a 70 / 15 / 15 split.
    Uses a fixed Generator so splits are identical across every run.
    """
    generator = torch.Generator().manual_seed(seed)
    indices   = torch.randperm(total, generator=generator).tolist()
    n_train   = int(0.70 * total)
    n_val     = int(0.15 * total)
    return (
        indices[:n_train],
        indices[n_train: n_train + n_val],
        indices[n_train + n_val:],
    )


# ── Canonical Manifest Initialisation ──────────────────────────────────────────
# Build the manifest once to ensure SAR, Optical, and Fusion datasets 
# always use the exact same paired samples in the exact same order.
_manifest = None

def get_manifest():
    global _manifest
    if _manifest is None:
        _manifest = build_canonical_manifest(BASE_DIR)
    return _manifest


# ── SAR dataloaders ────────────────────────────────────────────────────────────
def get_sar_dataloaders(batch_size=BATCH_SIZE):
    manifest = get_manifest()
    train_ds = SAR_Dataset(manifest, transform=SAR_TRAIN_TRANSFORM)
    val_ds   = SAR_Dataset(manifest, transform=SAR_EVAL_TRANSFORM)
    test_ds  = SAR_Dataset(manifest, transform=SAR_EVAL_TRANSFORM)

    train_idx, val_idx, test_idx = _split_indices(len(train_ds))

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds, test_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ── Optical dataloaders ────────────────────────────────────────────────────────
def get_optical_dataloaders(batch_size=BATCH_SIZE):
    manifest = get_manifest()
    train_ds = Optical_Dataset(manifest, transform=OPTICAL_TRAIN_TRANSFORM)
    val_ds   = Optical_Dataset(manifest, transform=OPTICAL_EVAL_TRANSFORM)
    test_ds  = Optical_Dataset(manifest, transform=OPTICAL_EVAL_TRANSFORM)

    train_idx, val_idx, test_idx = _split_indices(len(train_ds))

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds, test_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# ── Fusion dataloaders ─────────────────────────────────────────────────────────
def get_fusion_dataloaders(batch_size=BATCH_SIZE):
    manifest = get_manifest()
    train_ds = Fusion_Dataset(manifest,
                              sar_transform=FUSION_SAR_TRAIN_TRANSFORM,
                              optical_transform=FUSION_OPTICAL_TRAIN_TRANSFORM,
                              is_training=True)
    val_ds   = Fusion_Dataset(manifest,
                              sar_transform=SAR_EVAL_TRANSFORM,
                              optical_transform=OPTICAL_EVAL_TRANSFORM,
                              is_training=False)
    test_ds  = Fusion_Dataset(manifest,
                              sar_transform=SAR_EVAL_TRANSFORM,
                              optical_transform=OPTICAL_EVAL_TRANSFORM,
                              is_training=False)

    train_idx, val_idx, test_idx = _split_indices(len(train_ds))

    train_loader = DataLoader(Subset(train_ds, train_idx),
                              batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(Subset(val_ds, val_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(Subset(test_ds, test_idx),
                              batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader
