import os
import torch
from torch.utils.data import Dataset
from PIL import Image

def build_canonical_manifest(base_dir):
    """
    Builds a canonical manifest of SAR and Optical image pairs.
    It guarantees exactly one SAR and one Optical file per sample identity.
    """
    classes = {"agri": 0, "barrenland": 1, "grassland": 2, "urban": 3}
    manifest = []
    
    total_sar = 0
    total_opt = 0
    unmatched_sar = 0
    unmatched_opt = 0
    duplicates = 0

    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"Dataset directory not found: {base_dir}\n"
            "Please ensure TERRAFUSE_DATA_DIR is set to a valid path containing the dataset classes."
        )

    for cls_name, label in classes.items():
        sar_dir = os.path.join(base_dir, cls_name, "s1")
        opt_dir = os.path.join(base_dir, cls_name, "s2")
        
        if not os.path.exists(sar_dir) or not os.path.exists(opt_dir):
            raise FileNotFoundError(f"Missing class directories for {cls_name} in {base_dir}")

        sar_files = {f: os.path.join(sar_dir, f) for f in os.listdir(sar_dir) if f.endswith(".png")}
        opt_files = {f: os.path.join(opt_dir, f) for f in os.listdir(opt_dir) if f.endswith(".png")}
        
        total_sar += len(sar_files)
        total_opt += len(opt_files)

        # Build pair by using SAR filename as base, since the notebook assumed opt_fname = sar_fname.replace("_s1_", "_s2_")
        # We also check for reverse unmatched.
        matched_opt = set()
        
        class_manifest = []
        for sar_fname, sar_path in sorted(sar_files.items()):
            opt_fname = sar_fname.replace("_s1_", "_s2_")
            if opt_fname in opt_files:
                opt_path = opt_files[opt_fname]
                class_manifest.append({
                    "sample_id": f"{cls_name}_{sar_fname.replace('_s1_', '_')}",
                    "sar_path": sar_path,
                    "opt_path": opt_path,
                    "label": label
                })
                matched_opt.add(opt_fname)
            else:
                unmatched_sar += 1
                print(f"[Manifest] WARNING: Unmatched SAR file: {sar_path}")
                
        for opt_fname, opt_path in opt_files.items():
            if opt_fname not in matched_opt:
                unmatched_opt += 1
                print(f"[Manifest] WARNING: Unmatched Optical file: {opt_path}")

        manifest.extend(class_manifest)

    # Check for duplicate sample IDs
    seen_ids = set()
    unique_manifest = []
    for item in manifest:
        if item["sample_id"] in seen_ids:
            duplicates += 1
            print(f"[Manifest] WARNING: Duplicate sample ID: {item['sample_id']}")
        else:
            seen_ids.add(item["sample_id"])
            unique_manifest.append(item)
            
    print("=" * 60)
    print("  Dataset Manifest Integrity Check")
    print("=" * 60)
    print(f"  Total SAR images found      : {total_sar}")
    print(f"  Total Optical images found  : {total_opt}")
    print(f"  Unmatched SAR images        : {unmatched_sar}")
    print(f"  Unmatched Optical images    : {unmatched_opt}")
    print(f"  Duplicate sample IDs        : {duplicates}")
    print(f"  Canonical pairs created     : {len(unique_manifest)}")
    print("=" * 60)
    
    if len(unique_manifest) == 0:
        raise ValueError(f"No valid image pairs found in {base_dir}")

    return unique_manifest


class SAR_Dataset(Dataset):
    """Single-modal dataset for Sentinel-1 SAR (grayscale) images based on canonical manifest."""
    def __init__(self, manifest, transform=None):
        self.manifest = manifest
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        image = Image.open(item["sar_path"]).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(item["label"], dtype=torch.long)


class Optical_Dataset(Dataset):
    """Single-modal dataset for Sentinel-2 Optical (RGB) images based on canonical manifest."""
    def __init__(self, manifest, transform=None):
        self.manifest = manifest
        self.transform = transform

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        image = Image.open(item["opt_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(item["label"], dtype=torch.long)


class Fusion_Dataset(Dataset):
    """Dual-modal dataset returning (sar_tensor, opt_tensor, label) triplets from canonical manifest."""
    def __init__(self, manifest, sar_transform=None, optical_transform=None, is_training=False):
        self.manifest = manifest
        self.sar_transform = sar_transform
        self.optical_transform = optical_transform
        self.is_training = is_training

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        item = self.manifest[idx]
        sar_img = Image.open(item["sar_path"]).convert("L")
        opt_img = Image.open(item["opt_path"]).convert("RGB")
        
        # Synchronized spatial augmentation (horizontal flip)
        if self.is_training:
            import random
            from torchvision.transforms import functional as F
            if random.random() > 0.5:
                sar_img = F.hflip(sar_img)
                opt_img = F.hflip(opt_img)

        if self.sar_transform:
            sar_img = self.sar_transform(sar_img)
        if self.optical_transform:
            opt_img = self.optical_transform(opt_img)
            
        return sar_img, opt_img, torch.tensor(item["label"], dtype=torch.long)
