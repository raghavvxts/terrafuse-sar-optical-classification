import torch
import torch.nn as nn

# ── Shared building blocks ─────────────────────────────────────────────────────

def _make_conv_block(in_channels, out_channels):
    """Conv2d(3×3, pad=1) → BatchNorm → ReLU → MaxPool(2×2)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )

def _build_feature_extractor(in_channels):
    """
    5 conv blocks:  in_channels → 32 → 64 → 64 → 128 → 128
    Input 256×256   →  output shape  128 × 8 × 8
    Flattened dim   =  128 × 8 × 8  =  8 192
    """
    return nn.Sequential(
        _make_conv_block(in_channels, 32),
        _make_conv_block(32, 64),
        _make_conv_block(64, 64),
        _make_conv_block(64, 128),
        _make_conv_block(128, 128),
    )


FLAT_DIM = 128 * 8 * 8   # 8192  — single-branch flattened size


# ── Single-modal models ────────────────────────────────────────────────────────

class SAR_model(nn.Module):
    """Single-modal CNN for Sentinel-1 SAR (1-channel grayscale)."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = _build_feature_extractor(in_channels=1)
        self.classifier = nn.Sequential(
            nn.Linear(FLAT_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class Optical_model(nn.Module):
    """Single-modal CNN for Sentinel-2 Optical (3-channel RGB)."""

    def __init__(self, num_classes=4):
        super().__init__()
        self.features = _build_feature_extractor(in_channels=3)
        self.classifier = nn.Sequential(
            nn.Linear(FLAT_DIM, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ── Late feature-level fusion model ─────────────────────────────────────────────

class Fusion_model(nn.Module):
    """
    Late feature-level fusion dual-modal CNN.

    Architecture
    ────────────
    SAR branch   (1-ch)  →  128×8×8  →  flatten  →  8192-dim
    Optical branch (3-ch) →  128×8×8  →  flatten  →  8192-dim
                                         concat   →  16384-dim
                                    fusion head   →  4 logits

    Fusion head: Linear(16384→128) + ReLU + Dropout(0.5) + Linear(128→4)

    Optional warm-start
    ───────────────────
    Call load_pretrained_branches() after instantiation to initialise both
    CNN branches from the single-modal checkpoints.  The fusion head is
    always trained from scratch so the branches can co-adapt.
    """

    def __init__(self, num_classes=4):
        super().__init__()
        self.sar_branch     = _build_feature_extractor(in_channels=1)
        self.optical_branch = _build_feature_extractor(in_channels=3)

        # Concatenated flat dim = 8192 + 8192 = 16384
        self.fusion_classifier = nn.Sequential(
            nn.Linear(FLAT_DIM * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, sar_img, opt_img):
        sar_feat = self.sar_branch(sar_img)
        sar_feat = sar_feat.view(sar_feat.size(0), -1)   # (B, 8192)

        opt_feat = self.optical_branch(opt_img)
        opt_feat = opt_feat.view(opt_feat.size(0), -1)   # (B, 8192)

        fused = torch.cat([sar_feat, opt_feat], dim=1)   # (B, 16384)
        return self.fusion_classifier(fused)

    def load_pretrained_branches(self, sar_ckpt_path, opt_ckpt_path, device):
        """
        Warm-start both CNN branches from single-modal checkpoints.
        Only feature extractor weights are copied; fusion head stays random.
        """
        def _extract(state_dict, prefix="features."):
            return {k[len(prefix):]: v
                    for k, v in state_dict.items()
                    if k.startswith(prefix)}

        sar_ckpt = torch.load(sar_ckpt_path, map_location=device)
        opt_ckpt = torch.load(opt_ckpt_path, map_location=device)

        self.sar_branch.load_state_dict(_extract(sar_ckpt["model_state"]))
        self.optical_branch.load_state_dict(_extract(opt_ckpt["model_state"]))
        print("[Fusion_model] Pretrained branch weights loaded successfully.")
