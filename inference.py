import torch
import torch.nn.functional as F
from PIL import Image
from model import SAR_model, Optical_model, Fusion_model
from utils import (
    DEVICE,
    CLASS_NAMES,
    SAR_EVAL_TRANSFORM,
    OPTICAL_EVAL_TRANSFORM
)

def load_model(mode, ckpt_path, device=DEVICE):
    models = {"sar": SAR_model, "optical": Optical_model, "fusion": Fusion_model}
    if mode not in models:
        raise ValueError(f"Invalid mode. Choose from {list(models.keys())}")
        
    model = models[mode]()
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"Loaded '{mode}' | epoch {ckpt.get('epoch', -1)+1} | best_val_acc={ckpt.get('best_val_acc', 0.0):.4f}")
    return model

def predict_fusion(model, sar_path, opt_path, device=DEVICE):
    sar = SAR_EVAL_TRANSFORM(Image.open(sar_path).convert("L")).unsqueeze(0).to(device)
    opt = OPTICAL_EVAL_TRANSFORM(Image.open(opt_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(sar, opt), dim=1).squeeze()
    pred = probs.argmax().item()
    return {
        "predicted_class": CLASS_NAMES[pred],
        "confidence": round(probs[pred].item(), 4),
        "probabilities": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASS_NAMES)}
    }

def predict_sar(model, sar_path, device=DEVICE):
    img = SAR_EVAL_TRANSFORM(Image.open(sar_path).convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1).squeeze()
    pred = probs.argmax().item()
    return {
        "predicted_class": CLASS_NAMES[pred],
        "confidence": round(probs[pred].item(), 4),
        "probabilities": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASS_NAMES)}
    }

def predict_optical(model, opt_path, device=DEVICE):
    img = OPTICAL_EVAL_TRANSFORM(Image.open(opt_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1).squeeze()
    pred = probs.argmax().item()
    return {
        "predicted_class": CLASS_NAMES[pred],
        "confidence": round(probs[pred].item(), 4),
        "probabilities": {c: round(probs[i].item(), 4) for i, c in enumerate(CLASS_NAMES)}
    }

def print_result(result):
    print(f"\nPredicted : {result['predicted_class']}  (conf={result['confidence']:.4f})")
    for cls, prob in result["probabilities"].items():
        print(f"  {cls:<12} {prob:.4f}  {'█' * int(prob * 40)}")
