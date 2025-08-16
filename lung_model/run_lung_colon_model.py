"""
TransMIL Inference Code with Hugging Face model loader.

- Load lung and colon models from HF repo at runtime
- Analyze WSI (.svs) or small patch images (.png/.jpg)
- Returns highlighted image, diagnosis, and cancer confidence
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import openslide
import matplotlib.pyplot as plt
from torchvision import models, transforms
import matplotlib.cm as cm
from PIL import Image
from huggingface_hub import login, hf_hub_download


# -------------------------
# Model classes (TransMIL)
# -------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        output = torch.matmul(weights, V)
        output = output.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(output)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        weights = self.attention(x)        # [B, N, 1]
        weights = F.softmax(weights, dim=1)  # Softmax over patches dimension
        pooled = torch.sum(weights * x, dim=1)  # Weighted sum of patch features [B, dim]
        weights = weights.squeeze(-1)  # [B, N]
        return pooled, weights


class TransMIL(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, num_classes=2,
                 num_layers=6, num_heads=8, max_patches=50000, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, hidden_dim) * 0.02)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pooling = AttentionPooling(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.max_patches = max_patches

    def forward(self, x):
        B, N, _ = x.shape
        if N > self.max_patches:
            x = x[:, :self.max_patches, :]
            N = self.max_patches

        x = self.input_proj(x)             # [B, N, hidden_dim]
        x = x + self.pos_embedding[:, :N, :]
        x = self.dropout(x)

        for blk in self.encoder_blocks:
            x = blk(x)

        pooled, attention = self.pooling(x)   # pooled: [B, hidden_dim], attention: [B, N]
        logits = self.classifier(pooled)      # [B, num_classes]
        return logits, attention


def load_partial_state_dict(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    """
    Load matching parameters from checkpoint into model.
    This avoids hard failures when parameter names/shapes differ slightly.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()
    filtered_dict = {}
    skipped_keys = []

    for k, v in checkpoint.items():
        if k in model_dict:
            if v.size() == model_dict[k].size():
                filtered_dict[k] = v
            else:
                skipped_keys.append(k)
        else:
            skipped_keys.append(k)

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)

    print(f"[load_partial_state_dict] Loaded {len(filtered_dict)} keys; skipped {len(skipped_keys)} keys.")
    if skipped_keys:
        print(f"[load_partial_state_dict] Skipped keys (sample): {skipped_keys[:10]}")


# -------------------------
# Preprocessing helpers
# -------------------------

def get_tissue_mask(slide):
    thumb_level = slide.level_count - 1
    thumb = slide.read_region((0, 0), thumb_level, slide.level_dimensions[thumb_level])
    thumb = np.array(thumb.convert("RGB"))
    hsv = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, thumb_level


def extract_patches(slide, mask, thumb_level, level, patch_size, max_patches):
    coords, patches = [], []
    scale = slide.level_downsamples[thumb_level] / slide.level_downsamples[level]
    W, H = slide.level_dimensions[level]
    step = patch_size // 2  # 50% overlap

    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            mx, my = int(x / scale), int(y / scale)
            mps = max(1, int(patch_size / scale))

            if (my + mps <= mask.shape[0]) and (mx + mps <= mask.shape[1]):
                region = mask[my:my + mps, mx:mx + mps]
                if np.mean(region) > 30:
                    patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
                    patch_np = np.array(patch)
                    if np.mean(patch_np) < 220:
                        coords.append((x, y))
                        patches.append(patch_np)
                    if len(patches) >= max_patches:
                        return patches, coords
    return patches, coords


def extract_features(patches, device):
    if len(patches) == 0:
        return torch.empty((0, 2048), dtype=torch.float32)

    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC
    backbone.to(device).eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    features = []
    with torch.no_grad():
        for i in range(0, len(patches), 32):
            batch = [transform(p).to(device) for p in patches[i:i + 32]]
            if batch:
                inp = torch.stack(batch)
                out = backbone(inp).squeeze(-1).squeeze(-1)
                features.append(out.cpu())
    return torch.cat(features, dim=0)


# -------------------------
# Visualization helper
# -------------------------

def visualize_attention_heatmap(slide, coords, attention_scores, level, patch_size, alpha, model_prob):
    if len(coords) == 0:
        print("[visualize] No patches to visualize.")
        return None

    attention_scores = attention_scores.squeeze().cpu().numpy()
    n = min(len(coords), len(attention_scores))
    attention_scores = attention_scores[:n]
    coords = coords[:n]

    dims = slide.level_dimensions[level]
    thumb = slide.get_thumbnail(dims)
    thumb_np = np.array(thumb.convert("RGB"), dtype=np.float32)

    att_min, att_max = np.min(attention_scores), np.max(attention_scores)
    norm_scores = (attention_scores - att_min) / (att_max - att_min + 1e-8)

    cmap = cm.get_cmap('YlOrRd')
    result = thumb_np.copy()

    for idx, score in enumerate(norm_scores):
        x, y = coords[idx]
        x = int(x / slide.level_downsamples[level])
        y = int(y / slide.level_downsamples[level])
        w = int(patch_size / slide.level_downsamples[level])
        h = int(patch_size / slide.level_downsamples[level])

        color_index = 0.75 * score + 0.25 * model_prob
        color_index = float(np.clip(color_index, 0.0, 1.0))

        rgba = cmap(color_index)
        color = np.array(rgba[:3], dtype=np.float32) * 255.0

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(result.shape[1], x + w), min(result.shape[0], y + h)
        if x1 >= x2 or y1 >= y2:
            continue

        h_slice = y2 - y1
        w_slice = x2 - x1
        color_patch = np.ones((h_slice, w_slice, 3), dtype=np.float32) * color.reshape((1, 1, 3))

        result[y1:y2, x1:x2] = result[y1:y2, x1:x2] * (1.0 - alpha) + color_patch * alpha

    result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
    return result_uint8


# -------------------------
# Hugging Face model loader
# -------------------------

def load_models_from_hf(device, repo_id="draziza/lung-colon-model"):
    """
    Download lung.pth and colon.pth from HF repo and load into TransMIL models.
    """
    print("Downloading models from Hugging Face repo...")

    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            login(token=hf_token)
            print("[load_models_from_hf] Logged in to Hugging Face Hub using HF_TOKEN.")
        except Exception as e:
            raise RuntimeError(f"Failed to login to Hugging Face Hub with HF_TOKEN: {e}")
    else:
        print("[load_models_from_hf] No HF_TOKEN found in environment.")

    # Download checkpoint files from the specified repo.
    try:
        lung_ckpt_path = hf_hub_download(repo_id=repo_id, filename="lung.pth")
        print(f"[load_models_from_hf] Downloaded lung.pth -> {lung_ckpt_path}")
    except Exception as ex:
        raise RuntimeError(f"Failed to download lung.pth from {repo_id}: {ex}")

    try:
        colon_ckpt_path = hf_hub_download(repo_id=repo_id, filename="colon.pth")
        print(f"[load_models_from_hf] Downloaded colon.pth -> {colon_ckpt_path}")
    except Exception as ex:
        raise RuntimeError(f"Failed to download colon.pth from {repo_id}: {ex}")

    # Instantiate models and load checkpoints
    lung_model = TransMIL().to(device)
    colon_model = TransMIL().to(device)

    # Load weights safely (partial load to tolerate architectural differences)
    load_partial_state_dict(lung_model, lung_ckpt_path, device)
    load_partial_state_dict(colon_model, colon_ckpt_path, device)

    lung_model.eval()
    colon_model.eval()

    print("[load_models_from_hf] Models loaded and set to eval().")
    return {"lung": lung_model, "colon": colon_model}


# -------------------------
# Main analyze function
# -------------------------

def analyze_wsi(input_path, model,
                patch_size=224, level=0, max_patches=50000, alpha=0.3,
                device=None):
    """
    Analyze a WSI or a small image patch and return:
    - highlighted image (np.ndarray RGB)
    - diagnosis (str)
    - cancer confidence (float, 0-100)

    'model' is a TransMIL instance (already loaded on device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ext = os.path.splitext(input_path)[1].lower()

    if ext == '.svs':  # WSI input
        slide = openslide.OpenSlide(input_path)
        mask, thumb_level = get_tissue_mask(slide)
        patches, coords = extract_patches(slide, mask, thumb_level, level=level,
                                          patch_size=patch_size, max_patches=max_patches)

        if len(patches) == 0:
            raise RuntimeError("No patches extracted from WSI.")

        features = extract_features(patches, device).unsqueeze(0).to(device)  # [1, N, 2048]

        with torch.no_grad():
            logits, attention = model(features)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            prob_percent = prob * 100.0

        if prob_percent <= 30.0:
            diagnosis = "Normal"
            highlighted_img = np.array(slide.get_thumbnail(slide.level_dimensions[thumb_level]).convert("RGB"))
        elif 30.0 < prob_percent <= 70.0:
            diagnosis = "Suspicion of cancer"
            highlighted_img = visualize_attention_heatmap(slide, coords, attention, thumb_level,
                                                        patch_size, alpha, prob)
        else:
            diagnosis = "Cancer"
            highlighted_img = visualize_attention_heatmap(slide, coords, attention, thumb_level,
                                                        patch_size, alpha, prob)

        if highlighted_img is None:
            highlighted_img = np.array(slide.get_thumbnail(slide.level_dimensions[thumb_level]).convert("RGB"))

        return highlighted_img, diagnosis, prob_percent

    else:  # Assume small image patch (.png, .jpg, etc.)
        img = Image.open(input_path).convert("RGB")
        img_np = np.array(img)

        features = extract_features([img_np], device).unsqueeze(0).to(device)  # [1,1,2048]

        with torch.no_grad():
            logits, attention = model(features)
            prob = F.softmax(logits, dim=1)[0, 1].item()
            prob_percent = prob * 100.0

        if prob_percent <= 30.0:
            diagnosis = "Normal"
        elif 30.0 < prob_percent <= 70.0:
            diagnosis = "Suspicion of cancer"
        else:
            diagnosis = "Cancer"

        highlighted_img = img_np

        return highlighted_img, diagnosis, prob_percent


# -------------------------
# Minimal test main
# -------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python run_weak_model.py <wsi_or_patch_path> <HF_repo_id>")
        print("Example HF repo id: your-username/transmil-lung-colon-models")
        sys.exit(1)

    wsi_path = sys.argv[1]
    hf_repo_id = sys.argv[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models_from_hf(device, repo_id=hf_repo_id)

    # Example: pick lung model
    model = models["lung"]

    highlighted_img, diagnosis, confidence = analyze_wsi(wsi_path, model, device=device)

    print(f"Diagnosis: {diagnosis}")
    print(f"Cancer Confidence: {confidence:.2f}%")

    plt.figure(figsize=(12, 12))
    plt.imshow(highlighted_img)
    plt.title(f"{diagnosis} ({confidence:.2f}%)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
