import torch
import torchvision
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_tensor

# 👉 SET THIS to your real number of classes (incl. background)
NUM_CLASSES = 2  # e.g., 2 = {background, nodule}

def build_lungct_model():
    # If you trained a different architecture, replace this block with your model.
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_feats, NUM_CLASSES
    )
    return model

def load_lungct_model(repo_id: str, filename: str, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    ckpt = torch.load(ckpt_path, map_location=device)

    # If file is a full model object, just return it.
    if hasattr(ckpt, "eval") and callable(ckpt.eval):
        model = ckpt.to(device)
        model.eval()
        return model, device

    # Otherwise it’s a state_dict (OrderedDict) -> rebuild model and load weights.
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # strip common prefixes like "module." or "model."
    state = {k.replace("module.", "").replace("model.", ""): v for k, v in state.items()}

    model = build_lungct_model().to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, device

def default_transform(img):
    # detection models expect a float tensor in [0,1]
    return to_tensor(img)

def predict_lungct(model, device, transform, image):
    x = transform(image).to(device)
    # TorchVision detection models expect a list of images
    with torch.no_grad():
        out = model([x])[0]
    return out
