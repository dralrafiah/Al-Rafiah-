# lungct_model/lungct_loader.py
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

def load_lungct_model(repo_id="draziza/lung-colon-model", filename="lungct.pth", num_classes=2):
    """
    Loads Faster R-CNN from a HF checkpoint (state_dict) or full model.
    If the HF repo is private, set env var HF_TOKEN in your deployment.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token = os.getenv("HF_TOKEN")  # leave empty if public

    # download file from HF
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Case A: a state_dict (common)
    if isinstance(checkpoint, (dict, torch.collections.OrderedDict)):
        model = models.detection.fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=num_classes
        )
        # load weights (ignore missing/unexpected to be robust)
        model.load_state_dict(checkpoint, strict=False)
    else:
        # Case B: a full torch model object
        model = checkpoint

    model.to(device)
    model.eval()
    return model, device

def predict_lungct(model, device, transform, image):
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    return outputs[0]
