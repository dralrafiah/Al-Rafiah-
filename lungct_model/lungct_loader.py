import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download


def load_lungct_model(repo_id="draziza/lung-colon-model", filename="lungct.pth"):
    """
    Final version:
    Always rebuilds Faster R-CNN model and loads state_dict weights.
    Never returns an OrderedDict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model from HuggingFace
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Rebuild model every time
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # Handle state_dict correctly
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]  # unwrap nested dict
        model.load_state_dict(checkpoint)
    else:
        raise RuntimeError("❌ Unexpected checkpoint format. Expected state_dict, got something else.")

    model.to(device)
    model.eval()
    return model, device


def predict_lungct(model, device, transform, image):
    """
    Run inference on CT image.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    return outputs
