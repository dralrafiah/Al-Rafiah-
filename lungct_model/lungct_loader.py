import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download


def load_lungct_model(repo_id="draziza/lung-colon-model", filename="lungct.pth"):
    """
    Load Lung CT model from HuggingFace repo.
    Handles both full model objects and state_dict checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model file from HuggingFace
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Case 1: Full model saved -> has eval()
    if hasattr(checkpoint, "eval"):
        model = checkpoint
    else:
        # Case 2: state_dict only
        model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device


def predict_lungct(model, device, transform, image):
    """
    Run inference on a CT scan image.
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
