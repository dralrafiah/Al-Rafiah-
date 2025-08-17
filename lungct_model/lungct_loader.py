import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

def load_lungct_model(repo_id="draziza/lung-colon-model", filename="lungct.pth"):
    """
    Load Lung CT model from HuggingFace repo.
    Always rebuilds Faster R-CNN and loads the state_dict weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model file from HF
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Load state_dict (weights)
    checkpoint = torch.load(model_path, map_location=device)

    # Rebuild model and load weights
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
