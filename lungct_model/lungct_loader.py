# lungct_loader.py
import torch

def load_lungct_model(device=None, model_path="lungct.pth"):
    """
    Load the lung CT detection model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading Lung CT model from {model_path} on {device} ...")

    # Load model
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()

    return model, device


def predict_lungct(model, device, transform, image):
    """
    Run inference on a single image.
    - image: PIL image
    - transform: preprocessing transform
    """
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)[0]  # detection models usually return a dict

    return outputs
