# lungct_model/lungct_loader.py

import torch
from huggingface_hub import hf_hub_download

def load_lungct_model(repo_id="draziza/lung-colon-model", filename="lungct.pth"):
    """
    Downloads and loads the Lung CT model from Hugging Face Hub.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model, device

def predict_lungct(model, device, transform, image):
    """
    Runs inference on a given PIL image using the Lung CT model.
    """
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)[0]
    return outputs
