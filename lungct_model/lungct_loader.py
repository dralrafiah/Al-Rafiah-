import torch
import torchvision
from huggingface_hub import hf_hub_download
from torchvision import transforms

# Define the model architecture exactly like you trained it
def build_lungct_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model

def load_lungct_model(repo_id="draziza/lung-colon-model", filename="lungct.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Download model weights from Hugging Face
    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    # Build architecture
    model = build_lungct_model(num_classes=2)

    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, device

def predict_lungct(model, device, transform, image):
    model.eval()
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)[0]
    return outputs
