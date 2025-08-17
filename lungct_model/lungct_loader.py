import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

def load_lungct_model(checkpoint_path="lungct.pth"):
    """
    Load FasterRCNN model trained on lung CT nodules.
    Handles state_dict checkpoints.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build base model (no pretrained weights, just structure)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # Replace the head with correct number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    num_classes = 2  # background + nodule
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load your checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, device


def predict_lungct(model, device, image):
    """
    Run inference on a CT scan image.
    """
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    return outputs
