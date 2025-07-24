import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

from transformers import ViTForImageClassification, ViTImageProcessor
from huggingface_hub import login
import os

# Define class names for each tissue type
TISSUE_CLASSES = {
    'lung': ['adenocarcinoma', 'benign', 'squamous_carcinoma'],
    'colon': ['adenocarcinoma', 'benign']
}

def load_model(device):
    print("Loading ViT model from Hugging Face ...")

    # إذا كنت على Streamlit Cloud، حط التوكن في Secrets
    login(token=os.environ["HF_TOKEN"])

    processor = ViTImageProcessor.from_pretrained("JawaherAlsharif/lung-colon-vit-model", use_auth_token=True)
    model = ViTForImageClassification.from_pretrained("JawaherAlsharif/lung-colon-vit-model", use_auth_token=True)

    model = model.to(device)
    model.eval()
    return model, processor

def process_image(image_path, processor=None):
    image = Image.open(image_path).convert('RGB')
    if processor:
        inputs = processor(images=image, return_tensors="pt")
        return inputs.pixel_values
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0)

def predict_image(model, image_tensor, device, tissue_type):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()

        if hasattr(model.config, 'id2label') and predicted_idx in model.config.id2label:
            predicted_class = model.config.id2label[predicted_idx]
            confidence = probabilities[predicted_idx].item() * 100
            all_probs = {
                model.config.id2label[i]: probabilities[i].item() * 100
                for i in range(len(probabilities)) if i in model.config.id2label
            }
        else:
            class_names = TISSUE_CLASSES.get(tissue_type.lower(), [])
            if not class_names:
                raise ValueError(f"Invalid tissue type: {tissue_type}. Must be 'lung' or 'colon'.")

            predicted_class = class_names[predicted_idx] if predicted_idx < len(class_names) else f"Unknown Class ({predicted_idx})"
            confidence = probabilities[predicted_idx].item() * 100
            all_probs = {
                class_names[i]: probabilities[i].item() * 100
                for i in range(len(class_names)) if i < len(probabilities)
            }

        return predicted_class, confidence, all_probs

def display_prediction(image_path, predicted_class, confidence, all_probs, tissue_type, output_path=None):
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Input Image\nTissue Type: {tissue_type.capitalize()}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    classes = list(all_probs.keys())
    probs = list(all_probs.values())
    colors = ['green' if cls == 'benign' else 'red' for cls in classes]

    y_pos = np.arange(len(classes))
    bars = plt.barh(y_pos, probs, color=colors)
    plt.yticks(y_pos, classes)
    plt.xlabel('Probability (%)')
    plt.title('Prediction Results')

    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{probs[i]:.1f}%', va='center')

    cancer_status = "CANCER DETECTED" if predicted_class != 'benign' else "NO CANCER DETECTED"
    textstr = f"\n{cancer_status}\n\nPrediction: {predicted_class}\nConfidence: {confidence:.1f}%"
    props = dict(boxstyle='round', facecolor='wheat' if predicted_class != 'benign' else 'lightgreen', alpha=0.5)
    plt.text(0.5, -0.15, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center', bbox=props)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        print(f"Visualization saved to: {output_path}")
    plt.show()

# Main for script usage
def main():
    parser = argparse.ArgumentParser(description='Cancer Detection using Vision Transformer')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--tissue_type', type=str, required=True, choices=['lung', 'colon'], help='Tissue type')
    parser.add_argument('--output', type=str, default=None, help='Optional: path to save visualization')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, processor = load_model(device)
    image_tensor = process_image(args.image, processor)
    predicted_class, confidence, all_probs = predict_image(model, image_tensor, device, args.tissue_type)

    print("="*50)
    print(f"Tissue Type: {args.tissue_type.capitalize()}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")
    for cls, prob in all_probs.items():
        print(f"  {cls}: {prob:.2f}%")
    print(f"\nRESULT: {'NO CANCER DETECTED' if 'benign' in predicted_class.lower() else f'CANCER DETECTED ({predicted_class})'}")
    print("="*50)

    display_prediction(args.image, predicted_class, confidence, all_probs, args.tissue_type, args.output)

# ✅ Optimized callable function for Streamlit
_loaded_model = None
_loaded_processor = None

def predict_vit(image_pil, tissue_type):
    global _loaded_model, _loaded_processor

    if tissue_type not in ['lung', 'colon']:
        raise ValueError("Invalid tissue type. Must be 'lung' or 'colon'")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if _loaded_model is None or _loaded_processor is None:
        _loaded_model, _loaded_processor = load_model(device)

    inputs = _loaded_processor(images=image_pil, return_tensors="pt")
    image_tensor = inputs.pixel_values.to(device)

    predicted_class, confidence, _ = predict_image(_loaded_model, image_tensor, device, tissue_type)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    main()
