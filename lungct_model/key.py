import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from datetime import date
import numpy as np
import pydicom
from huggingface_hub import hf_hub_download, login

# ---------------- Flask config ----------------
app = Flask(__name__)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "static/uploads")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm', 'mhd', 'npy'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------- Model setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Login using token from Streamlit Cloud secrets (not hardcoded!)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN is missing. Please set it in Streamlit Cloud secrets.")
login(token=hf_token)

# Define model architecture
model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

# Download model weights from Hugging Face
model_path = hf_hub_download(
    repo_id="draziza/lung-colon-model",
    filename="lungct.pth",
    token=hf_token
)

state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = T.Compose([T.ToTensor()])

# ---------------- Image loading helpers ----------------
def load_image(path):
    ext = path.rsplit('.', 1)[1].lower()
    if ext == 'dcm':
        ds = pydicom.dcmread(path)
        img = ds.pixel_array
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        img = Image.fromarray(img).convert("RGB")
    elif ext == 'npy':
        arr = np.load(path)
        if len(arr.shape) == 2:
            arr = np.stack([arr]*3, axis=-1)
        img = Image.fromarray(arr).convert("RGB")
    else:
        img = Image.open(path).convert("RGB")
    return img

def get_location_name(box):
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    if y_center < 100:
        vert = "Upper"
    elif y_center < 200:
        vert = "Middle"
    else:
        vert = "Lower"
    if x_center < 100:
        horiz = "Left"
    elif x_center < 200:
        horiz = "Middle"
    else:
        horiz = "Right"
    return f"{vert} {horiz} lung region"

# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def home():
    return "API running"

@app.route("/aimodel", methods=["POST"])
def aimodel():
    if 'file' not in request.files:
        return render_template("report.html", scan_date="Unknown", user="Anonymous User", summary="No file uploaded.", detections=[], purpose="", observations="", info_note="")
    file = request.files['file']
    if file.filename == "" or not allowed_file(file.filename):
        return render_template("report.html", scan_date="Unknown", user="Anonymous User", summary="No valid file selected.", detections=[], purpose="", observations="", info_note="")

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        image = load_image(filepath)
    except Exception:
        return render_template("report.html", scan_date="Unknown", user="Anonymous User", summary="Failed to process image file.", detections=[], purpose="", observations="", info_note="")

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    details = []
    if outputs and len(outputs[0]["boxes"]) > 0:
        for i, box in enumerate(outputs[0]["boxes"]):
            conf = outputs[0]["scores"][i].item()
            if conf > 0.05:
                width = box[2].item() - box[0].item()
                height = box[3].item() - box[1].item()
                approx_size = f"{round(width)} × {round(height)} pixels"
                location = get_location_name(box)
                priority = "High" if conf > 0.7 else "Moderate"
                details.append({
                    "nodule_id": i + 1,
                    "confidence": f"{round(conf * 100, 2)}%",
                    "location": location,
                    "size": approx_size,
                    "priority": priority
                })

    scan_date = date.today().strftime("%Y-%m-%d")
    user = "Anonymous User"
    count = len(details)

    if count > 0:
        summary = f"The AI model has analyzed the provided lung image and identified {count} potential nodules. These findings are meant to assist in screening and should be reviewed by a qualified medical professional for confirmation."
        purpose = "This AI-powered tool assists in the early detection of lung nodules in uploaded images. It provides a fast, preliminary analysis to guide further investigation and does not replace a formal diagnosis by a medical professional."
        observations = ("All detected nodules are small to medium-sized in the provided scan. "
                        "The confidence levels are below typical medical certainty thresholds; further scans are recommended. "
                        "Nodules may appear due to infection, inflammation, benign growth, or malignancy. Only a medical professional can confirm their nature.")
        info_note = ("ℹ️ Important Information: The location and size values are approximate based on the uploaded image. "
                     "For precise clinical measurements, a full medical scan with original imaging data is required.")
    else:
        summary = "No nodules detected above the threshold."
        purpose = ""
        observations = ""
        info_note = "ℹ️ Important Information: The location and size values are approximate based on the uploaded image."

    return render_template("report.html",
                           scan_date=scan_date,
                           user=user,
                           summary=summary,
                           detections=details,
                           purpose=purpose,
                           observations=observations,
                           info_note=info_note)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
