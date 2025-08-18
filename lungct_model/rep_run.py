# lungct_model/rep_run.py
import os
from reports.generate_lungct_report import generate_lungct_report

def run_lungct_report(outputs, uploaded_image_name="uploaded_image.png"):
    """
    Build detections + call generate_lungct_report.
    Returns the path to the generated PDF.
    """
    boxes = outputs["boxes"].cpu().numpy() if "boxes" in outputs else []
    scores = outputs["scores"].cpu().numpy() if "scores" in outputs else []

    details = []
    for i in range(len(scores)):
        if scores[i] >= 0.5:  # confidence threshold
            x1, y1, x2, y2 = boxes[i]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            approx_size = f"{int(round(w))} x {int(round(h))} pixels"

            # simple location naming
            xc = (x1 + x2) / 2.0
            yc = (y1 + y2) / 2.0
            vert = "Upper" if yc < 170 else ("Middle" if yc < 340 else "Lower")
            horiz = "Left" if xc < 170 else ("Middle" if xc < 340 else "Right")
            location = f"{vert} {horiz} lung region"

            priority = "High" if scores[i] > 0.7 else "Moderate"

            details.append({
                "nodule_id": int(i + 1),
                "confidence": f"{scores[i]*100:.2f}%",
                "location": location,
                "size": approx_size,
                "priority": priority
            })

    # Human-readable text
    if len(details) > 0:
        summary = f"The AI model analyzed the lung CT image and identified {len(details)} potential nodules. "\
                  "These findings are screening aids and require confirmation by a qualified medical professional."
        purpose = "Assist in early detection of lung nodules with a fast, preliminary AI analysis."
        observations = ("Sizes are approximate based on the uploaded image. Confidence levels are below clinical certainty; "
                        "follow-up imaging/clinical evaluation is advised. Nodules can arise from benign or malignant causes.")
        info_note = ("Important Information: The location and size values are approximate based on the uploaded image. "
                     "For precise clinical measurements, please use original DICOM data and standard clinical workflow.")
    else:
        summary = "No nodules detected above the threshold."
        purpose = ""
        observations = ""
        info_note = "Important Information: The location and size values are approximate based on the uploaded image."

    # Generate PDF
    pdf_path = generate_lungct_report(
        detections=details,
        summary=summary,
        purpose=purpose,
        observations=observations,
        info_note=info_note,
        user="Anonymous User",
        logo_path="assets/alrafiah_logo.png",
    )
    return pdf_path
