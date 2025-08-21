from fpdf import FPDF
from datetime import datetime
import uuid
from PIL import Image
import numpy as np
import os
import tempfile

def generate_lung_colon_report(
    organ: str,
    analysis_type: str,
    diagnosis: str,
    confidence: float,
    highlighted_image,   
    output_path: str = "Alrafiah_AI_Report.pdf"
) -> str:
    """
    Generate a PDF report for Lung/Colon models.

    Args:
        organ: Name of the organ (Lung/Colon)
        analysis_type: e.g. "Histology"
        diagnosis: human-readable diagnosis returned by the model runner (do not recompute)
        confidence: numeric confidence in percent (0-100)
        highlighted_image: either a file path or an in-memory image (PIL.Image or numpy.ndarray)
        output_path: destination PDF path

    Returns:
        output_path (str)
    """

    # Helper: persist any in-memory image to temporary PNG and return the path
    def _ensure_image_path(img) -> str:
        if img is None:
            return None
        # If already a path
        if isinstance(img, str) and os.path.exists(img):
            return img
        # If numpy array
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img.astype("uint8"))
        elif hasattr(img, "save") and isinstance(img, Image.Image):
            pil_img = img
        else:
            raise ValueError("highlighted_image must be a file path, PIL.Image, or numpy.ndarray")

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        tmp_name = tmp.name
        tmp.close()
        pil_img.save(tmp_name, format="PNG")
        return tmp_name

    # Prepare image paths
    inserted_images = []
    highlighted_path = None
    try:
        highlighted_path = _ensure_image_path(highlighted_image)
    except Exception:
        highlighted_path = None

    # Compute basic metadata
    organ = organ if organ in ["Lung", "Colon"] else "Unknown"
    case_prefix = "LC" if organ == "Lung" else ("CC" if organ == "Colon" else "AI")
    case_id = f"{case_prefix}_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8].upper()}"
    analysis_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Build PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    try:
        logo_path = "assets/alrafiah_logo.png"
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=30)
    except Exception:
        pass

    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(80)
    pdf.cell(30, 10, "Al-Rafiah Medical AI Report", ln=True, align='C')
    pdf.ln(14)

    # 1. Case identification
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "1. CASE IDENTIFICATION", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f"Case ID: {case_id}", ln=True)
    pdf.cell(0, 7, f"Analysis Date & Time: {analysis_datetime}", ln=True)
    pdf.cell(0, 7, f"Organ System: {organ}", ln=True)
    pdf.ln(8)

    # 2. Image analysis summary
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "2. IMAGE ANALYSIS SUMMARY", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 7, f"Diagnosis (model): {diagnosis}", ln=True)
    pdf.cell(0, 7, f"Propability of cancer presence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 7, f"Analysis Type: {analysis_type}", ln=True)
    pdf.ln(8)

    # Include highlighted image if present
    try:
        if highlighted_path is not None and os.path.exists(highlighted_path):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 7, "Diagnosis Image:", ln=True)
            pdf.ln(2)
            desired_w = 100  # width in mm, adjust as needed
            page_width = pdf.w
            x_offset = (page_width - desired_w) / 2  # center the image
            pdf.image(highlighted_path, x=x_offset, w=desired_w)
            pdf.ln(6)
            inserted_images.append(highlighted_path)

    except Exception:
        pdf.ln(2)

    # 3. Clinical notes / recommendation
    pdf.set_font("Arial", "B", 13)
    pdf.cell(0, 8, "3. CLINICAL NOTES", ln=True)
    pdf.ln(4)
    pdf.set_font("Arial", "", 11)
    rec_text = (
        "This report presents AI-based analysis results. It is intended as an aid for clinicians and pathologists. "
        "All AI findings should be reviewed by qualified specialists and correlated with clinical and laboratory data."
    )
    pdf.multi_cell(0, 6, rec_text)
    pdf.ln(6)

    # Disclaimer
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(150, 0, 0)
    pdf.cell(0, 8, "IMPORTANT DISCLAIMER", ln=True)
    pdf.ln(3)
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    disclaimer_text = (
        "This AI-generated report is for research and educational purposes only. "
        "It does not substitute professional medical diagnosis or treatment. "
        "Consult a specialist for final diagnosis and management."
    )
    pdf.multi_cell(0, 6, disclaimer_text)
    pdf.ln(6)

    # Footer
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated by Al-Rafiah {organ} Cancer AI | {analysis_datetime}", ln=True, align='C')

    # Save PDF
    pdf.output(output_path)

    # Cleanup temporary highlighted image file if we created one
    try:
        # If we created a temp file and it's not the original path, remove it
        if highlighted_path is not None and highlighted_path not in inserted_images:
            os.remove(highlighted_path)
    except Exception:
        pass

    return output_path
