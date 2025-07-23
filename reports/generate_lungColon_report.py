from fpdf import FPDF
from datetime import datetime
import uuid

def generate_vit_report(analysis_type, result, confidence, image_name, output_path="Alrafiah_AI_Report_LungColon.pdf"):
    """
    Generate PDF report for ViT lung/colon model ONLY - shows exactly what model provides
    
    Args:
        analysis_type: Type of analysis (e.g., "Histology") 
        result: ViT model result (e.g., "colon_aca", "lung_n")
        confidence: Confidence percentage (e.g., 99.7)
        image_name: Name of analyzed image
        output_path: Path to save PDF
    """
    
    # Parse ViT result - keep exactly what model provides
    vit_class = result  # e.g., "colon_aca", "lung_n", "lung_scc"
    
    # Determine organ and case prefix from ViT class
    if vit_class.startswith("lung"):
        organ = "Lung"
        case_prefix = "LC"
    elif vit_class.startswith("colon"):
        organ = "Colon" 
        case_prefix = "CC"
    else:
        organ = "Unknown"
        case_prefix = "AI"
    
    # Generate case ID
    case_id = f"{case_prefix}_{datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8].upper()}"
    analysis_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()

    # --- Header with Logo ---
    try:
        logo_path = "assets/alrafiah_logo.png"
        pdf.image(logo_path, x=10, y=8, w=30)
    except:
        pass  # Continue without logo if not found

    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(80)
    pdf.cell(30, 10, "Al-Rafiah Medical AI Report", ln=True, align='C')
    pdf.ln(15)

    # --- 1. Case Identification Section ---
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(0, 10, "1. CASE IDENTIFICATION", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Case ID: {case_id}", ln=True)
    pdf.cell(0, 8, f"Analysis Date & Time: {analysis_datetime}", ln=True)
    pdf.cell(0, 8, f"Image File: {image_name}", ln=True)
    pdf.cell(0, 8, f"Organ System: {organ}", ln=True)
    pdf.ln(10)

    # --- 2. Image Analysis Summary Section ---
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(0, 10, "2. IMAGE ANALYSIS SUMMARY", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    
    # Determine basic cancer status from ViT class (minimal interpretation)
    cancerous_classes = ["lung_aca", "lung_scc", "colon_aca"]
    cancer_status = "Cancerous" if vit_class in cancerous_classes else "Non-cancerous"
    
    pdf.cell(0, 8, f"Classification Result: {cancer_status}", ln=True)
    pdf.cell(0, 8, f"Model Confidence Score: {confidence:.1f}%", ln=True)
    pdf.cell(0, 8, f"Analysis Type: {analysis_type}", ln=True)
    pdf.ln(10)

    # --- 3. Model Output Section (What ViT actually provides) ---
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(0, 10, "3. MODEL OUTPUT", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 8, f"Model Prediction Class: {vit_class}", ln=True)
    pdf.cell(0, 8, f"Raw Confidence Score: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    # Add explanation of ViT classes
    class_explanation = get_vit_class_explanation(vit_class)
    pdf.cell(0, 8, f"Class Description: {class_explanation}", ln=True)
    pdf.ln(5)

    # Add confidence interpretation
    confidence_level, interpretation = get_vit_confidence_interpretation(confidence)
    pdf.cell(0, 8, f"Confidence Level: {confidence_level} ({confidence:.1f}%)", ln=True)
    pdf.multi_cell(0, 8, f"Interpretation: {interpretation}")
    pdf.ln(10)

    # --- 4. Clinical Notes Section ---
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(0, 10, "4. CLINICAL NOTES", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(0, 0, 0)
    
    # ViT-specific clinical recommendations
    recommendation = get_vit_clinical_recommendation(confidence, cancer_status, organ)
    pdf.multi_cell(0, 8, f"Recommendation: {recommendation}")
    pdf.ln(5)

    # --- Important Disclaimer ---
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(150, 0, 0)
    pdf.cell(0, 10, "IMPORTANT DISCLAIMER", ln=True)
    pdf.ln(2)

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    disclaimer_text = ("This AI-generated report is intended for research and educational purposes only. "
                      "It should not be used as a substitute for professional medical diagnosis or treatment decisions. "
                      "All results require validation by qualified medical professionals. "
                      "Clinical correlation and expert pathologist review are essential for final diagnosis.")
    
    pdf.multi_cell(0, 6, disclaimer_text)
    pdf.ln(10)

    # --- Footer ---
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f"Generated by Al-Rafiah {organ} Cancer AI | {analysis_datetime}", ln=True, align='C')

    # Save PDF
    pdf.output(output_path)
    return output_path

def get_vit_class_explanation(vit_class):
    """Get explanation of what each ViT class means (minimal interpretation)"""
    explanations = {
        "lung_aca": "Lung tissue classified as adenocarcinoma",
        "lung_scc": "Lung tissue classified as squamous cell carcinoma", 
        "lung_n": "Lung tissue classified as normal/benign",
        "colon_aca": "Colon tissue classified as adenocarcinoma",
        "colon_n": "Colon tissue classified as normal/benign"
    }
    return explanations.get(vit_class, f"Unknown classification: {vit_class}")

def get_vit_confidence_interpretation(confidence):
    """Get confidence interpretation specific to ViT model"""
    if confidence >= 95:
        return "Very High", "Strong model prediction based on image features"
    elif confidence >= 85:
        return "High", "Good model confidence in classification"
    elif confidence >= 70:
        return "Moderate", "Moderately confident prediction, clinical correlation advised"
    else:
        return "Low", "Low model confidence, expert review recommended"

def get_vit_clinical_recommendation(confidence, cancer_status, organ):
    """Get clinical recommendations specific to ViT model results"""
    is_cancerous = cancer_status.lower() == "cancerous"
    
    if confidence >= 90:
        if is_cancerous:
            return f"High confidence {organ.lower()} cancer prediction. Recommend multidisciplinary oncology consultation and staging evaluation."
        else:
            return f"High confidence normal {organ.lower()} tissue prediction. Routine surveillance appropriate per clinical guidelines."
    elif confidence >= 70:
        return f"Moderate confidence {organ.lower()} tissue prediction. Clinical correlation with imaging and expert pathologist review recommended."
    else:
        return f"Low confidence prediction. Manual histopathological review by {organ.lower()} specialist required before clinical decision."