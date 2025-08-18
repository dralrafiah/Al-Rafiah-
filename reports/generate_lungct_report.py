from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import os
import datetime
import uuid

# --- Helper for interpretation ---
def get_lung_nodule_interpretation(confidence, priority):
    try:
        confidence = float(str(confidence).replace('%',''))
    except (TypeError, ValueError):
        confidence = 0.0

    if confidence >= 95:
        interpretation = "Very High"
        recommendation = "Immediate clinical follow-up recommended."
        color = colors.green
    elif confidence >= 85:
        interpretation = "High"
        recommendation = "Strong likelihood, review by pulmonologist suggested."
        color = colors.limegreen
    elif confidence >= 70:
        interpretation = "Moderate"
        recommendation = "Moderately confident, clinical correlation advised."
        color = colors.orange
    else:
        interpretation = "Low"
        recommendation = "Low confidence, manual radiologist review required."
        color = colors.red

    if priority and str(priority).lower() in ["high","critical"]:
        recommendation += " Urgent attention needed."

    return interpretation, recommendation, color

# --- Main report generation ---
def generate_lungct_report(detections, summary, purpose, observations, info_note, user="User", output_path="reports/Alrafiah_AI_Report_LungCT.pdf"):
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # Styles
    styles.add(ParagraphStyle(name="Heading", fontName="HeiseiMin-W3", fontSize=18, spaceAfter=12, textColor=colors.darkgreen))
    styles.add(ParagraphStyle(name="SubHeading", fontName="HeiseiMin-W3", fontSize=14, spaceAfter=8, textColor=colors.darkblue))
    styles.add(ParagraphStyle(name="Body", fontName="HeiseiMin-W3", fontSize=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Disclaimer", fontName="HeiseiMin-W3", fontSize=10, spaceAfter=6, textColor=colors.red))
    
    # Header
    try:
        logo_path = "assets/alrafiah_logo.png"
        story.append(Image(logo_path, width=80, height=40))
    except:
        pass
    story.append(Paragraph("Al-Rafiah Medical AI Report", styles["Heading"]))
    story.append(Spacer(1,6))

    # Case Identification
    case_id = f"LC_{datetime.datetime.now().strftime('%Y%m%d')}_{uuid.uuid4().hex[:8].upper()}"
    analysis_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Paragraph("1. CASE IDENTIFICATION", styles["SubHeading"]))
    story.append(Paragraph(f"Case ID: {case_id}", styles["Body"]))
    story.append(Paragraph(f"Analysis Date & Time: {analysis_datetime}", styles["Body"]))
    story.append(Paragraph(f"Generated for: {user}", styles["Body"]))
    story.append(Spacer(1,12))

    # Summary
    story.append(Paragraph("2. SUMMARY", styles["SubHeading"]))
    story.append(Paragraph(summary, styles["Body"]))
    story.append(Spacer(1,12))

    # Purpose
    if purpose:
        story.append(Paragraph("3. PURPOSE", styles["SubHeading"]))
        story.append(Paragraph(purpose, styles["Body"]))
        story.append(Spacer(1,12))

    # Observations
    if observations:
        story.append(Paragraph("4. OBSERVATIONS", styles["SubHeading"]))
        story.append(Paragraph(observations, styles["Body"]))
        story.append(Spacer(1,12))

    # Detected Nodules
    if detections and len(detections) > 0:
        story.append(Paragraph("5. DETECTED NODULES", styles["SubHeading"]))
        table_data = [["Nodule ID", "Confidence (%)", "Location", "Size (mm)", "Priority", "Interpretation", "Recommendation"]]
        for det in detections:
            confidence = det.get("confidence",0)
            interpretation, recommendation, color = get_lung_nodule_interpretation(confidence, det.get("priority","Normal"))
            try:
                confidence_display = f"{float(str(confidence).replace('%','')):.1f}"
            except:
                confidence_display = "-"
            table_data.append([
                det.get("nodule_id","-"),
                confidence_display,
                det.get("location","-"),
                det.get("size","-"),
                det.get("priority","-"),
                Paragraph(f'<font color="{color}">{interpretation}</font>', styles["Body"]),
                Paragraph(recommendation, styles["Body"])
            ])
        table = Table(table_data, hAlign="LEFT", colWidths=[60,60,80,60,60,80,120])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("FONTNAME", (0,0), (-1,-1), "HeiseiMin-W3")
        ]))
        story.append(table)
        story.append(Spacer(1,12))

    # Additional Information
    if info_note:
        story.append(Paragraph("6. ADDITIONAL INFORMATION", styles["SubHeading"]))
        story.append(Paragraph(info_note, styles["Body"]))
        story.append(Spacer(1,12))

    # Disclaimer
    story.append(Paragraph("IMPORTANT DISCLAIMER", styles["Disclaimer"]))
    disclaimer_text = (
        "This AI-generated report is for research and educational purposes only. "
        "It should not replace professional medical diagnosis or treatment decisions. "
        "All results require validation by qualified medical professionals. "
        "Clinical correlation and expert review are essential for final diagnosis."
    )
    story.append(Paragraph(disclaimer_text, styles["Body"]))
    story.append(Spacer(1,12))

    # Footer
    story.append(Paragraph(f"Generated by Al-Rafiah Lung CT AI | {analysis_datetime}", styles["Body"]))

    # Build PDF
    doc.build(story)
    return output_path
