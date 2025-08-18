from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import os
import datetime

def generate_lungct_report(detections, summary, purpose, observations, info_note, user="User"):
    """
    Generate a PDF report for Lung CT analysis
    """

    # ✅ Use Unicode-capable font (fixes latin-1 error)
    pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

    # Setup file
    output_path = "reports/Alrafiah_AI_Report_LungCT.pdf"
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Custom styles with Unicode font
    styles.add(ParagraphStyle(name="Heading", fontName="HeiseiMin-W3", fontSize=16, spaceAfter=12))
    styles.add(ParagraphStyle(name="Body", fontName="HeiseiMin-W3", fontSize=12, spaceAfter=8))

    # Title
    story.append(Paragraph("Al-Rafiah Medical AI Report", styles["Heading"]))
    story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Body"]))
    story.append(Paragraph(f"Generated for: {user}", styles["Body"]))
    story.append(Spacer(1, 12))

    # Sections
    story.append(Paragraph("📌 Summary", styles["Heading"]))
    story.append(Paragraph(summary, styles["Body"]))
    story.append(Spacer(1, 12))

    if purpose:
        story.append(Paragraph("🎯 Purpose", styles["Heading"]))
        story.append(Paragraph(purpose, styles["Body"]))
        story.append(Spacer(1, 12))

    if observations:
        story.append(Paragraph("🔬 Observations", styles["Heading"]))
        story.append(Paragraph(observations, styles["Body"]))
        story.append(Spacer(1, 12))

    # Detected nodules
    if detections and len(detections) > 0:
        story.append(Paragraph("📍 Detected Nodules", styles["Heading"]))
        table_data = [["Nodule ID", "Confidence", "Location", "Size", "Priority"]]
        for det in detections:
            table_data.append([
                det["nodule_id"],
                det["confidence"],
                det["location"],
                det["size"],
                det["priority"],
            ])
        table = Table(table_data, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("GRID", (0,0), (-1,-1), 0.5, colors.black),
            ("FONTNAME", (0,0), (-1,-1), "HeiseiMin-W3"),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

    # Info note
    if info_note:
        story.append(Paragraph("ℹ️ Additional Information", styles["Heading"]))
        story.append(Paragraph(info_note, styles["Body"]))
        story.append(Spacer(1, 12))

    # Build PDF
    doc.build(story)
    return output_path
