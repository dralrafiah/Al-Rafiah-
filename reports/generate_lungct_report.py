# reports/generate_lungct_report.py

from fpdf import FPDF
from datetime import datetime
import os
from typing import List, Dict, Optional

# ----------------------------
# Text sanitization
# ----------------------------
def _coerce_latin1(text: Optional[str]) -> str:
    """
    Ensure text is safe for fpdf 1.x (latin-1). Removes unsupported chars (emojis, etc.).
    Also normalizes some common unicode punctuation to ASCII.
    """
    if text is None:
        return ""
    s = str(text)

    # normalize common punctuation
    replacements = {
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201C": '"',  # left double quote
        "\u201D": '"',  # right double quote
        "\u2022": "-",  # bullet
        "\u00A0": " ",  # nbsp
        "\u200B": "",   # zero-width space
        "\uFE0F": "",   # variation selector (emoji)
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # final safety net: strip anything not in latin-1
    return s.encode("latin-1", "ignore").decode("latin-1")


def _clean_dict(d: Dict) -> Dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            out[k] = _coerce_latin1(v)
        else:
            out[k] = v
    return out


def _truncate(s: str, max_chars: int) -> str:
    s = s or ""
    return s if len(s) <= max_chars else (s[: max_chars - 1] + ".")


# ----------------------------
# Themed PDF
# ----------------------------
class ThemedPDF(FPDF):
    brand_green = (19, 71, 52)
    line_gray = (210, 210, 210)

    def header_with_brand(self, logo_path: str, title: str):
        # Logo (optional)
        try:
            if logo_path and os.path.exists(logo_path):
                self.image(logo_path, x=10, y=8, w=28)
        except Exception:
            pass

        # Title
        self.set_font("Arial", "B", 16)
        self.set_text_color(*self.brand_green)
        self.cell(0, 10, _coerce_latin1(title), ln=True, align="C")
        self.ln(6)

        # Divider
        self.set_draw_color(*self.line_gray)
        self.set_line_width(0.4)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def section_title(self, text: str):
        self.set_font("Arial", "B", 13)
        self.set_text_color(*self.brand_green)
        self.cell(0, 8, _coerce_latin1(text), ln=True)
        self.set_text_color(0, 0, 0)

    def info_box(self, text: str):
        self.ln(2)
        self.set_fill_color(245, 245, 245)  # light gray
        self.set_font("Arial", size=10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 6, _coerce_latin1(text), fill=True)
        self.ln(1)

    def disclaimer_box(self, text: str):
        self.ln(2)
        self.set_font("Arial", "B", 12)
        self.set_text_color(150, 0, 0)
        self.cell(0, 7, "IMPORTANT DISCLAIMER", ln=True)
        self.set_font("Arial", size=10)
        self.set_text_color(100, 100, 100)
        self.multi_cell(0, 6, _coerce_latin1(text))
        self.set_text_color(0, 0, 0)


def _para(pdf: FPDF, text: str, size: int = 11):
    pdf.set_font("Arial", size=size)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, _coerce_latin1(text))


def _render_detections_table(pdf: FPDF, detections: List[Dict]):
    """
    Table: | ID | Confidence | Location | Size | Priority |
    """
    # sanitize
    clean = [_clean_dict(d) for d in detections]

    col_w = {
        "ID": 15,
        "Confidence": 28,
        "Location": 70,
        "Size": 32,
        "Priority": 25,
    }
    header_h = 8
    row_h = 7

    # Header
    pdf.ln(2)
    pdf.set_fill_color(230, 238, 234)  # pale green tint
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.2)
    pdf.set_font("Arial", "B", 11)
    pdf.set_text_color(19, 71, 52)
    pdf.cell(col_w["ID"],         header_h, "ID",         border=1, align="C", fill=True)
    pdf.cell(col_w["Confidence"], header_h, "Confidence", border=1, align="C", fill=True)
    pdf.cell(col_w["Location"],   header_h, "Location",   border=1, align="C", fill=True)
    pdf.cell(col_w["Size"],       header_h, "Size",       border=1, align="C", fill=True)
    pdf.cell(col_w["Priority"],   header_h, "Priority",   border=1, align="C", fill=True)
    pdf.ln(header_h)

    # Rows
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(0, 0, 0)
    for det in clean:
        nid  = str(det.get("nodule_id", ""))
        conf = str(det.get("confidence", ""))
        loc  = _truncate(str(det.get("location", "")), max_chars=42)
        size = str(det.get("size", ""))
        pri  = str(det.get("priority", ""))

        pdf.cell(col_w["ID"],         row_h, _coerce_latin1(nid),  border=1, align="C")
        pdf.cell(col_w["Confidence"], row_h, _coerce_latin1(conf), border=1, align="C")
        pdf.cell(col_w["Location"],   row_h, _coerce_latin1(loc),  border=1, align="L")
        pdf.cell(col_w["Size"],       row_h, _coerce_latin1(size), border=1, align="C")
        pdf.cell(col_w["Priority"],   row_h, _coerce_latin1(pri),  border=1, align="C")
        pdf.ln(row_h)

    pdf.ln(2)


# ----------------------------
# Public API
# ----------------------------
def generate_lungct_report(
    detections: List[Dict],
    summary: str,
    purpose: str,
    observations: str,
    info_note: str,
    user: str = "Anonymous User",
    scan_date: Optional[str] = None,
    output_path: Optional[str] = None,
    logo_path: str = "assets/alrafiah_logo.png",
    case_id: Optional[str] = None,
) -> str:
    """
    Create a styled, branded PDF report for Lung CT detections.
    Works with fpdf==1.7.x and avoids latin-1 crashes by sanitizing text.
    """

    # --------- metadata ---------
    ts = datetime.now()
    scan_date = scan_date or ts.strftime("%Y-%m-%d")
    if case_id is None:
        case_id = f"LC_{ts.strftime('%Y%m%d_%H%M%S')}"
    if output_path is None:
        os.makedirs("reports", exist_ok=True)
        output_path = os.path.join("reports", f"LungCT_Report_{ts.strftime('%Y%m%d_%H%M%S')}.pdf")

    # --------- build PDF ---------
    pdf = ThemedPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.header_with_brand(logo_path=logo_path, title="Al-Rafiah Medical AI Report")

    # 1) Case info
    pdf.section_title("1. CASE INFORMATION")
    pdf.set_font("Arial", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.cell(0, 7, _coerce_latin1(f"Case ID: {case_id}"), ln=True)
    pdf.cell(0, 7, _coerce_latin1(f"Scan Date: {scan_date}"), ln=True)
    pdf.cell(0, 7, _coerce_latin1(f"Subject: {user}"), ln=True)
    pdf.cell(0, 7, "Organ System: Lung (CT)", ln=True)
    pdf.ln(3)

    # 2) Findings
    pdf.section_title("2. AI FINDINGS (CT DETECTIONS)")
    if detections and len(detections) > 0:
        _render_detections_table(pdf, detections)
        count = len(detections)
        _para(pdf, f"Detected {count} potential nodules at the chosen confidence threshold.")
    else:
        pdf.set_font("Arial", size=11)
        pdf.set_text_color(80, 80, 80)
        _para(pdf, "No nodules detected above the threshold in this scan.")
    pdf.ln(2)

    # 3) Summary
    if summary:
        pdf.section_title("3. SUMMARY")
        _para(pdf, summary)
        pdf.ln(2)

    # 4) Observations
    if observations:
        pdf.section_title("4. OBSERVATIONS")
        _para(pdf, observations)
        pdf.ln(2)

    # 5) Purpose
    if purpose:
        pdf.section_title("5. PURPOSE")
        _para(pdf, purpose)
        pdf.ln(2)

    # 6) Info note (gray)
    if info_note:
        pdf.info_box(info_note)

    # Disclaimer
    pdf.disclaimer_box(
        "This AI-generated report is for research and educational purposes only. "
        "It must not be used as a substitute for professional medical diagnosis or treatment. "
        "All findings require review by a qualified medical professional."
    )

    # Footer
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.ln(4)
    pdf.cell(0, 8, _coerce_latin1(f"Generated by Al-Rafiah Lung CT AI | {ts.strftime('%Y-%m-%d %H:%M:%S')}"),
             ln=True, align="C")

    # Save
    pdf.output(output_path)
    return os.path.abspath(output_path)
