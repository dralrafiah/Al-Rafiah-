# lungct_model/rep_run.py

from reports.generate_lungct_report import generate_lungct_report
from check_lungct import run_inference   # we’ll make sure check_lungct has this

# Run inference on a test file (replace with your input path)
detections, summary, purpose, observations, info_note = run_inference("test.png")

# Generate PDF report
pdf_path = generate_lungct_report(
    detections=detections,
    summary=summary,
    purpose=purpose,
    observations=observations,
    info_note=info_note,
    user="Anonymous User"
)

print(f"✅ Report generated at {pdf_path}")

