# api.py
from flask import Flask, jsonify, send_file
from flask_cors import CORS
from db import get_all_violations, get_all_reports

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import datetime
import os

app = Flask(__name__)
CORS(app)

@app.route("/violations")
def get_violations():
    raw_data = get_all_violations(limit=50)

    data = []
    for idx, v in enumerate(raw_data, start=1):
        data.append({
            "id": idx,
            "type": v.get("violation_type", "Unknown"),
            "timestamp": v.get("timestamp", "N/A"),
            "vehicle_no": v.get("vehicle_no", "N/A"),
            "reason": v.get("reason", "N/A"),
            "confidence": v.get("conf", 0.0),
            "snapshot": v.get("snapshot_path")
        })
    return jsonify(data)


@app.route("/audit")
def get_audit():
    reports = get_all_reports(limit=1)
    if not reports:
        return jsonify({"error": "No audit reports found"}), 404
    return jsonify(reports[0])  # send latest


# ✅ PDF generation route
@app.route("/audit/pdf")
def get_audit_pdf():
    reports = get_all_reports(limit=1)
    if not reports:
        return jsonify({"error": "No audit reports found"}), 404
    
    audit = reports[0]
    file_path = "audit_report.pdf"

    # Create PDF
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("📊 Audit Report", styles["Title"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now()}", styles["Normal"]))
    story.append(Spacer(1, 20))

    # Model Info
    story.append(Paragraph(f"Model: {audit.get('model', 'N/A')}", styles["Normal"]))
    story.append(Paragraph(f"Date: {audit.get('date', 'N/A')}", styles["Normal"]))
    story.append(Spacer(1, 15))

    # Metrics Table
    metrics = audit.get("metrics", [])
    if metrics:
        data = [["Class", "Precision", "Recall", "F1"]]
        for m in metrics:
            data.append([m["class"], m["precision"], m["recall"], m["f1"]])

        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(Paragraph("Per-class Metrics:", styles["Heading2"]))
        story.append(table)
        story.append(Spacer(1, 20))

    # Violations Table
    violations = audit.get("violations", [])
    if violations:
        data = [["Type", "Vehicle", "Time", "Reason", "Confidence"]]
        for v in violations:
            data.append([
                v.get("violation_type", "Unknown"),
                v.get("vehicle_no", "N/A"),
                v.get("timestamp", "N/A"),
                v.get("reason", "-"),
                str(v.get("conf", 0))
            ])

        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.grey),
            ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(Paragraph("Violations Log:", styles["Heading2"]))
        story.append(table)

    doc.build(story)

    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
