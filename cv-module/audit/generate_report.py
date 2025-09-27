# cv-module/audit/generate_report.py
import json, os, pandas as pd
from pathlib import Path
from jinja2 import Template
from datetime import datetime

from db import insert_report   # ✅ ab db helper use karenge

# HTML template with two tables
TEMPLATE = """ 
<html>
<head><title>Model Audit</title></head>
<body>
    <h1>Model Audit Report</h1>
    <h2>Summary</h2>
    <ul>
        <li>Model: {{ model }}</li>
        <li>Date: {{ date }}</li>
    </ul>

    <h2>Per-class metrics</h2>
    {{ metrics_html|safe }}

    <h2>Detected Violations Log</h2>
    {{ violations_html|safe }}

    <h2>Adversarial robustness (summary)</h2>
    <pre>{{ adv_summary }}</pre>

    <h2>Drift / Stability</h2>
    <pre>{{ drift_text }}</pre>

    <h2>Explainability samples</h2>
    {% for p in examples %}
        <div><h3>{{ p.title }}</h3><img src="{{ p.img }}" width="600"/></div>
    {% endfor %}
</body>
</html>
"""

def generate(model_name, metrics_df, violations_df=None, adv_json=None, examples=None, out_html="audit_report.html"):
    adv = json.load(open(adv_json)) if adv_json and Path(adv_json).exists() else {}
    adv_summary = json.dumps(adv, indent=2)[:4000]
    drift_text = "Drift metrics not run."
    examples = examples or []

    # Handle violations_df (may be None)
    if violations_df is not None and not violations_df.empty:
        violations_html = violations_df.to_html(index=False)
    else:
        violations_html = "<p>No violations recorded.</p>"

    # Render HTML
    html = Template(TEMPLATE).render(
        model=model_name,
        date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics_html=metrics_df.to_html(index=False),
        violations_html=violations_html,
        adv_summary=adv_summary,
        drift_text=drift_text,
        examples=examples
    )

    # Save to file
    with open(out_html, "w") as f:
        f.write(html)
    print("Saved report:", out_html)

    # Save in MongoDB (via helper)
    report_doc = {
        "model": model_name,
        "date": datetime.now(),
        "metrics": metrics_df.to_dict(orient="records"),
        "violations": violations_df.to_dict(orient="records") if violations_df is not None else [],
        "adv_summary": adv_summary,
        "examples": examples,
        "report_path": out_html
    }
    insert_report(report_doc)   # ✅ helper se insert
    print("Saved report in MongoDB too.")
