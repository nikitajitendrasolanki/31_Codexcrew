# cv-module/audit/generate_report.py
import json, os, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from jinja2 import Template

TEMPLATE = """
<html>
<head><title>Model Audit</title></head><body>
<h1>Model Audit Report</h1>
<h2>Summary</h2>
<ul>
<li>Model: {{ model }}</li>
<li>Date: {{ date }}</li>
</ul>
<h2>Per-class metrics</h2>
{{ table_html|safe }}
<h2>Adversarial robustness (summary)</h2>
<pre>{{ adv_summary }}</pre>
<h2>Drift / Stability</h2>
<pre>{{ drift_text }}</pre>
<h2>Explainability samples</h2>
{% for p in examples %}
  <div><h3>{{ p.title }}</h3><img src="{{ p.img }}" width="600"/></div>
{% endfor %}
</body></html>
"""

def generate(model_name, metrics_df, adv_json, examples, out_html="audit_report.html"):
    adv = json.load(open(adv_json)) if adv_json and Path(adv_json).exists() else {}
    adv_summary = json.dumps(adv, indent=2)[:4000]
    drift_text = "Drift metrics not run."  # placeholder
    html = Template(TEMPLATE).render(model=model_name, date=pd.Timestamp.now(), table_html=metrics_df.to_html(index=False),
                                     adv_summary=adv_summary, drift_text=drift_text, examples=examples)
    open(out_html, "w").write(html)
    print("Saved report:", out_html)
