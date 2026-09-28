"""HTML reporter: a single self-contained file.

Security defaults: inline CSS only, no JavaScript, no remote assets. The
report can be committed beside benchmark results or attached to a PR without
leaking anything or phoning home.
"""
from __future__ import annotations

from jinja2 import Environment

from ..engine import AuditResult
from ..model import Severity

_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>evalint report: {{ model.eval_id }}</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
         margin: 0; color: #1a1a1a; background: #fafafa; line-height: 1.5; }
  .wrap { max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; }
  .verdict { border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; color: #fff; }
  .verdict.blocked { background: #b3261e; }
  .verdict.pass { background: #1e7b34; }
  .verdict h1 { margin: 0 0 4px; font-size: 22px; }
  .verdict .score { font-size: 40px; font-weight: 700; }
  .verdict p { margin: 4px 0 0; opacity: .92; }
  .card { background: #fff; border: 1px solid #e3e3e3; border-radius: 10px;
          padding: 18px 20px; margin-bottom: 16px; }
  .card h2 { margin: 0 0 8px; font-size: 16px; }
  .badge { display: inline-block; font-size: 12px; font-weight: 700; border-radius: 6px;
           padding: 2px 8px; margin-right: 6px; text-transform: uppercase; letter-spacing: .04em; }
  .sev-error { background: #fdecea; color: #b3261e; border: 1px solid #f5c6c2; }
  .sev-high { background: #fff4e5; color: #8a5a00; border: 1px solid #f0d9a8; }
  .sev-medium { background: #eef4ff; color: #1a56db; border: 1px solid #c9dbff; }
  .sev-low { background: #f1f1f1; color: #555; border: 1px solid #ddd; }
  .conf { background: #f1f1f1; color: #444; border: 1px solid #ddd; text-transform: none; }
  .meta { color: #666; font-size: 13px; }
  ul.ev { margin: 8px 0; padding-left: 20px; }
  code.loc { background: #f4f4f4; padding: 1px 6px; border-radius: 4px; font-size: 12.5px; }
  .fix { background: #f6fef7; border-left: 4px solid #1e7b34; padding: 10px 12px;
         border-radius: 0 8px 8px 0; margin-top: 10px; }
  .fix strong { color: #1e7b34; }
  table.map { width: 100%; border-collapse: collapse; font-size: 14px; }
  table.map th, table.map td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }
  table.map th { color: #666; font-weight: 600; width: 34%; }
  .fp { font-family: ui-monospace, monospace; font-size: 12px; color: #888; }
  footer { margin-top: 32px; color: #888; font-size: 13px; }
</style>
</head>
<body>
<div class="wrap">

  <div class="verdict {{ 'blocked' if result.verdict == 'BLOCKED' else 'pass' }}">
    <h1>evalint report: {{ model.eval_id }}</h1>
    <div class="score">{{ result.score }} / 100</div>
    <p>
      {% if result.verdict == "BLOCKED" %}
      <strong>BLOCKED</strong> &mdash; {{ blocked_ids|join(", ") }}: the reported score
      cannot be trusted until these are fixed.
      {% else %}
      <strong>PASS</strong> &mdash; no blocking findings observed under this policy.
      {% endif %}
    </p>
    <p class="meta" style="color:#fff;opacity:.85">Adapter: {{ model.adapter_name }} {{ model.adapter_version }}
    &middot; {{ result.findings|length }} finding(s)
    &middot; integrity score is diagnostic, not a certification.</p>
  </div>

  {% if result.findings %}
  <div class="card">
    <h2>Findings ({{ result.findings|length }})</h2>
    <p class="meta">Every finding carries a confidence label. Precision over recall:
    a finding is only raised on direct evidence or a strong, named signal.</p>
  </div>
  {% for f in result.findings %}
  <div class="card">
    <h2>
      <span class="badge sev-{{ f.severity.value }}">{{ f.severity.value }}</span><span class="badge conf">confidence: {{ f.confidence.value }}</span>{{ f.id }} &mdash; {{ f.title }}
    </h2>
    <p>{{ f.description }}</p>
    {% if f.evidence %}
    <ul class="ev">
      {% for ev in f.evidence %}<li>{{ ev }}</li>{% endfor %}
    </ul>
    {% endif %}
    {% if f.locations %}
    <p>
      {% for loc in f.locations %}<code class="loc">{{ loc.render() }}</code>{% if not loop.last %} {% endif %}{% endfor %}
    </p>
    {% endif %}
    {% if f.remediation %}
    <div class="fix"><strong>Fix:</strong> {{ f.remediation }}</div>
    {% endif %}
    <p class="fp">fingerprint: {{ f.fingerprint }}</p>
  </div>
  {% endfor %}
  {% else %}
  <div class="card">
    <h2>No findings</h2>
    <p class="meta">No blocking findings observed under this policy. This is not a
    certification that the evaluation is sound &mdash; only that this audit's
    checks found nothing to flag.</p>
  </div>
  {% endif %}

  <div class="card">
    <h2>Measurement map</h2>
    <table class="map">
      <tr><th>Dataset</th><td>{{ model.tasks|length }} task(s)</td></tr>
      <tr><th>Evidence boundary</th><td>{{ model.environment.env_vars|length }} env var(s) visible to agent,
        {{ model.environment.mounts|length }} mount(s)</td></tr>
      <tr><th>Grader</th><td>{{ model.grader.kind }}{% if model.grader.verifier_path %} &middot; verifier: {{ model.grader.verifier_path }}{% endif %}</td></tr>
      <tr><th>Runs</th><td>{{ passes }} passed / {{ model.attempts|length }} attempted</td></tr>
    </table>
  </div>

  {% if model.cost_summary %}
  <div class="card">
    <h2>Cost summary</h2>
    <table class="map">
      <tr><th>Attempts / successes</th><td>{{ model.cost_summary.attempts }} / {{ model.cost_summary.successes }}</td></tr>
      <tr><th>Tokens in / out</th><td>{{ model.cost_summary.total_tokens_in }} / {{ model.cost_summary.total_tokens_out }}</td></tr>
      <tr><th>Estimated total</th><td>${{ "%.4f"|format(model.cost_summary.estimated_usd) }}</td></tr>
      <tr><th>Cost per success</th><td>{% if model.cost_summary.cost_per_success_usd is not none %}${{ "%.4f"|format(model.cost_summary.cost_per_success_usd) }}{% else %}n/a (no successes){% endif %}</td></tr>
      {% if model.cost_summary.avg_tool_calls_per_success is not none %}
      <tr><th>Avg tool calls per success</th><td>{{ model.cost_summary.avg_tool_calls_per_success }}</td></tr>
      {% endif %}
    </table>
    <p class="meta">Estimates from ${{ model.cost_summary.price_in_per_1m }}/$ {{ model.cost_summary.price_out_per_1m }} per 1M input/output tokens. Not metered billing.</p>
  </div>
  {% endif %}

  <div class="card">
    <h2>Reproducibility manifest</h2>
    <table class="map">
      <tr><th>Adapter</th><td>{{ model.adapter_name }} {{ model.adapter_version }}</td></tr>
      {% for fname, digest in model.digests.items() %}
      <tr><th>{{ fname }}</th><td class="fp">sha256:{{ digest[:16] }}&hellip;</td></tr>
      {% endfor %}
    </table>
    {% if model.unsupported %}
    <p class="meta">Coverage gaps &mdash; seen but not translated into the integrity model:</p>
    <ul class="ev meta">
      {% for u in model.unsupported %}<li>{{ u }}</li>{% endfor %}
    </ul>
    {% endif %}
  </div>

  <footer>
    Generated by evalint {{ version }}. Secret values are never stored in reports.
    Offline, read-only audit: input files were not modified.
  </footer>

</div>
</body>
</html>
"""


def render_html(result: AuditResult, version: str = "0.1.0") -> str:
    env = Environment(autoescape=True)
    template = env.from_string(_TEMPLATE)
    passes = sum(1 for a in result.model.attempts if a.status == "pass")
    blocked_ids = sorted({f.id for f in result.blocked_by})
    return template.render(result=result, model=result.model, passes=passes, blocked_ids=blocked_ids, version=version)
