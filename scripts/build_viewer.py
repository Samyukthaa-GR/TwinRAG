"""
Build the standalone twin viewer.

Injects the exported graph payload into viewer/template.html to produce a
single self-contained HTML file — no server, no CDN, no external assets.

Run scripts/export_graph_view.py first.
"""

import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PAYLOAD_PATH = PROJECT_ROOT / "data" / "generated" / "graph_view.json"
TEMPLATE_PATH = PROJECT_ROOT / "viewer" / "template.html"

#: A complete HTML document -- open it, host it, email it.
OUTPUT_PATH = PROJECT_ROOT / "data" / "generated" / "twin_viewer.html"

#: The same page as a bare fragment, for publishers that supply their own
#: <html>/<head> skeleton (Claude Artifacts does this).
FRAGMENT_PATH = PROJECT_ROOT / "data" / "generated" / "twin_viewer.fragment.html"

BASELINE_KEY = "normal"

PAGE_TITLE = "TwinRAG — Net3 Digital Twin"

#: Wrapper for the standalone build. Without a doctype the browser falls
#: into quirks mode, and without the charset the em-dashes, m³ and Δ in
#: the page render as mojibake when served over plain HTTP.
DOCUMENT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Cpath d='M16 3C16 3 6 15 6 21a10 10 0 0 0 20 0C26 15 16 3 16 3Z' fill='%230e7c86'/%3E%3C/svg%3E">
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{ background: #eef1f3; }}
  @media (prefers-color-scheme: dark) {{ body {{ background: #0c1418; }} }}
  :root[data-theme="dark"] body {{ background: #0c1418; }}
  :root[data-theme="light"] body {{ background: #eef1f3; }}
  #themeToggle {{
    position: fixed; top: 10px; right: 12px; z-index: 20;
    font: 500 12px/1 -apple-system, "Segoe UI", Roboto, sans-serif;
    color: #40525c; background: #fff; border: 1px solid #cbd6dc;
    border-radius: 999px; padding: 7px 12px; cursor: pointer;
  }}
  @media (prefers-color-scheme: dark) {{
    #themeToggle {{ color: #a9bcc6; background: #121d23; border-color: #24343c; }}
  }}
  :root[data-theme="dark"] #themeToggle {{ color: #a9bcc6; background: #121d23; border-color: #24343c; }}
  :root[data-theme="light"] #themeToggle {{ color: #40525c; background: #fff; border-color: #cbd6dc; }}
</style>
</head>
<body>
<button type="button" id="themeToggle" aria-label="Switch between light and dark">Theme</button>
{fragment}
<script>
(function () {{
  var root = document.documentElement;
  var saved = null;
  try {{ saved = localStorage.getItem("twinrag-theme"); }} catch (e) {{}}
  if (saved) root.setAttribute("data-theme", saved);
  document.getElementById("themeToggle").addEventListener("click", function () {{
    var dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") &&
        window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try {{ localStorage.setItem("twinrag-theme", next); }} catch (e) {{}}
  }});
}})();
</script>
</body>
</html>
"""


def _humanise(key: str) -> str:
    """
    Turn a scenario key into something a person would say.

    leak_101_sev90_t10_16 -> "Leak at junction 101 — 90% severity, 10:00-16:00"
    """

    if key == BASELINE_KEY:
        return "Normal operation (no fault)"

    match = re.match(r"(.+?)_([^_]+)_sev(\d+)_t(\d+)_(\d+|end)$", key)

    if not match:
        return key

    fault_type, target, severity, start, end = match.groups()

    window = f"{int(start):02d}:00–" + (
        "end of day" if end == "end" else f"{int(end):02d}:00"
    )

    phrasing = {
        "leak": f"Leak at junction {target}",
        "pump_failure": f"Pump {target} outage",
        "blockage": f"Pipe {target} blocked",
    }.get(fault_type, f"{fault_type} at {target}")

    # Blockage and full pump outage are all-or-nothing in the model, so
    # quoting a severity percentage there would overstate what was simulated.
    if fault_type == "leak":
        phrasing += f" — {severity}% severity"

    return f"{phrasing}, {window}"


def _percentile(values, q):
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def main() -> None:
    if not PAYLOAD_PATH.exists():
        raise SystemExit(
            f"Missing {PAYLOAD_PATH.relative_to(PROJECT_ROOT)}. "
            "Run scripts/export_graph_view.py first."
        )

    payload = json.loads(PAYLOAD_PATH.read_text(encoding="utf-8"))

    if BASELINE_KEY not in payload["scenarios"]:
        raise SystemExit(
            f"Scenario '{BASELINE_KEY}' missing from the payload; the viewer "
            "needs it to compute change-vs-normal."
        )

    payload["baseline"] = BASELINE_KEY

    for key, scenario in payload["scenarios"].items():
        scenario["label"] = _humanise(key)

    # --------------------------------------------------
    # Colour domains
    #
    # Pressure uses a 2nd-98th percentile clamp so a single extreme node
    # cannot flatten the ramp for everything else. Deviation uses a
    # symmetric domain so the diverging scale stays centred on zero.
    # --------------------------------------------------

    pressures = []

    for scenario in payload["scenarios"].values():
        for table in scenario["series"]["pressure"].values():
            pressures.extend(v for v in table.values() if v is not None)

    base = payload["scenarios"][BASELINE_KEY]["series"]["pressure"]
    deltas = []

    for scenario in payload["scenarios"].values():
        for stamp, table in scenario["series"]["pressure"].items():
            reference = base.get(stamp, {})
            for asset, value in table.items():
                other = reference.get(asset)
                if value is not None and other is not None:
                    deltas.append(abs(value - other))

    payload["domain"] = {
        "pressure": [
            round(_percentile(pressures, 0.02), 2),
            round(_percentile(pressures, 0.98), 2),
        ],
        "delta": round(max(_percentile(deltas, 0.98), 0.5), 2),
    }

    print("pressure domain:", payload["domain"]["pressure"])
    print("delta domain:  ±", payload["domain"]["delta"])

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    if "__TWIN_DATA__" not in template:
        raise SystemExit("Template is missing the __TWIN_DATA__ placeholder.")

    # The payload sits inside a <script type="application/json"> block, so the
    # only sequence that could break out of it is a literal closing tag.
    encoded = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")

    fragment = template.replace("__TWIN_DATA__", encoded)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    FRAGMENT_PATH.write_text(fragment, encoding="utf-8")

    # The standalone document carries its own <title>, so drop the one the
    # fragment declares rather than emitting it twice.
    body = re.sub(r"^\s*<title>.*?</title>\s*", "", fragment, count=1, flags=re.S)

    OUTPUT_PATH.write_text(
        DOCUMENT.format(title=PAGE_TITLE, fragment=body),
        encoding="utf-8",
    )

    print(f"\nScenarios bundled: {len(payload['scenarios'])}")
    for key in payload["scenarios"]:
        print(f"  {payload['scenarios'][key]['label']}")

    print(
        f"\nWrote {OUTPUT_PATH.relative_to(PROJECT_ROOT)} "
        f"({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)  <- open or host this"
    )

    print(
        f"      {FRAGMENT_PATH.relative_to(PROJECT_ROOT)} "
        f"({FRAGMENT_PATH.stat().st_size / 1024:.0f} KB)  <- for embedding"
    )


if __name__ == "__main__":
    main()
