#!/usr/bin/env python3
"""
Stop hook — auto-records newly written incident runbooks to long-term memory.

Fires after each Claude turn. Scans incidents/ for .md runbooks not yet present
in incidents.jsonl (matched by runbook_file field), parses them, and appends a
record. Skips plan- files and runbooks with no root cause section (not yet resolved).
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
INCIDENTS_DIR = PROJECT_ROOT / "incidents"
INCIDENTS_FILE = PROJECT_ROOT / ".claude" / "memory" / "episodic" / "incidents.jsonl"


def already_recorded(runbook_rel: str) -> bool:
    if not INCIDENTS_FILE.exists():
        return False
    with open(INCIDENTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("runbook_file") == runbook_rel:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def parse_runbook(path: Path) -> dict:
    text = path.read_text()

    def extract(pattern, flags=0, group=1, default=""):
        m = re.search(pattern, text, flags)
        return m.group(group).strip() if m else default

    title = extract(r"##\s+Incident:\s+(.+)")
    if not title:
        title = path.stem.replace("-", " ").title()

    severity = extract(r"\*\*Severity\*\*:\s*(P\d|unknown)", re.IGNORECASE) or "unknown"

    affected_raw = extract(r"\*\*Affected\*\*:\s*(.+)")
    affected_services = [s.strip() for s in re.split(r"[,/]", affected_raw) if s.strip()]

    detection = extract(r"\*\*Detection\*\*:\s*(.+)") or "unknown"

    root_cause = extract(r"###\s+Root Cause\s*\n(.*?)(?=###|\Z)", re.DOTALL)
    resolution = extract(r"###\s+Resolution Steps?\s*\n(.*?)(?=###|\Z)", re.DOTALL)
    prevention = extract(r"###\s+Prevention\s*\n(.*?)(?=###|\Z)", re.DOTALL)

    # Infer platform from content
    lower = text.lower()
    layers = []
    if any(k in lower for k in ("kubectl", " pod ", "namespace", "deployment", "daemonset")):
        layers.append("k8s")
    if any(k in lower for k in ("gcloud", " gcp ", "cloud sql", "gke", "cloud run")):
        layers.append("gcp")
    if any(k in lower for k in (" aws ", "lambda", " s3 ", "iam policy", "cloudwatch")):
        layers.append("aws")
    if "terraform" in lower and not layers:
        layers.append("terraform")
    platform = "full-stack" if len(layers) > 1 else (layers[0] if layers else "unknown")

    # Timeline bullet count as proxy for duration
    timeline_hits = re.findall(r"^\s*[-*]\s+\d{2}:\d{2}", text, re.MULTILINE)
    duration_minutes = len(timeline_hits) * 5  # rough estimate

    return {
        "id": path.stem,
        "title": title,
        "severity": severity,
        "platform": platform,
        "affected_services": affected_services,
        "symptoms": [],
        "root_cause": root_cause[:600],
        "root_cause_category": "",
        "resolution": resolution[:600],
        "environment": "prod",
        "duration_minutes": duration_minutes,
        "detection": detection,
        "runbook_file": f"incidents/{path.name}",
        "lessons": prevention[:300],
        "auto_recorded": True,
    }


def record_incident(incident: dict) -> None:
    INCIDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    incident["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with open(INCIDENTS_FILE, "a") as f:
        f.write(json.dumps(incident) + "\n")


def main():
    payload = {}
    try:
        payload = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, OSError):
        pass

    # stop_hook_active=True means Claude is stopping after being blocked by us.
    # Always allow; we only use `continue`, never `block`.
    if payload.get("stop_hook_active"):
        print(json.dumps({"continue": True}))
        return

    if not INCIDENTS_DIR.exists():
        print(json.dumps({"continue": True}))
        return

    recorded = []
    for runbook in sorted(INCIDENTS_DIR.glob("*.md")):
        if runbook.name.startswith("plan-"):
            continue
        rel = f"incidents/{runbook.name}"
        if already_recorded(rel):
            continue
        incident = parse_runbook(runbook)
        # Only record runbooks that have a root cause — incomplete drafts are skipped
        if not incident["root_cause"]:
            continue
        record_incident(incident)
        recorded.append(runbook.name)

    if recorded:
        print(
            f"[Memory] Auto-recorded {len(recorded)} runbook(s) to incidents.jsonl: "
            + ", ".join(recorded),
            file=sys.stderr,
        )

    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
