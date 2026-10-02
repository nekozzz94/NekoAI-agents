#!/usr/bin/env python3
"""
PostToolUse hook — appends every Bash command and its output to a daily markdown log.
Log file: logs/YYYY-MM-DD.md  (one file per calendar day)
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"continue": True}))
        return

    if payload.get("tool_name") != "Bash":
        print(json.dumps({"continue": True}))
        return

    tool_input = payload.get("tool_input", {})
    command = tool_input.get("command", "").strip()
    description = tool_input.get("description", "").strip()
    tool_response = payload.get("tool_response", {})
    if isinstance(tool_response, dict):
        output = tool_response.get("output", "").strip()
    else:
        output = str(tool_response).strip()

    if not command:
        print(json.dumps({"continue": True}))
        return

    now = datetime.now(timezone.utc)
    log_file = LOGS_DIR / f"{now.strftime('%Y-%m-%d')}.md"
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Write header if file is new
    is_new = not log_file.exists() or log_file.stat().st_size == 0
    with open(log_file, "a") as f:
        if is_new:
            f.write(f"# Bash Command Log — {now.strftime('%Y-%m-%d')}\n\n")

        f.write(f"## {now.strftime('%H:%M:%S UTC')}\n\n")
        if description:
            f.write(f"_{description}_\n\n")
        f.write(f"```bash\n{command}\n```\n\n")
        if output:
            # Truncate very long outputs to keep the log readable
            if len(output) > 4000:
                output = output[:4000] + "\n... (truncated)"
            f.write(f"```\n{output}\n```\n\n")
        f.write("---\n\n")

    print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
