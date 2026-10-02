#!/usr/bin/env python3
"""
PreToolUse hook — blocks mutating AWS CLI commands.
Reads JSON from stdin (Claude Code hook payload), checks .tool_input.command,
and returns {"continue": false, "stopReason": "..."} for any mutating aws call.
"""
import json
import re
import sys

# Operations that write, mutate, or execute — blocked
MUTATING_VERBS = {
    "create", "delete", "modify", "update", "put", "set",
    "attach", "detach", "associate", "disassociate",
    "authorize", "revoke",
    "register", "deregister",
    "allocate", "release",
    "enable", "disable",
    "import", "export",
    "restore", "rotate", "reset", "cancel", "schedule",
    "tag", "untag",
    "add", "remove", "replace",
    "flush", "purge", "truncate",
    "deploy", "execute", "apply",
    "copy",  # ec2 copy-image etc
}

# Full operation strings that are mutating regardless of prefix
MUTATING_EXACT = {
    # EC2 lifecycle
    "run-instances", "terminate-instances",
    "start-instances", "stop-instances", "reboot-instances",
    "hibernate-instances",
    # Lambda execution
    "invoke",
    # SNS
    "publish", "subscribe", "unsubscribe",
    # SQS
    "send-message", "send-message-batch",
    "delete-message", "delete-message-batch",
    "change-message-visibility", "change-message-visibility-batch",
    # Route53
    "change-resource-record-sets", "change-tags-for-resource",
    # KMS
    "generate-data-key", "generate-data-key-without-plaintext",
    "generate-random", "encrypt", "decrypt", "re-encrypt",
    "sign", "verify",
    # CloudFormation
    "execute-change-set",
    # S3 surface commands (aws s3 <op>)
    "cp", "mv", "rm", "sync", "mb", "rb",
}

# Read-only operations that start with a normally-mutating verb — allow through
MUTATING_VERB_ALLOWLIST = {
    # aws logs start-query — initiates a log read, not a write
    ("logs", "start-query"),
    ("logs", "start-live-tail"),
    # aws ec2 copy-* that are read-ish — but copy-image IS mutating, so keep blocked
    # aws cloudwatch get-metric-statistics starts with "get-" → already allowed
}


def is_mutating(service: str, operation: str) -> bool:
    """Return True if the aws <service> <operation> is a mutating call."""
    if (service, operation) in MUTATING_VERB_ALLOWLIST:
        return False

    if operation in MUTATING_EXACT:
        return True

    # Extract the verb: everything before the first hyphen, or the whole string
    verb = operation.split("-")[0] if "-" in operation else operation
    return verb in MUTATING_VERBS


def parse_aws_command(cmd: str):
    """
    Parse 'aws [global-opts] <service> <operation> [...]' and return
    (service, operation) or (None, None) if not an AWS command.
    """
    tokens = cmd.strip().split()
    if not tokens or tokens[0] != "aws":
        return None, None

    # Skip global flags: --profile, --region, --output, --endpoint-url, --no-verify-ssl, etc.
    positional = []
    skip_next = False
    for tok in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--profile", "--region", "--output", "--endpoint-url",
                   "--endpoint", "--color", "--no-sign-request",
                   "--ca-bundle", "--cli-read-timeout", "--cli-connect-timeout",
                   "--debug", "--query", "--version"):
            skip_next = True
            continue
        if tok.startswith("--"):
            continue
        positional.append(tok)

    if len(positional) < 2:
        return None, None

    return positional[0], positional[1]


def main():
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Not JSON — pass through
        print(json.dumps({"continue": True}))
        return

    tool_name = payload.get("tool_name", "")
    if tool_name != "Bash":
        print(json.dumps({"continue": True}))
        return

    cmd = payload.get("tool_input", {}).get("command", "")
    if not cmd.strip().startswith("aws "):
        print(json.dumps({"continue": True}))
        return

    service, operation = parse_aws_command(cmd)
    if service is None:
        # Can't parse — allow through; the user will see the prompt if needed
        print(json.dumps({"continue": True}))
        return

    if is_mutating(service, operation):
        print(json.dumps({
            "continue": False,
            "stopReason": (
                f"[AWS Read-Only Guard] Blocked: 'aws {service} {operation}' is a mutating operation. "
                f"Only read-only commands (describe-*, list-*, get-*, lookup-*, filter-*, check-*) are allowed. "
                f"Ask the user for confirmation before running mutating AWS commands."
            )
        }))
    else:
        print(json.dumps({"continue": True}))


if __name__ == "__main__":
    main()
