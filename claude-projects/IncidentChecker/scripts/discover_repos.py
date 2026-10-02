#!/usr/bin/env python3
"""
Discover and fingerprint all Terraform repositories under a root directory.

Scans every subdirectory, identifies those containing .tf files, extracts:
  - All resource types defined
  - Environments detected (dev/staging/prod/uat etc.)
  - README/CLAUDE.md description
  - Stack structure
  - atlantis.yaml project names

Then writes (or merges) the results into knowledge/repository-registry.json.

Usage:
  python3 discover_repos.py --root ~/terraform-repos
  python3 discover_repos.py --root ~/terraform-repos --dry-run
  python3 discover_repos.py --root ~/terraform-repos --repo platform-terraform
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
REGISTRY_FILE = KNOWLEDGE_DIR / "repository-registry.json"

# Repo name patterns → classification
# Customize these patterns to match your organization's repository naming conventions
REPO_TYPE_PATTERNS = [
    (r"gke-.*terraform",       "gke",       "GKE cluster provisioning"),
    (r"vpc-.*terraform",       "vpc",       "VPC networking (networks, subnets, firewall, DNS, NAT)"),
    (r"iam-.*terraform",       "iam",       "IAM bindings and service accounts"),
    (r"monitoring-.*terraform", "monitoring", "Cloud Monitoring, alerting, dashboards"),
    (r"infra-.*terraform",     "infra",     "General shared infrastructure"),
    (r"(\w+)-gcp-terraform",   "app",      "App-specific GCP resources"),
    (r"platform-terraform",    "platform",  "Platform resources"),
    (r"groups-terraform",      "iam",       "IAM groups"),
    (r"users-terraform", "iam", "User and group management"),
    (r".*-helm$",              "helm",      "Helm/Kubernetes deployment"),
]

KNOWN_ENVS = ["dev", "test", "uat", "acc", "staging", "prod", "nonprod"]


def classify_repo(repo_name: str) -> tuple[str, str]:
    """Return (type, description) for a repo name."""
    for pattern, rtype, desc in REPO_TYPE_PATTERNS:
        if re.search(pattern, repo_name):
            # Expand app description with the app name
            if rtype == "app":
                m = re.match(r"(\w+)-gcp-terraform", repo_name)
                app = m.group(1) if m else repo_name
                return rtype, f"GCP resources for {app} app"
            return rtype, desc
    return "unknown", repo_name


def extract_resource_types(repo_path: Path) -> dict[str, list[str]]:
    """Scan .tf files and return {resource_type: [resource_name, ...]}."""
    resources: dict[str, list[str]] = defaultdict(list)
    for tf_file in repo_path.rglob("*.tf"):
        try:
            text = tf_file.read_text(errors="ignore")
        except (PermissionError, OSError):
            continue
        for m in re.finditer(r'\bresource\s+"([^"]+)"\s+"([^"]+)"', text):
            rtype, rname = m.group(1), m.group(2)
            if rname not in resources[rtype]:
                resources[rtype].append(rname)
    return dict(resources)


def detect_environments(repo_path: Path) -> list[str]:
    """Detect environment names from directory names, tfvars files, and atlantis.yaml."""
    envs = set()
    text_blob = ""

    # Check directory names
    for item in repo_path.rglob("*"):
        name = item.name.lower()
        for env in KNOWN_ENVS:
            if env in name.split("-") or name == env:
                envs.add(env)

    # Check atlantis.yaml project names
    for atlantis_file in repo_path.glob("atlantis*.yaml"):
        try:
            text_blob += atlantis_file.read_text(errors="ignore")
        except (PermissionError, OSError):
            pass

    for env in KNOWN_ENVS:
        if re.search(rf"\b{env}\b", text_blob, re.IGNORECASE):
            envs.add(env)

    return sorted(envs) if envs else ["unknown"]


def read_description(repo_path: Path) -> str:
    """Extract a one-line description from README.md or CLAUDE.md."""
    for fname in ["CLAUDE.md", "README.md", "readme.md"]:
        fpath = repo_path / fname
        if fpath.exists():
            try:
                text = fpath.read_text(errors="ignore")
                # First non-empty line that isn't a heading marker itself
                for line in text.splitlines():
                    line = line.strip().lstrip("#").strip()
                    if line and len(line) > 5:
                        return line[:200]
            except (PermissionError, OSError):
                pass
    return ""


def read_atlantis_projects(repo_path: Path) -> list[str]:
    """Extract atlantis project names."""
    projects = []
    for atlantis_file in repo_path.glob("atlantis*.yaml"):
        try:
            text = atlantis_file.read_text(errors="ignore")
            for m in re.finditer(r"^\s*name:\s*(.+)$", text, re.MULTILINE):
                name = m.group(1).strip()
                if name not in projects:
                    projects.append(name)
        except (PermissionError, OSError):
            pass
    return projects


def fingerprint_repo(repo_path: Path, workspace: str = "myorg") -> dict:
    """Build a full registry entry for one repository."""
    repo_name = repo_path.name
    rtype, desc = classify_repo(repo_name)
    readme_desc = read_description(repo_path)
    resource_map = extract_resource_types(repo_path)
    envs = detect_environments(repo_path)
    atlantis_projects = read_atlantis_projects(repo_path)

    # Derive keywords from resource types and repo name parts
    resource_types = sorted(resource_map.keys())
    keywords = []
    for part in re.split(r"[-_]", repo_name):
        if part not in ("gcp", "terraform", "aws", "helm", ""):
            keywords.append(part)
    for rt in resource_types:
        # e.g. google_sql_database_instance → sql, database
        parts = rt.replace("google_", "").split("_")
        keywords.extend(parts[:3])

    # Deduplicate keywords
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen and len(kw) > 2:
            seen.add(kw)
            unique_keywords.append(kw)

    return {
        "id": repo_name,
        "display_name": repo_name.replace("-", " ").replace("_", " ").title(),
        "local_path": f"~/terraform-repos/{repo_name}",
        "bitbucket_slug": repo_name,
        "description": readme_desc or desc,
        "type": rtype,
        "team_owner": rtype if rtype != "team" else re.sub(r"-gcp-terraform$", "", repo_name),
        "environments": envs,
        "atlantis_projects": atlantis_projects,
        "resource_types": resource_types,
        "resource_instances": {k: v[:5] for k, v in resource_map.items()},  # sample max 5 names
        "resource_count": sum(len(v) for v in resource_map.values()),
        "keywords": unique_keywords[:30],
    }


def is_terraform_repo(path: Path) -> bool:
    """Return True if the directory contains any .tf files."""
    return any(path.rglob("*.tf"))


def is_helm_repo(path: Path) -> bool:
    """Return True if the directory looks like a helm/k8s repo."""
    return (
        any(path.glob("**/Chart.yaml"))
        or any(path.glob("**/values.yaml"))
        or path.name.endswith("-helm")
    )


def build_resource_type_to_repo(repos: list[dict]) -> dict[str, str]:
    """Build a flat resource_type → repo_id map, preferring more specific repos."""
    mapping: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for repo in repos:
        score = repo.get("resource_count", 0)
        for rt in repo.get("resource_types", []):
            mapping[rt].append((score, repo["id"]))

    # For each resource type, pick the repo with the highest resource count
    return {rt: sorted(entries, reverse=True)[0][1] for rt, entries in mapping.items()}


def load_existing_registry() -> dict:
    if REGISTRY_FILE.exists():
        try:
            return json.loads(REGISTRY_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description="Discover and fingerprint Terraform repos")
    parser.add_argument("--root", default="~/terraform-repos",
                        help="Root directory containing all repositories")
    parser.add_argument("--repo", help="Scan only this one repo (by name)")
    parser.add_argument("--dry-run", action="store_true", help="Print result, don't write")
    parser.add_argument("--include-helm", action="store_true", help="Also fingerprint Helm repos")
    parser.add_argument("--workspace", default="myorg")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    if not root.exists():
        print(f"Root directory not found: {root}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {root} ...", file=sys.stderr)

    # Determine which repos to scan
    if args.repo:
        candidates = [root / args.repo]
    else:
        candidates = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]

    repos = []
    skipped = []
    for candidate in sorted(candidates):
        name = candidate.name
        if not (is_terraform_repo(candidate) or (args.include_helm and is_helm_repo(candidate))):
            skipped.append(name)
            continue
        print(f"  fingerprinting {name} ...", file=sys.stderr)
        try:
            entry = fingerprint_repo(candidate, workspace=args.workspace)
            repos.append(entry)
        except Exception as e:
            print(f"  [WARN] {name}: {e}", file=sys.stderr)

    print(f"\nFound {len(repos)} repos ({len(skipped)} skipped — no .tf files)", file=sys.stderr)

    # Build resource type → repo map
    resource_type_to_repo = build_resource_type_to_repo(repos)

    registry = {
        "_meta": {
            "description": "Auto-generated repository registry. Re-run discover_repos.py to refresh.",
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "root_dir": str(root),
            "repo_count": len(repos),
        },
        "workspace": args.workspace,
        "bitbucket_base_url": f"https://bitbucket.org/{args.workspace}",
        "local_base_path": str(root),
        "repositories": repos,
        "resource_type_to_repo": resource_type_to_repo,
    }

    if args.dry_run:
        print(json.dumps(registry, indent=2))
    else:
        REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
        REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
        print(f"\nRegistry written to {REGISTRY_FILE}", file=sys.stderr)
        print(f"  {len(repos)} repositories indexed", file=sys.stderr)
        print(f"  {len(resource_type_to_repo)} resource type mappings", file=sys.stderr)

    # Print summary table to stdout
    print(f"\n{'REPO':<45} {'TYPE':<18} {'RESOURCES':>9} {'ENVS'}", file=sys.stderr)
    print("-" * 90, file=sys.stderr)
    for repo in sorted(repos, key=lambda r: r["resource_count"], reverse=True):
        envs = ",".join(repo["environments"][:4])
        print(f"  {repo['id']:<43} {repo['type']:<18} {repo['resource_count']:>9} {envs}", file=sys.stderr)


if __name__ == "__main__":
    main()
