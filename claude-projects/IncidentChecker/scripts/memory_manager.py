#!/usr/bin/env python3
"""
Memory manager for DevOps incident response.
Handles read/write/search for episodic and semantic memory.

Incident memory commands:
  python3 memory_manager.py add-incident --stdin < incident.json
  python3 memory_manager.py search-incidents --keywords "OOMKill,payment-service"
  python3 memory_manager.py get-patterns --symptom "CrashLoopBackOff"
  python3 memory_manager.py add-pattern --category ssl-cert-expiry --stdin < pattern.json

Legacy provisioning commands:
  python3 memory_manager.py find-repo --resource-type google_sql_database_instance
  python3 memory_manager.py find-repo --keywords "postgres,database"
  python3 memory_manager.py scan-root --resource-type google_redis_instance
  python3 memory_manager.py scan-root --keywords "alloydb"
  python3 memory_manager.py search-episodes --keywords "cloud sql,platform"
  python3 memory_manager.py add-episode --stdin < episode.json
  python3 memory_manager.py get-workflow --name provision_new_resource
  python3 memory_manager.py get-conventions
  python3 memory_manager.py get-resource-pattern --type google_sql_database_instance
  python3 memory_manager.py list-repos [--type gcp|helm|all]
  python3 memory_manager.py refresh-registry
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BITBUCKET_ROOT = Path("~/terraform-repos").expanduser()

MEMORY_DIR     = Path(__file__).parent.parent / ".claude" / "memory"
KNOWLEDGE_DIR  = Path(__file__).parent.parent / "knowledge"

EPISODIC_FILE      = MEMORY_DIR / "episodic" / "episodes.jsonl"
INCIDENTS_FILE     = MEMORY_DIR / "episodic" / "incidents.jsonl"
RESOURCES_FILE     = MEMORY_DIR / "semantic" / "gcp_resources.json"
CONVENTIONS_FILE   = MEMORY_DIR / "semantic" / "conventions.json"
PATTERNS_FILE      = MEMORY_DIR / "semantic" / "incident_patterns.json"
WORKFLOWS_FILE     = MEMORY_DIR / "procedural" / "workflows.json"
REGISTRY_FILE      = KNOWLEDGE_DIR / "repository-registry.json"
DISCOVER_SCRIPT    = Path(__file__).parent / "discover_repos.py"


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {"repositories": [], "resource_type_to_repo": {}}
    with open(REGISTRY_FILE) as f:
        return json.load(f)


def registry_find_by_resource_type(resource_type: str) -> dict | None:
    registry = load_registry()
    repo_id = registry.get("resource_type_to_repo", {}).get(resource_type)
    if not repo_id:
        return None
    for repo in registry["repositories"]:
        if repo["id"] == repo_id:
            return repo
    return None


def registry_find_by_keywords(keywords: str) -> list[dict]:
    """Fuzzy keyword search across id, description, type, keywords, resource_types."""
    registry = load_registry()
    kw_list = [k.strip().lower() for k in keywords.replace(" ", ",").split(",") if k.strip()]
    results = []
    for repo in registry["repositories"]:
        search_text = " ".join([
            repo.get("id", ""),
            repo.get("description", ""),
            repo.get("type", ""),
            " ".join(repo.get("keywords", [])),
            " ".join(repo.get("resource_types", [])),
        ]).lower()
        score = sum(1 for kw in kw_list if kw in search_text)
        if score > 0:
            results.append({"score": score, "repo": repo})
    results.sort(key=lambda x: x["score"], reverse=True)
    return [r["repo"] for r in results]


# ---------------------------------------------------------------------------
# Live scan of BITBUCKET_ROOT (fallback when registry misses)
# ---------------------------------------------------------------------------

def scan_root_by_resource_type(resource_type: str, root: Path = BITBUCKET_ROOT) -> list[dict]:
    """
    Grep all .tf files under root for the given resource type.
    Returns list of {repo_id, local_path, files, match_count}.
    """
    if not root.exists():
        return []
    pattern = f'resource\\s+"{re.escape(resource_type)}"'
    try:
        result = subprocess.run(
            ["grep", "-rl", "--include=*.tf", "-E", pattern, str(root)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    hits: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        p = Path(line)
        # repo is the first-level subdirectory under root
        try:
            relative = p.relative_to(root)
            repo_name = relative.parts[0]
        except (ValueError, IndexError):
            continue
        hits.setdefault(repo_name, []).append(str(p.relative_to(root / repo_name)))

    return [
        {
            "repo_id": repo_name,
            "local_path": str(root / repo_name),
            "matching_files": files,
            "match_count": len(files),
        }
        for repo_name, files in sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)
    ]


def scan_root_by_keywords(keywords: str, root: Path = BITBUCKET_ROOT) -> list[dict]:
    """
    Grep all .tf files for any of the keywords.
    Returns hits grouped by repo, sorted by match count.
    """
    if not root.exists():
        return []
    kw_list = [k.strip() for k in keywords.replace(" ", ",").split(",") if k.strip()]
    if not kw_list:
        return []

    pattern = "|".join(re.escape(kw) for kw in kw_list)
    try:
        result = subprocess.run(
            ["grep", "-rl", "--include=*.tf", "-iE", pattern, str(root)],
            capture_output=True, text=True, timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    hits: dict[str, list[str]] = {}
    for line in result.stdout.splitlines():
        p = Path(line)
        try:
            relative = p.relative_to(root)
            repo_name = relative.parts[0]
        except (ValueError, IndexError):
            continue
        hits.setdefault(repo_name, []).append(str(p))

    return [
        {
            "repo_id": repo_name,
            "local_path": str(root / repo_name),
            "matching_files": files[:10],
            "match_count": len(files),
        }
        for repo_name, files in sorted(hits.items(), key=lambda x: len(x[1]), reverse=True)
    ]


# ---------------------------------------------------------------------------
# Unified find-repo: registry first, live scan fallback
# ---------------------------------------------------------------------------

def find_repo(resource_type: str | None = None,
              keywords: str | None = None,
              root: Path = BITBUCKET_ROOT) -> dict:
    """
    Find the best repo. Strategy:
      1. Registry exact match by resource_type
      2. Registry keyword search
      3. Live grep across root (fallback)
    Returns {method, results, top_repo_id, top_local_path}
    """
    if resource_type:
        # 1. Registry exact match
        repo = registry_find_by_resource_type(resource_type)
        if repo:
            return {
                "method": "registry:resource_type",
                "resource_type": resource_type,
                "results": [repo],
                "top_repo_id": repo["id"],
                "top_local_path": repo["local_path"],
            }
        # 2. Live scan fallback
        hits = scan_root_by_resource_type(resource_type, root)
        if hits:
            top = hits[0]
            return {
                "method": "live_scan:resource_type",
                "resource_type": resource_type,
                "results": hits,
                "top_repo_id": top["repo_id"],
                "top_local_path": top["local_path"],
                "note": "Not in registry — found by live grep. Run refresh-registry to update.",
            }
        return {
            "method": "none",
            "resource_type": resource_type,
            "results": [],
            "error": f"No repository found for resource type '{resource_type}' in registry or live scan.",
        }

    if keywords:
        # 1. Registry keyword search
        repos = registry_find_by_keywords(keywords)
        if repos:
            return {
                "method": "registry:keywords",
                "keywords": keywords,
                "results": repos[:5],
                "top_repo_id": repos[0]["id"],
                "top_local_path": repos[0]["local_path"],
            }
        # 2. Live scan fallback
        hits = scan_root_by_keywords(keywords, root)
        if hits:
            top = hits[0]
            return {
                "method": "live_scan:keywords",
                "keywords": keywords,
                "results": hits[:5],
                "top_repo_id": top["repo_id"],
                "top_local_path": top["local_path"],
                "note": "Not in registry — found by live grep. Run refresh-registry to update.",
            }
        return {
            "method": "none",
            "keywords": keywords,
            "results": [],
            "error": f"No repository found for keywords '{keywords}'.",
        }

    return {"error": "Provide --resource-type or --keywords"}


# ---------------------------------------------------------------------------
# Episodic memory
# ---------------------------------------------------------------------------

def search_episodes(keywords: str, limit: int = 5) -> list[dict]:
    if not EPISODIC_FILE.exists() or EPISODIC_FILE.stat().st_size == 0:
        return []
    kw_list = [k.strip().lower() for k in keywords.split(",")]
    results = []
    with open(EPISODIC_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = json.dumps(episode).lower()
            score = sum(1 for kw in kw_list if kw in text)
            if score > 0:
                results.append({"score": score, "episode": episode})
    results.sort(key=lambda x: x["score"], reverse=True)
    return [r["episode"] for r in results[:limit]]


def add_episode(episode: dict) -> None:
    episode["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with open(EPISODIC_FILE, "a") as f:
        f.write(json.dumps(episode) + "\n")


# ---------------------------------------------------------------------------
# Incident episodic memory
# ---------------------------------------------------------------------------

def add_incident(incident: dict) -> None:
    """Append a resolved incident record to incidents.jsonl."""
    INCIDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    incident["recorded_at"] = datetime.now(timezone.utc).isoformat()
    with open(INCIDENTS_FILE, "a") as f:
        f.write(json.dumps(incident) + "\n")


def search_incidents(keywords: str, limit: int = 5) -> list[dict]:
    """Full-text keyword search across incidents.jsonl."""
    if not INCIDENTS_FILE.exists() or INCIDENTS_FILE.stat().st_size == 0:
        return []
    kw_list = [k.strip().lower() for k in keywords.split(",")]
    results = []
    with open(INCIDENTS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                incident = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = json.dumps(incident).lower()
            score = sum(1 for kw in kw_list if kw in text)
            if score > 0:
                results.append({"score": score, "incident": incident})
    results.sort(key=lambda x: x["score"], reverse=True)
    return [r["incident"] for r in results[:limit]]


# ---------------------------------------------------------------------------
# Incident semantic memory — patterns
# ---------------------------------------------------------------------------

def load_patterns() -> dict:
    if not PATTERNS_FILE.exists():
        return {"patterns": {}}
    with open(PATTERNS_FILE) as f:
        return json.load(f)


def get_patterns(symptom: str) -> list[dict]:
    """Return patterns whose symptom list contains any word from symptom string."""
    data = load_patterns()
    kw_list = [k.strip().lower() for k in symptom.replace(",", " ").split() if k.strip()]
    matches = []
    for category, pattern in data.get("patterns", {}).items():
        symptom_text = " ".join(pattern.get("symptoms", [])).lower()
        title_text = pattern.get("title", "").lower()
        score = sum(1 for kw in kw_list if kw in symptom_text or kw in title_text)
        if score > 0:
            matches.append({"score": score, "category": category, "pattern": pattern})
    matches.sort(key=lambda x: x["score"], reverse=True)
    return [{"category": m["category"], **m["pattern"]} for m in matches]


def add_pattern(category: str, patch: dict) -> None:
    """
    Upsert a pattern. If category exists, merge patch fields (seen_count is incremented).
    If new, insert patch as the full pattern.
    """
    data = load_patterns()
    patterns = data.setdefault("patterns", {})
    if category in patterns:
        existing = patterns[category]
        for key, val in patch.items():
            if key == "seen_count":
                existing["seen_count"] = existing.get("seen_count", 0) + 1
            elif key == "symptoms":
                combined = existing.get("symptoms", [])
                for s in val:
                    if s not in combined:
                        combined.append(s)
                existing["symptoms"] = combined
            else:
                existing[key] = val
        existing["last_seen"] = datetime.now(timezone.utc).date().isoformat()
    else:
        patch.setdefault("seen_count", 1)
        patch["last_seen"] = datetime.now(timezone.utc).date().isoformat()
        patterns[category] = patch
    data["_meta"]["last_updated"] = datetime.now(timezone.utc).date().isoformat()
    with open(PATTERNS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Semantic / procedural memory
# ---------------------------------------------------------------------------

def get_workflow(name: str) -> dict | None:
    with open(WORKFLOWS_FILE) as f:
        workflows = json.load(f)
    return workflows.get("workflows", {}).get(name)


def get_conventions() -> dict:
    with open(CONVENTIONS_FILE) as f:
        return json.load(f)


def get_resource_pattern(resource_type: str) -> dict | None:
    with open(RESOURCES_FILE) as f:
        resources = json.load(f)
    return resources.get("resource_patterns", {}).get(resource_type)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Memory manager for Terraform Provisioner")
    parser.add_argument("--root", default=str(BITBUCKET_ROOT),
                        help="Root directory for live repo scan (default: ~/terraform-repos)")
    sub = parser.add_subparsers(dest="command", required=True)

    # find-repo: registry + live fallback
    p = sub.add_parser("find-repo",
                       help="Find repo by resource type or keywords (registry + live scan fallback)")
    p.add_argument("--resource-type", help="Terraform resource type")
    p.add_argument("--keywords", help="Comma-separated keywords")

    # scan-root: live grep only (bypass registry)
    p = sub.add_parser("scan-root",
                       help="Live grep across root dir (bypasses registry — useful for exploration)")
    p.add_argument("--resource-type", help="Terraform resource type to grep for")
    p.add_argument("--keywords", help="Comma-separated keywords to grep for")

    # incident episodic memory
    p = sub.add_parser("add-incident", help="Record a resolved incident to incidents.jsonl")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--file")

    p = sub.add_parser("search-incidents", help="Search past incidents by keyword")
    p.add_argument("--keywords", required=True, help="Comma-separated keywords (symptom, service, root cause, jira id)")
    p.add_argument("--limit", type=int, default=5)

    # incident semantic memory — patterns
    p = sub.add_parser("get-patterns", help="Get known failure patterns matching a symptom")
    p.add_argument("--symptom", required=True, help="Symptom text (e.g. 'CrashLoopBackOff OOMKill')")

    p = sub.add_parser("add-pattern", help="Add or update a failure pattern after an incident")
    p.add_argument("--category", required=True, help="Pattern category key (e.g. ssl-cert-expiry)")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--file")

    # legacy provisioning episodic
    p = sub.add_parser("search-episodes", help="Search provisioning episodic memory")
    p.add_argument("--keywords", required=True)
    p.add_argument("--limit", type=int, default=5)

    p = sub.add_parser("add-episode", help="Record a completed provisioning task")
    p.add_argument("--stdin", action="store_true")
    p.add_argument("--file")

    # procedural
    p = sub.add_parser("get-workflow", help="Retrieve a procedural workflow")
    p.add_argument("--name", required=True)

    # semantic
    sub.add_parser("get-conventions", help="Print Terraform conventions")

    p = sub.add_parser("get-resource-pattern", help="Get Terraform pattern for a resource type")
    p.add_argument("--type", required=True, dest="resource_type")

    # list
    p = sub.add_parser("list-repos", help="List all known repositories")
    p.add_argument("--type", default="all",
                   choices=["all", "gcp", "helm", "iam", "vpc", "gke", "monitoring", "infra"],
                   help="Filter by repo type")
    p.add_argument("--sort", default="name", choices=["name", "resources"],
                   help="Sort order")

    # refresh registry
    sub.add_parser("refresh-registry",
                   help="Re-scan ~/terraform-repos and regenerate repository-registry.json")

    args = parser.parse_args()
    root = Path(args.root).expanduser()

    if args.command == "find-repo":
        result = find_repo(
            resource_type=getattr(args, "resource_type", None),
            keywords=getattr(args, "keywords", None),
            root=root,
        )
        print(json.dumps(result, indent=2))
        if result.get("error"):
            sys.exit(1)

    elif args.command == "scan-root":
        if args.resource_type:
            hits = scan_root_by_resource_type(args.resource_type, root)
        elif args.keywords:
            hits = scan_root_by_keywords(args.keywords, root)
        else:
            print("Provide --resource-type or --keywords", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(hits, indent=2))

    elif args.command == "add-incident":
        if args.stdin:
            incident = json.load(sys.stdin)
        elif args.file:
            with open(args.file) as f:
                incident = json.load(f)
        else:
            print("Provide --stdin or --file", file=sys.stderr)
            sys.exit(1)
        add_incident(incident)
        print(f"Incident recorded at {INCIDENTS_FILE}")

    elif args.command == "search-incidents":
        results = search_incidents(args.keywords, args.limit)
        print(json.dumps(results, indent=2))

    elif args.command == "get-patterns":
        results = get_patterns(args.symptom)
        if results:
            print(json.dumps(results, indent=2))
        else:
            print(json.dumps({"message": f"No known patterns for symptom: {args.symptom}"}))

    elif args.command == "add-pattern":
        if args.stdin:
            patch = json.load(sys.stdin)
        elif args.file:
            with open(args.file) as f:
                patch = json.load(f)
        else:
            print("Provide --stdin or --file", file=sys.stderr)
            sys.exit(1)
        add_pattern(args.category, patch)
        print(f"Pattern '{args.category}' updated in {PATTERNS_FILE}")

    elif args.command == "search-episodes":
        results = search_episodes(args.keywords, args.limit)
        print(json.dumps(results, indent=2))

    elif args.command == "add-episode":
        if args.stdin:
            episode = json.load(sys.stdin)
        elif args.file:
            with open(args.file) as f:
                episode = json.load(f)
        else:
            print("Provide --stdin or --file", file=sys.stderr)
            sys.exit(1)
        add_episode(episode)
        print(f"Episode recorded at {EPISODIC_FILE}")

    elif args.command == "get-workflow":
        wf = get_workflow(args.name)
        if wf:
            print(json.dumps(wf, indent=2))
        else:
            available = list(json.load(open(WORKFLOWS_FILE)).get("workflows", {}).keys())
            print(f"Workflow '{args.name}' not found. Available: {available}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "get-conventions":
        print(json.dumps(get_conventions(), indent=2))

    elif args.command == "get-resource-pattern":
        pattern = get_resource_pattern(args.resource_type)
        if pattern:
            print(json.dumps(pattern, indent=2))
        else:
            print(f"No pattern for: {args.resource_type}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list-repos":
        # Type aliases: "gcp" means all GCP-provider repos (excludes helm/vcs/unknown)
        GCP_TYPES = {"team", "gke", "vpc", "iam", "monitoring", "infra",
                     "platform", "cmek", "resourcemanager", "idp"}
        registry = load_registry()
        repos = registry.get("repositories", [])
        if args.type == "all":
            pass
        elif args.type == "gcp":
            repos = [r for r in repos if r.get("type") in GCP_TYPES]
        else:
            repos = [r for r in repos if r.get("type") == args.type]
        if args.sort == "resources":
            repos = sorted(repos, key=lambda r: r.get("resource_count", 0), reverse=True)
        else:
            repos = sorted(repos, key=lambda r: r["id"])
        for repo in repos:
            envs = ",".join(repo.get("environments", [])[:4])
            rtype = repo.get("type", "?")
            print(f"{repo['id']:<48} {repo.get('resource_count',0):>5} resources  type:{rtype:<14} [{envs}]")
            print(f"  path: {repo['local_path']}")
            if repo.get("description"):
                print(f"  desc: {repo['description'][:120]}")

    elif args.command == "refresh-registry":
        import subprocess as sp
        print(f"Re-scanning {root} ...", file=sys.stderr)
        result = sp.run(
            ["python3", str(DISCOVER_SCRIPT), "--root", str(root)],
            capture_output=False,
        )
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
