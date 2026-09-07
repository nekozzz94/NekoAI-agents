# Claude Code Projects

A collection of Claude Code project configurations for AI-assisted engineering workflows.

## Projects

| Project | Description |
|---|---|
| [`DevOps`](./DevOps/) | AI-assisted incident response for GCP and Kubernetes environments |

---

## Usage

Each project is a self-contained Claude Code workspace with custom agents, skills, and a `CLAUDE.md` that scopes Claude's behavior to the domain.

```bash
cd <project>
claude
```

Then interact naturally or invoke a slash command:

```
/gcp-incident   # start a GCP investigation
/k8s-incident   # start a Kubernetes investigation
```

---

## Project Structure

```
<project>/
├── CLAUDE.md                  # Domain context, conventions, and agent instructions
└── .claude/
    ├── agents/                # Sub-agent definitions (role, tools, constraints)
    └── skills/                # Slash-command skill definitions
```

---

## DevOps

Incident troubleshooting for Google Cloud Platform and Kubernetes.

**Skills**

| Command | Description |
|---|---|
| `/gcp-incident` | Investigate GCP quotas, networking, IAM, Cloud SQL, GKE control plane |
| `/k8s-incident` | Deep-dive into pod/deployment health, HPA, OOMKill, CrashLoops |
| `/incident-runbook` | Generate a structured runbook: symptoms → root cause → resolution |

**Agents**

| Agent | Role |
|---|---|
| `GCP Investigator` | Queries GCP APIs, reads logs/metrics, checks quotas and IAM |
| `K8s Investigator` | Inspects cluster state via kubectl, finds failing workloads |
| `Incident Commander` | Orchestrates both agents, synthesizes findings, produces incident report |

**Prerequisites**

```bash
gcloud auth list
kubectl config current-context
gcloud config get-value project
```

---

## Adding a New Project

1. Create a directory: `mkdir <project>`
2. Add a `CLAUDE.md` with domain context and conventions
3. Add agents under `.claude/agents/<name>.md`
4. Add skills under `.claude/skills/<name>.md`
5. Run `claude` from the project directory

See the [Claude Code quickstart](https://code.claude.com/docs/en/quickstart) for reference.
