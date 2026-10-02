# Claude Code Projects

A collection of Claude Code project configurations for AI-assisted engineering workflows.

## Projects

| Project | Description |
|---|---|
| [`IncidentChecker`](./IncidentChecker/) | AI-assisted incident response for GCP, AWS, and Kubernetes environments with memory system and Terraform inspection |

---

## Usage

Each project is a self-contained Claude Code workspace with custom agents, skills, and a `CLAUDE.md` that scopes Claude's behavior to the domain.

```bash
cd <project>
claude
```

Then interact naturally or invoke a slash command:

```
/gcp-incident         # start a GCP investigation
/k8s-incident         # start a Kubernetes investigation
/read-jira            # fetch Jira ticket details
/incident-runbook     # generate structured incident runbook
/bitbucket-pr-review  # review Terraform code in PRs
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

## IncidentChecker

AI-assisted incident response for multi-cloud environments (GCP, AWS, Kubernetes).

**Key Features**
- Multi-cloud investigation with specialized agents
- Long-term memory system that learns from past incidents
- Terraform code inspection and security scanning
- Jira/Bitbucket integration for ticket input and PR reviews
- Automatic runbook generation

See [`IncidentChecker/README.md`](./IncidentChecker/README.md) for full documentation, setup guide, and available skills/agents.

---

## Adding a New Project

1. Create a directory: `mkdir <project>`
2. Add a `CLAUDE.md` with domain context and conventions
3. Add agents under `.claude/agents/<name>.md`
4. Add skills under `.claude/skills/<name>.md`
5. Run `claude` from the project directory

See the [Claude Code quickstart](https://code.claude.com/docs/en/quickstart) for reference.
