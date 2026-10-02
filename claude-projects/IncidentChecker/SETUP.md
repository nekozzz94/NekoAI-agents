# Setup Guide for IncidentChecker

This guide will help you configure IncidentChecker for your organization.

## Step 1: Replace Placeholder Values

Search and replace these placeholders throughout the project:

### Domain and URLs
- `jira.example.com` → Your Jira instance URL
- `user@example.com` → Your organization email

### Paths
- `~/terraform-repos/` → Your actual infrastructure repository path
- Update `.claude/settings.json` `additionalDirectories` to include your repo paths

### Jira Custom Fields
Find all instances of `customfield_xxx` and replace with your actual custom field IDs:
- `.claude/skills/read-jira.md`
- `.claude/agents/requirements-reader.md`

## Step 2: Configure Repository Discovery

Edit `scripts/discover_repos.py`:

1. Update `REPO_TYPE_PATTERNS` to match your repository naming conventions:
   ```python
   REPO_TYPE_PATTERNS = [
       (r"your-pattern-.*terraform", "type", "Description"),
       # Add your organization's patterns
   ]
   ```

2. Update `KNOWN_ENVS` if you use different environment names:
   ```python
   KNOWN_ENVS = ["dev", "staging", "prod", "uat", ...]
   ```

3. Update default workspace in `main()`:
   ```python
   parser.add_argument("--workspace", default="your-org-name")
   ```

## Step 3: Configure Memory Patterns

Edit `.claude/memory/semantic/incident_patterns.json`:

1. Review existing patterns and customize `first_checks` commands for your infrastructure
2. Add your organization's specific failure patterns
3. Update platform-specific categories as needed

## Step 4: Set Up Credentials

Store credentials securely in macOS Keychain:

```bash
# Jira API token
security add-generic-password -U -s JiraAPIToken -a your-email@example.com -w

# Bitbucket API token (if using Bitbucket PR review)
security add-generic-password -U -s BitbucketAPIToken -a your-email@example.com -w
```

## Step 5: Configure Environment Variables

Add to your shell profile (`.zshrc` or `.bashrc`):

```bash
# GCP
export GCP_PROJECT=your-gcp-project-id

# Kubernetes
export K8S_CONTEXT=your-kubectl-context
export K8S_NAMESPACE=default

# AWS
export AWS_REGION=us-east-1
export AWS_PROFILE=your-profile

# Jira
export JIRA_BASE_URL=https://your-jira.example.com
export JIRA_EMAIL=your-email@example.com

# Bitbucket (if using)
export BITBUCKET_USERNAME=your-email@example.com
```

## Step 6: Customize Agents

Review and customize agent definitions in `.claude/agents/`:

- **gcp-investigator.md**: Update GCP-specific commands for your project structure
- **aws-investigator.md**: Adjust AWS CLI commands for your account structure
- **terraform-inspector.md**: Update repository paths and patterns

## Step 7: Initialize Memory

The memory system starts empty. After your first few incidents:

1. **Episodic memory** (`.claude/memory/episodic/incidents.jsonl`):
   - Automatically populated by the Stop hook when runbooks contain "Root Cause"
   - Or manually add with: `python3 scripts/memory_manager.py add-incident --stdin < incident.json`

2. **Semantic memory** (`.claude/memory/semantic/incident_patterns.json`):
   - Manually update after each incident with: `python3 scripts/memory_manager.py add-pattern --category <category> --stdin < pattern.json`

## Step 8: Test the Setup

1. **Test Jira integration**:
   ```
   /read-jira TEST-123
   ```

2. **Test GCP investigation**:
   ```
   /gcp-incident
   ```

3. **Test Kubernetes investigation**:
   ```
   /k8s-incident
   ```

4. **Test repository discovery**:
   ```bash
   python3 scripts/discover_repos.py --root ~/terraform-repos --dry-run
   ```

## Step 9: Run Repository Discovery

Generate the repository registry:

```bash
python3 scripts/discover_repos.py --root ~/terraform-repos
```

This creates `knowledge/repository-registry.json` with all your Terraform repositories indexed.

## Step 10: Customize Incident Patterns

Based on your infrastructure, add common failure patterns to `.claude/memory/semantic/incident_patterns.json`:

- Add platform-specific categories (`platform: "gcp" | "k8s" | "aws" | "terraform"`)
- Document your organization's specific failure modes
- Include first-check commands that work in your environment
- Update prevention steps to match your monitoring/alerting setup

## Optional: External Hook Scripts

If the hooks reference external scripts (e.g., `$HOME/NekoAgents/DevOps/scripts/`), either:

1. **Remove the hooks** from `.claude/settings.json` if not needed
2. **Copy the scripts** to the project and update paths in `settings.json`
3. **Implement your own versions** of:
   - `aws_readonly_guard.py` - Blocks mutating AWS commands
   - `bash_logger.py` - Logs all bash commands
   - `session_memory_hook.py` - Auto-records completed runbooks

## Verification Checklist

- [ ] All `example.com` references replaced
- [ ] Repository paths updated throughout
- [ ] Jira custom field IDs updated
- [ ] Repository discovery patterns customized
- [ ] Credentials stored in Keychain
- [ ] Environment variables configured
- [ ] Repository registry generated
- [ ] First test incident completed
- [ ] Memory patterns updated with organization-specific knowledge

## Troubleshooting

### Jira authentication fails
- Verify token in Keychain: `security find-generic-password -s JiraAPIToken -w`
- Check token permissions in Jira
- Ensure JIRA_BASE_URL doesn't have trailing slash

### Repository discovery finds no repos
- Check `--root` path exists and contains `.tf` files
- Verify repository naming patterns in `REPO_TYPE_PATTERNS`
- Run with `--dry-run` first to see what would be found

### Memory search returns no results
- Ensure `incidents.jsonl` has at least one entry
- Check keywords match content in incident records
- Verify file permissions on memory directories

## Next Steps

1. Document your first incident using `/incident-runbook`
2. Update semantic patterns based on lessons learned
3. Customize agent prompts for your infrastructure specifics
4. Add organization-specific failure patterns
5. Train your team on using the incident workflow
