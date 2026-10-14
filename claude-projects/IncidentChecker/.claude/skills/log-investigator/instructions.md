# Cloud Logging Investigator Instructions

Execute the following steps when the user invokes `/log-investigator <URL>`.

## Step 1: Validate prerequisites

```bash
# Check gcloud auth
gcloud auth list

# Get current project
gcloud config get-value project
```

If no project is set, ask the user which GCP project contains the logs.

## Step 2: Expand and parse the URL

### If URL is shortened (`cloudlogging.app.goo.gl`)

```bash
# Expand the shortened URL to get the full console URL
EXPANDED_URL=$(curl -Ls -o /dev/null -w %{url_effective} "<SHORTENED_URL>")
echo "Expanded URL: $EXPANDED_URL"
```

### Extract query parameters from the full console URL

The full URL format is typically:
```
https://console.cloud.google.com/logs/query;query=<QUERY>;timeRange=<RANGE>;...?project=<PROJECT>
```

Parse:
- **Project ID**: from `?project=` query param
- **Query filter**: from `;query=` (URL-encoded, needs decoding)
- **Time range**: from `;timeRange=` or `;startTime=` and `;endTime=`
- **Resource type**: from `;cursorTimestamp=` context or within the query itself

Use this bash snippet to extract and decode:

```bash
# Extract project
PROJECT=$(echo "$EXPANDED_URL" | grep -oP 'project=\K[^&]+' | head -1)

# Extract and URL-decode the query filter
QUERY_ENCODED=$(echo "$EXPANDED_URL" | grep -oP 'query=\K[^;]+' | head -1)
QUERY=$(python3 -c "import sys, urllib.parse; print(urllib.parse.unquote_plus('$QUERY_ENCODED'))")

# Extract time range (if present)
TIME_RANGE=$(echo "$EXPANDED_URL" | grep -oP 'timeRange=\K[^;]+' | head -1)
```

If time range is not explicit, default to last 1 hour: `--freshness=1h`

## Step 3: Convert to gcloud logging read format

Cloud Logging console query uses a different syntax than `gcloud logging read`. Common conversions:

| Console Query | gcloud Format |
|---|---|
| `resource.type="k8s_container"` | Same |
| `resource.labels.pod_name="service-worker-1-*"` | Same |
| `severity>="ERROR"` | Same |
| `timestamp>="2024-10-12T10:00:00Z"` | Use `--freshness` or omit (time range handled separately) |
| `textPayload=~".*timeout.*"` | Same (supports regex) |
| `jsonPayload.message=~".*error.*"` | Same |

Remove `timestamp` filters from the query string if using `--freshness`.

## Step 4: Fetch logs

```bash
# Generate timestamped filename
LOG_FILE="/tmp/logs-$(date +%Y%m%d-%H%M%S).json"

gcloud logging read "$QUERY" \
  --project="$PROJECT" \
  --limit=<LIMIT> \
  --freshness=<FRESHNESS> \
  --format=json > "$LOG_FILE"
```

Default `--limit`: 100 (override with user's `--limit` arg)
Default `--freshness`: 1h (use extracted time range if available)

## Step 5: Analyze logs

Parse the JSON output and compute:

1. **Total entries**: `jq length "$LOG_FILE"`
2. **Severity distribution**: count by `severity` field
3. **Time range**: earliest and latest `timestamp`
4. **Resource types**: unique `resource.type` values
5. **Top error messages**: 
   - Extract `textPayload`, `jsonPayload.message`, or `protoPayload.status.message`
   - Count occurrences of each unique message
   - Show top 10 with counts
6. **Affected resources**: pod names, service names, instance IDs from `resource.labels`

Use `jq` for JSON parsing:

```bash
# Severity breakdown
echo "## Severity Distribution"
jq -r '.[] | .severity // "UNKNOWN"' "$LOG_FILE" | sort | uniq -c | sort -rn

# Time range
echo "## Time Range"
jq -r '[.[].timestamp] | min, max' "$LOG_FILE"

# Resource types
echo "## Resource Types"
jq -r '.[].resource.type' "$LOG_FILE" | sort | uniq -c | sort -rn

# Top error messages (first 100 chars of each message to avoid duplication from timestamps)
echo "## Top Error Messages"
jq -r '.[] | .textPayload // .jsonPayload.message // .protoPayload.status.message // "N/A"' "$LOG_FILE" \
  | sed 's/^\(.\{100\}\).*/\1.../' \
  | sort | uniq -c | sort -rn | head -10

# Affected pods (if K8s logs)
echo "## Affected Pods"
jq -r '.[] | select(.resource.type=="k8s_container") | .resource.labels.pod_name' "$LOG_FILE" \
  | sort | uniq -c | sort -rn | head -20
```

## Step 6: Surface findings and write report

Create the logs directory if it doesn't exist and generate a timestamped markdown report:

```bash
# Create logs directory
mkdir -p ./logs

# Generate timestamped report filename
REPORT_FILE="./logs/analyze-$(date +%Y%m%d-%H%M%S).md"
```

Write the structured summary to both stdout and the markdown file:

```bash
# Write to file and display
tee "$REPORT_FILE" <<EOF
## Log Investigation Results

**Date**: $(date '+%Y-%m-%d %H:%M:%S')
**Project**: $PROJECT
**Query**: \`$QUERY\`
**Time Range**: $START_TIME to $END_TIME
**Total Entries**: $(jq length "$LOG_FILE")

### Severity Breakdown
$(jq -r '.[] | .severity // "UNKNOWN"' "$LOG_FILE" | sort | uniq -c | sort -rn | awk '{print "- " $2 ": " $1}')

### Resource Types
$(jq -r '.[].resource.type // "unknown"' "$LOG_FILE" | sort | uniq -c | sort -rn | awk '{print "- " $2 ": " $1}')

### Time Range Details
- **Earliest**: $(jq -r '[.[].timestamp] | min' "$LOG_FILE")
- **Latest**: $(jq -r '[.[].timestamp] | max' "$LOG_FILE")

### Top API Operations
$(jq -r '.[] | .protoPayload.methodName // .jsonPayload.method // "unknown"' "$LOG_FILE" | sort | uniq -c | sort -rn | head -10 | awk '{print "- " $2 ": " $1}')

### Top Error Messages
$(jq -r '.[] | .textPayload // .jsonPayload.message // .protoPayload.status.message // "N/A"' "$LOG_FILE" | sed 's/^\(.\{100\}\).*/\1.../' | sort | uniq -c | sort -rn | head -10 | awk '{count=$1; $1=""; print "- (" count "x) " $0}')

### Affected Resources
$(jq -r '.[] | select(.resource.type=="k8s_container") | .resource.labels.pod_name // empty' "$LOG_FILE" | sort | uniq -c | sort -rn | head -20 | awk '{print "- " $2 ": " $1 " entries"}')

### Recommended Next Steps
- Review the full logs at: \`$LOG_FILE\`
- If errors found, investigate top error message: \`gcloud logging read '<filter>' --limit=10\`
- For K8s issues, check pod status: \`kubectl describe pod <pod> -n <namespace>\`
- For Cloud SQL errors, check instance: \`gcloud sql instances describe <instance>\`
- For quota issues, check quotas: \`gcloud compute project-info describe --format="yaml(quotas)"\`

---

**Raw logs**: \`$LOG_FILE\`
**Report generated**: $(date '+%Y-%m-%d %H:%M:%S')
EOF

echo ""
echo "📄 Analysis report saved to: $REPORT_FILE"
```

## Step 7: Raw mode (if `--format=raw`)

If user passes `--format=raw`, skip the analysis and just output the raw JSON:

```bash
cat "$LOG_FILE" | jq -C '.'
echo ""
echo "Raw logs saved to: $LOG_FILE"
```

## Error Handling

- If URL expansion fails (404, timeout): ask user to verify the URL is accessible and they have permissions
- If `gcloud logging read` fails with 403: user lacks `logging.logEntries.list` permission
- If `gcloud logging read` returns 0 entries: query may be too restrictive or time range too narrow
- If query parsing fails: show the extracted query and ask user to provide the filter manually

## Example Invocation

```
/log-investigator https://cloudlogging.app.goo.gl/J9DM8FAhiiFHdfDt9
/log-investigator https://cloudlogging.app.goo.gl/abc123 --limit 200
/log-investigator <FULL_CONSOLE_URL> --format=raw
```

## Integration with Incident Workflow

This skill is typically used at **Step 3-4** (Triage/Diagnose) after a Jira ticket or user report mentions a Cloud Logging URL as evidence. After log investigation:

1. Record key findings in the investigation plan
2. If errors point to a specific service, proceed with the appropriate investigator (GCP/K8s/AWS)
3. Cross-reference error timestamps with deployment history or GCP operations
4. Include log excerpts in the final runbook under **Evidence** section
5. Reference the analysis report and raw logs in the runbook:
   - Analysis: `./logs/analyze-YYYYMMDD-HHMMSS.md`
   - Raw logs: `/tmp/logs-YYYYMMDD-HHMMSS.json`
6. The markdown report can be directly included in incident documentation or attached to Jira tickets

---

**Output**: present the structured summary above, write it to `./logs/analyze-<timestamp>.md`, display both file paths, and suggest next investigator based on resource types found (e.g., if `k8s_container` dominates → K8s Investigator; if `cloud_sql_database` → GCP Investigator).

At the end of the investigation, always include:

```
📄 **Analysis report**: ./logs/analyze-YYYYMMDD-HHMMSS.md
📦 **Raw logs**: /tmp/logs-YYYYMMDD-HHMMSS.json
```
