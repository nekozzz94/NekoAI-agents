# log-investigator

Investigate Google Cloud Platform logs from a Cloud Logging console URL or direct query.

## What it does

1. Accepts a Cloud Logging URL (shortened `cloudlogging.app.goo.gl` or full console URL)
2. Expands shortened URLs and extracts the query, time range, and filters
3. Converts console parameters to `gcloud logging read` format
4. Fetches and analyzes the logs for errors, patterns, and anomalies
5. Surfaces key findings with timestamps, severity distribution, and common error messages

## When to use

- User provides a Cloud Logging URL and asks to investigate
- Need to analyze logs from a shared GCP logging link
- Debugging issues where logs are already filtered in the console
- Quick triage of error spikes or anomalies visible in Cloud Logging

## Input

- `url`: Cloud Logging console URL (shortened or full)
- `--limit`: Maximum number of log entries to fetch (default: 100)
- `--format`: Output format - `summary` (default) or `raw`

## Prerequisites

- `gcloud` CLI authenticated (`gcloud auth list`)
- Access to the GCP project containing the logs
- `curl` for URL expansion

## Output

- Time range analyzed
- Total entries fetched
- Severity breakdown (ERROR, WARNING, INFO, etc.)
- Top error messages with occurrence counts
- Resource types affected
- Recommended next steps
- **Markdown report**: `./logs/analyze-<timestamp>.md`
- **Raw JSON logs**: `/tmp/logs-<timestamp>.json`

## Example

```bash
/log-investigator https://cloudlogging.app.goo.gl/J9DM8FAhiiFHdfDt9
/log-investigator https://console.cloud.google.com/logs/query --limit 200
```
