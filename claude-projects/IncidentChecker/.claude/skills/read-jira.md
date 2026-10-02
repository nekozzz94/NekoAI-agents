# /read-jira

Use this skill whenever you need to read from or write to Jira — fetching ticket details, searching with JQL, posting comments, or logging work against `jira.example.com`.

> **Security rule: Never print, echo, or expose the Jira API token.** Always fetch it inline using `$(security find-generic-password -s "JiraAPIToken" -w)` — never assign it to a variable that gets printed.

---

## Authentication

The API token is stored in macOS Keychain under the service name `JiraAPIToken`. Always fetch inline:

```bash
# Correct — token is never exposed
curl -H "Authorization: Bearer $(security find-generic-password -s 'JiraAPIToken' -w)" ...
```

---

## Fetch Ticket Details

```bash
curl -s -X GET \
  -H "Authorization: Bearer $(security find-generic-password -s 'JiraAPIToken' -w)" \
  -H "Accept: application/json" \
  "https://jira.example.com/rest/api/2/issue/<ticket-id>?fields=summary,description,status,assignee,customfield_xxx" \
  | jq '{
      key: .key,
      summary: .fields.summary,
      status: .fields.status.name,
      assignee: .fields.assignee.displayName,
      _platform: [.fields.customfield_xxx[]?.value],
      description: .fields.description
    }'
```

`customfield_xxx` is an example custom field — replace with your own custom field IDs. Custom fields are arrays of `{value: "..."}` objects. Empty field returns `[]`.

---

## Fetch Ticket Comments

```bash
curl -s -X GET \
  -H "Authorization: Bearer $(security find-generic-password -s 'JiraAPIToken' -w)" \
  -H "Accept: application/json" \
  "https://jira.example.com/rest/api/2/issue/<ticket-id>/comment?maxResults=10" \
  | jq '{
      total: .total,
      comments: [.comments[] | {
        author: .author.displayName,
        created: .created,
        body: .body
      }]
    }'
```

---

## Search Tickets with JQL

Use `--get` + `--data-urlencode` to safely URL-encode JQL without manual escaping:

```bash
curl -s -X GET \
  -H "Authorization: Bearer $(security find-generic-password -s 'JiraAPIToken' -w)" \
  -H "Accept: application/json" \
  --get \
  --data-urlencode 'jql=Sprint in openSprints() AND assignee in ("user1@example.com") ORDER BY status DESC' \
  --data-urlencode 'fields=summary,assignee,status' \
  --data-urlencode 'maxResults=100' \
  "https://jira.example.com/rest/api/2/search" \
  | jq '[.issues[] | {key: .key, summary: .fields.summary, assignee: .fields.assignee.displayName, status: .fields.status.name}]'
```

Paginate using `startAt` when `total > maxResults`.

---

## Post a Comment

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(security find-generic-password -s 'JiraAPIToken' -w)" \
  -H "Content-Type: application/json" \
  -d '{"body": "<comment text>"}' \
  "https://jira.example.com/rest/api/2/issue/<ticket-id>/comment" \
  | jq '{id: .id, author: .author.displayName, body: .body}'
```

---

## Log Work (Worklog)

`timeSpent` format: `30m`, `1h`, `2h 30m`.

```bash
curl -s -X POST \
  -H "Authorization: Bearer $(security find-generic-password -s 'JiraAPIToken' -w)" \
  -H "Content-Type: application/json" \
  -d "{\"timeSpent\": \"<time>\", \"started\": \"$(date +%Y-%m-%dT%H:%M:%S.000%z)\"}" \
  "https://jira.example.com/rest/api/2/issue/<ticket-id>/worklog" \
  | jq '{id: .id, timeSpent: .timeSpent, author: .author.displayName}'
```

---

## Quick Reference

| What | Endpoint |
|---|---|
| Ticket details | `GET /rest/api/2/issue/<ticket-id>?fields=...` |
| Comments (read) | `GET /rest/api/2/issue/<ticket-id>/comment` |
| Post a comment | `POST /rest/api/2/issue/<ticket-id>/comment` |
| Log work | `POST /rest/api/2/issue/<ticket-id>/worklog` |
| Search with JQL | `GET /rest/api/2/search?jql=...&fields=...&maxResults=...` |
| Base URL | `https://jira.example.com` |
| Auth | Bearer token from macOS Keychain (`JiraAPIToken`) |
| ITOPS Platform field | `customfield_xxx` → array of `{value: "..."}` objects |
