# /bitbucket-pr-review

Authenticate with Bitbucket using the current browser session and trigger a focused Terraform code review for a Pull Request URL.

## When to use

Invoke when you want to:
- Review Terraform changes in a Bitbucket PR before merge
- Validate a PR against a Jira ticket or a Technical Design document
- Get a structured BLOCKER / WARNING / INFO finding report without creating new credentials

## Instructions

When this skill is invoked:

---

### Step 1 — Collect inputs

Ask the user for the following if not already provided:

1. **Bitbucket PR URL** (required)
   - `https://bitbucket.org/{workspace}/{repo}/pull-requests/{id}`

2. **Design context** (optional but recommended — pick one):
   - **Jira ticket key** (e.g. `INFRA-1234`)
   - **Technical Design file** — local path or URL to a markdown file

3. **Post findings as PR comments?** (yes/no) — requires confirmation before any write.

---

### Step 2 — Resolve Bitbucket authentication

Try each method in order. Stop at the first one that succeeds.

#### Method A — Auto-extract from browser (preferred)

Write the extraction helper, then run it:

```bash
cat > /tmp/bb_extract_cookie.py << 'PYEOF'
#!/usr/bin/env python3
"""
Extract Bitbucket cloud.session.token from Chrome, Firefox, or Safari on macOS.
Prints the raw token value on stdout; prints nothing if not found.
"""
import os, sys, sqlite3, shutil, subprocess
from pathlib import Path

TOKEN_NAME = "cloud.session.token"
HOME = Path.home()


def from_chrome():
    db = HOME / "Library/Application Support/Google/Chrome/Default/Cookies"
    if not db.exists():
        return None
    tmp = Path("/tmp/bb_chrome_cookies_tmp")
    shutil.copy2(db, tmp)
    try:
        # Decrypt Chrome v10 cookies (macOS AES-128-CBC + PBKDF2 via Keychain)
        key_pw = subprocess.check_output(
            ["security", "find-generic-password", "-wa", "Chrome Safe Storage"],
            stderr=subprocess.DEVNULL,
        ).strip()

        import hashlib, base64
        key = hashlib.pbkdf2_hmac("sha1", key_pw, b"saltysalt", 1003, dklen=16)

        conn = sqlite3.connect(str(tmp))
        cur = conn.cursor()
        cur.execute(
            "SELECT encrypted_value FROM cookies "
            "WHERE host_key LIKE '%bitbucket.org%' AND name = ?",
            (TOKEN_NAME,),
        )
        row = cur.fetchone()
        conn.close()

        if not row:
            return None

        enc = row[0]
        if enc[:3] != b"v10":
            return None

        # Try cryptography library first
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
            from cryptography.hazmat.backends import default_backend
            iv = b" " * 16
            cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
            dec = cipher.decryptor()
            raw = dec.update(enc[3:]) + dec.finalize()
            # Strip PKCS7 padding and the leading 16-byte prefix Chrome adds
            pad = raw[-1]
            return raw[16:-pad].decode("utf-8")
        except ImportError:
            pass

        # Fallback: pycryptodome
        try:
            from Crypto.Cipher import AES as CryptoAES
            iv = b" " * 16
            c = CryptoAES.new(key, CryptoAES.MODE_CBC, IV=iv)
            raw = c.decrypt(enc[3:])
            pad = raw[-1]
            return raw[16:-pad].decode("utf-8")
        except ImportError:
            pass

    except Exception:
        pass
    finally:
        if tmp.exists():
            tmp.unlink()
    return None


def from_firefox():
    ff_dir = HOME / "Library/Application Support/Firefox/Profiles"
    if not ff_dir.exists():
        return None
    for profile in ff_dir.iterdir():
        db = profile / "cookies.sqlite"
        if not db.exists():
            continue
        tmp = Path("/tmp/bb_ff_cookies_tmp")
        shutil.copy2(db, tmp)
        try:
            conn = sqlite3.connect(str(tmp))
            cur = conn.cursor()
            cur.execute(
                "SELECT value FROM moz_cookies "
                "WHERE host LIKE '%bitbucket.org%' AND name = ?",
                (TOKEN_NAME,),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return row[0]
        except Exception:
            pass
        finally:
            if tmp.exists():
                tmp.unlink()
    return None


def from_safari():
    # Safari uses binary cookies — parse with Python struct
    bc = HOME / "Library/Cookies/Cookies.binarycookies"
    if not bc.exists():
        return None
    try:
        import struct
        with open(bc, "rb") as f:
            magic = f.read(4)
            if magic != b"cook":
                return None
            num_pages = struct.unpack(">I", f.read(4))[0]
            page_sizes = [struct.unpack(">I", f.read(4))[0] for _ in range(num_pages)]
            for size in page_sizes:
                page = f.read(size)
                # Page header: magic(4) + num_cookies(4) + offsets(4*n)
                num_cookies = struct.unpack("<I", page[4:8])[0]
                offsets = [struct.unpack("<I", page[8 + i*4:12 + i*4])[0] for i in range(num_cookies)]
                for off in offsets:
                    # Cookie record: size(4)+flags(4)+...+name(null-term)+value(null-term)+...
                    rec_size = struct.unpack("<I", page[off:off+4])[0]
                    rec = page[off:off+rec_size]
                    name_off  = struct.unpack("<I", rec[16:20])[0]
                    value_off = struct.unpack("<I", rec[20:24])[0]
                    domain_off = struct.unpack("<I", rec[24:28])[0]
                    def cstr(data, start):
                        end = data.index(b"\x00", start)
                        return data[start:end].decode("utf-8", errors="ignore")
                    domain = cstr(rec, domain_off)
                    name   = cstr(rec, name_off)
                    if "bitbucket.org" in domain and name == TOKEN_NAME:
                        return cstr(rec, value_off)
    except Exception:
        pass
    return None


for extractor in [from_chrome, from_firefox, from_safari]:
    token = extractor()
    if token:
        print(token)
        sys.exit(0)

sys.exit(1)
PYEOF
python3 /tmp/bb_extract_cookie.py
```

If the script prints a token, set it and verify:

```bash
export BITBUCKET_SESSION_TOKEN="$(python3 /tmp/bb_extract_cookie.py)"

# Confirm the session is still active
curl -s -o /dev/null -w "Auth check: %{http_code}\n" \
  "https://api.bitbucket.org/2.0/user"
# 200 = success  |  401 = session expired, use Method B
```

#### Method B — Manual copy from DevTools (session expired or extraction failed)

1. Open any Bitbucket page in the browser where you are logged in
2. Open DevTools:
   - **Chrome / Edge**: `F12` or `⌘⌥I` → **Application** → **Storage** → **Cookies** → `https://bitbucket.org`
   - **Firefox**: `F12` → **Storage** → **Cookies** → `https://bitbucket.org`
   - **Safari**: Enable DevTools via Preferences → Advanced → "Show Develop menu" → **Develop** → **Show Web Inspector** → **Storage** → **Cookies**
3. Find `cloud.session.token`, double-click the Value cell, copy it
4. Paste into the terminal:

```bash
export BITBUCKET_SESSION_TOKEN="<paste here>"
```

#### Method C — App Password or Access Token (no browser, CI/CD use)

```bash
# App Password (Bitbucket → Settings → App passwords → scopes: PR read + Repo read)
export BITBUCKET_USERNAME="your-username"
export BITBUCKET_APP_PASSWORD="your-app-password"

# OR: Workspace/Repo Access Token (Workspace/Repo Settings → Access tokens)
export BITBUCKET_TOKEN="your-access-token"
```

---

### Step 3 — Write the bb_curl helper

This function is used by the Terraform PR Reviewer agent for all API calls:

```bash
bb_curl() {
  if   [ -n "$BITBUCKET_SESSION_TOKEN" ]; then
    curl -s -H "Cookie: cloud.session.token=${BITBUCKET_SESSION_TOKEN}" "$@"
  elif [ -n "$BITBUCKET_TOKEN" ]; then
    curl -s -H "Authorization: Bearer ${BITBUCKET_TOKEN}" "$@"
  elif [ -n "$BITBUCKET_USERNAME" ] && [ -n "$BITBUCKET_APP_PASSWORD" ]; then
    curl -s -u "${BITBUCKET_USERNAME}:${BITBUCKET_APP_PASSWORD}" "$@"
  else
    echo "[ERROR] No Bitbucket auth configured." >&2; return 1
  fi
}
```

---

### Step 4 — Verify Jira credentials (if Jira context is requested)

```bash
echo "JIRA_BASE_URL: ${JIRA_BASE_URL:-(NOT SET)}"
echo "JIRA_EMAIL:    ${JIRA_EMAIL:-(NOT SET)}"
# [REMOVED: secret was being echoed]:+(SET)}"
```

To get a Jira API token: <https://id.atlassian.com/manage-profile/security/api-tokens>

```bash
export JIRA_BASE_URL="https://your-org.atlassian.net"
export JIRA_EMAIL="your@email.com"
export JIRA_TOKEN="your-jira-api-token"
```

---

### Step 5 — Smoke-test the PR URL

```bash
WORKSPACE=$(echo "$PR_URL" | sed -E 's|https://bitbucket.org/([^/]+)/.*|\1|')
REPO=$(echo "$PR_URL"      | sed -E 's|https://bitbucket.org/[^/]+/([^/]+)/.*|\1|')
PR_ID=$(echo "$PR_URL"     | sed -E 's|.*pull-requests/([0-9]+).*|\1|')

bb_curl -o /dev/null -w "PR check: %{http_code}\n" \
  "https://api.bitbucket.org/2.0/repositories/${WORKSPACE}/${REPO}/pullrequests/${PR_ID}"
# 200 = OK  |  401 = auth failed  |  404 = wrong URL or no access
```

---

### Step 6 — Run the Terraform PR Reviewer agent

Invoke the **Terraform PR Reviewer** agent with:

- PR URL + parsed coordinates (workspace, repo, PR ID)
- Auth method resolved: browser session / token / app password
- Design context: Jira key and/or Technical Design MD path
- Whether to post inline comments (held until user confirms)

---

### Step 7 — Present the review

After the agent completes, surface:

- Summary table: BLOCKER / WARNING / INFO counts
- All findings with file, line, diff excerpt, explanation, and fix
- Design coverage table
- Verdict: Approve or Request Changes

---

## Quick invocation examples

```
/bitbucket-pr-review
PR URL: https://bitbucket.org/my-org/infra/pull-requests/42
Design: INFRA-1234

/bitbucket-pr-review
PR URL: https://bitbucket.org/my-org/infra/pull-requests/99
Design: ./docs/network-redesign.md
Post comments: yes
```

---

## Environment variable reference

| Variable | Used when | Description |
|---|---|---|
| `BITBUCKET_SESSION_TOKEN` | Browser session | Value of `cloud.session.token` cookie |
| `BITBUCKET_USERNAME` | App Password | Bitbucket username |
| `BITBUCKET_APP_PASSWORD` | App Password | App password (PR read + Repo read scopes) |
| `BITBUCKET_TOKEN` | Access Token | Workspace or repo access token |
| `JIRA_BASE_URL` | Jira context | e.g. `https://your-org.atlassian.net` |
| `JIRA_EMAIL` | Jira context | Account email for Jira API |
| `JIRA_TOKEN` | Jira context | Jira API token |

Store long-lived credentials in a local env file (never commit it):

```bash
# .env.local — add to .gitignore
export BITBUCKET_USERNAME=...
export BITBUCKET_APP_PASSWORD=...
export JIRA_BASE_URL=https://your-org.atlassian.net
export JIRA_EMAIL=...
export JIRA_TOKEN=...
```

```bash
source .env.local && /bitbucket-pr-review
```
