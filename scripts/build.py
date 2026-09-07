#!/usr/bin/env python3
"""
build.py — Discovers all .md files and builds the corresponding HTML pages.

Usage:
  python build.py           # rebuild all discovered pages
  python build.py architect # rebuild only matching pages (substring)

To skip a file, add its name to SKIP below.
To give a new file a custom nav label or icon, add it to LABELS / ICONS.
"""

import re
import shutil
import sys
from pathlib import Path

# DOCS = Path(__file__).parent
DOCS = Path.cwd()
SITE = DOCS / "site"
SITE_TITLE = "NEKO WIKI"
# Markdown files that are not standalone pages
SKIP = {"DEPLOYMENT.md"}

# Special-case: filename → html output name (everything else uses stem + ".html")
MD_TO_HTML = {"overview.md": "index.html"}

# Optional nav-label overrides keyed by filename stem
LABELS = {
    "overview":     "Overview",
    "architect":    "Architecture",
    "architecture": "Architecture",
    "brainstorm":   "Brainstorm",
    "guardrails":   "Guardrails",
    "memory":       "Memory &amp; Cache",
    "rag_pipeline": "RAG Pipeline",
    "project_plan": "Project Plan",
}

# Optional icon overrides keyed by output html filename
ICONS = {
    "index.html":        "◈",
    "architect.html":    "⬡",
    "brainstorm.html":   "◎",
    "guardrails.html":   "◼",
    "memory.html":       "◑",
    "rag_pipeline.html": "≋",
    "project_plan.html": "▦",
}
DEFAULT_ICON = "◆"

# Tokens that should be uppercased in auto-generated labels
UPPER_WORDS = {"fin", "ai", "rag", "api", "gcp", "llm", "pii", "ui", "ux"}


def stem_to_label(stem: str) -> str:
    if stem in LABELS:
        return LABELS[stem]
    words = stem.replace("-", "_").split("_")
    return " ".join(w.upper() if w.lower() in UPPER_WORDS else w.title() for w in words)


def discover_pages() -> list[dict]:
    pages = []
    for md_path in sorted(DOCS.rglob("*.md")):
        if md_path.is_relative_to(SITE):
            continue
        if any(p.name in {".venv", ".claude"} for p in md_path.parents):
            continue
        if md_path.name in SKIP:
            continue
        rel = md_path.relative_to(DOCS)
        flat_name = "-".join(rel.with_suffix("").parts) + ".html"
        html_name = MD_TO_HTML.get(md_path.name, flat_name)
        pages.append({
            "html":      html_name,
            "flat_name": flat_name.removesuffix(".html"),
            "md":        str(rel),
            "label":     stem_to_label(md_path.stem),
            "icon":  ICONS.get(html_name, DEFAULT_ICON),
        })
    # index.html first, then alphabetical
    pages.sort(key=lambda p: (0 if p["html"] == "index.html" else 1, p["html"]))

    # If no md file maps to index.html, auto-generate a listing page
    if not any(p["html"] == "index.html" for p in pages):
        lines = ["# Documentation\n"]
        for p in pages:
            lines.append(f"- [{p['label']}]({p['html']})")
        pages.insert(0, {
            "html":     "index.html",
            "md":       None,
            "label":    "Overview",
            "icon":     ICONS.get("index.html", DEFAULT_ICON),
            "_content": "\n".join(lines),
        })

    return pages


def nav_items(pages: list[dict], active_html: str) -> str:
    rows = []
    for p in pages:
        cls = "nav-item active" if p["html"] == active_html else "nav-item"
        rows.append(
            f'    <a class="{cls}" href="{p["html"]}">'
            f'<span class="nav-icon">{p["icon"]}</span>{p.get("flat_name", p["label"])}</a>'
        )
    return "\n".join(rows)


def render(page: dict, pages: list[dict], md_content: str) -> str:
    depth = len(Path(page["html"]).parts) - 1
    root = "../" * depth
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{page["label"]} | {SITE_TITLE}</title>
  <link rel="stylesheet" href="{root}style.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
</head>
<body>

<aside class="sidebar">
  <div class="sidebar-head">
    <div class="sidebar-eyebrow">GCP · Python · ADK</div>
    <div class="sidebar-title">{SITE_TITLE}</div>
    <div class="sidebar-subtitle">Docs</div>
  </div>
  <nav class="nav-group">
    <div class="nav-label">Pages</div>
{nav_items(pages, page["html"])}
  </nav>
  <div class="sidebar-footer">v1.0 · 18-week build</div>
</aside>

<div class="main-wrap">
  <div class="content" id="content"></div>
  <nav class="toc" id="toc">
    <div class="toc-label">On this page</div>
    <ul class="toc-list" id="toc-list"></ul>
  </nav>
</div>

<script type="text/markdown" id="md-source">
{md_content}
</script>

<script src="https://cdn.jsdelivr.net/npm/marked@4/marked.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="{root}app.js"></script>
<script>renderPage();</script>
</body>
</html>
"""


def resolve_images(md_content: str, md_file: Path) -> str:
    def replace(m):
        alt, src = m.group(1), m.group(2)
        if src.startswith(("http://", "https://", "data:", "/")):
            return m.group(0)
        img_abs = (md_file.parent / src).resolve()
        if not img_abs.is_file():
            return m.group(0)
        try:
            rel = img_abs.relative_to(DOCS)
        except ValueError:
            return m.group(0)
        flat = "-".join(rel.parts)
        shutil.copy2(img_abs, SITE / flat)
        return f"![{alt}]({flat})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", replace, md_content)


def build_page(page: dict, pages: list[dict]) -> None:
    html_path = SITE / page["html"]
    html_path.parent.mkdir(parents=True, exist_ok=True)
    if page.get("md"):
        md_file = DOCS / page["md"]
        md_content = md_file.read_text(encoding="utf-8").strip()
        md_content = resolve_images(md_content, md_file)
        source = page["md"]
    else:
        md_content = page.get("_content", "")
        source = "(auto-generated)"
    html_path.write_text(render(page, pages, md_content), encoding="utf-8")
    print(f"  OK    {source} → {page['html']}")


def main() -> None:
    targets = sys.argv[1:]
    pages = discover_pages()  # full list — always used for the nav

    selected = (
        [p for p in pages if any(t in p["html"] or (p["md"] and t in p["md"]) for t in targets)]
        if targets else pages
    )

    if not selected:
        print(f"No pages matched: {targets}")
        sys.exit(1)

    print(f"Discovered {len(pages)} page(s), building {len(selected)} → {SITE}...")
    for page in selected:
        build_page(page, pages)
    print("Done.")


if __name__ == "__main__":
    main()
