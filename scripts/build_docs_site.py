#!/usr/bin/env python3
"""Build the Mintlify-style static documentation site into site/docs/.

Single source of truth: reads Markdown files from site/docs/raw/, skills/,
and src/dashboard-rs/README.md, then writes
self-contained static HTML, search-index.json, and llms.txt into site/docs/
for local preview and GitHub Pages deployment.

Usage:
    python3 scripts/build_docs_site.py
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
OUT_DIR = SITE_DIR / "docs"
REPO_BLOB_BASE = "https://github.com/Tongyi-MAI/MAI-Gym/blob/main"
REPO_URL = "https://github.com/Tongyi-MAI/MAI-Gym"
PAPER_URL = "https://arxiv.org/abs/2610.20939"
MODEL_URL = "https://huggingface.co/Tongyi-MAI/MAI-Gym-9B"
DOCS_BASE_URL = "https://tongyi-mai.github.io/MAI-Gym/docs"
OG_IMAGE_URL = "https://tongyi-mai.github.io/MAI-Gym/assets/figs/overview.webp"


@dataclass(frozen=True)
class PageSpec:
    slug: str  # output path relative to site/docs/ (e.g. "index.html", "envs/mobile.html")
    source: str  # source path relative to repo root
    title: str
    description: str
    group: str  # unified vertical sidebar section & eyebrow label


PAGES: list[PageSpec] = [
    # Overview
    PageSpec(
        slug="index.html",
        source="site/docs/raw/index.md",
        title="Quickstart",
        description="Install the MAI-Gym CLI and run your first verified GUI agent evaluation—locally on Mocked Apps (no Docker/KVM) or on MobileWorld—plus visual replay and optional online RL.",
        group="Overview",
    ),
    PageSpec(
        slug="concepts.html",
        source="site/docs/raw/concepts.md",
        title="Platform & Core Concepts",
        description="System architecture, executable task model, shared rollout harness, EnvClient lifecycle, Hybrid Action Space (GUI + CLI), and the evaluate–diagnose–synthesize–retrain loop.",
        group="Overview",
    ),
    PageSpec(
        slug="mai-gym-9b.html",
        source="site/docs/raw/mai-gym-9b.md",
        title="MAI-Gym-9B",
        description="Open 9B GUI agent trained with joint mobile–computer online RL: benchmark results, clone-to-real transfer highlights, and quick evaluation setup.",
        group="Overview",
    ),
    PageSpec(
        slug="dashboard.html",
        source="src/dashboard-rs/README.md",
        title="Trajectory Dashboard",
        description="Visual Control Room tour plus setup for the embedded Rust/SQLite dashboard server, indexer, and HTTP diagnosis API.",
        group="Overview",
    ),
    # User Guide
    PageSpec(
        slug="evaluation.html",
        source="site/docs/raw/evaluation.md",
        title="Evaluation Guide",
        description="Run single- and multi-benchmark GUI agent evaluations with native verifiers, pass@N metrics, and automatic SGLang serving.",
        group="User Guide",
    ),
    PageSpec(
        slug="training.html",
        source="site/docs/raw/training.md",
        title="Online RL Training",
        description="Train GUI agents with online RL (GRPO, DAPO, GSPO, REINFORCE++, SAO, PPO) using policy-adaptive sampling and window-wise packing.",
        group="User Guide",
    ),
    PageSpec(
        slug="rollout.html",
        source="site/docs/raw/rollout.md",
        title="Rollout CLI",
        description="Run direct task rollouts for agent debugging, trajectory collection, multi-pass execution, and controlled retries.",
        group="User Guide",
    ),
    PageSpec(
        slug="kubernetes.html",
        source="site/docs/raw/kubernetes.md",
        title="Kubernetes Fleets",
        description="Scale MobileWorld environments across Kubernetes nodes with SandboxSet warm pools, session leasing, and demand autoscaling.",
        group="User Guide",
    ),
    # Benchmarks & Environments
    PageSpec(
        slug="environments.html",
        source="site/docs/raw/environments.md",
        title="Benchmarks Overview",
        description="Overview of all 10 supported GUI benchmarks across Mobile, Computer, and Web alongside Mocked Apps, native metrics, and reproducible setup skills.",
        group="Benchmarks & Environments",
    ),
    PageSpec(
        slug="mocked-apps.html",
        source="site/docs/raw/mocked-apps.md",
        title="Mocked Apps",
        description="Self-contained, resettable Real-World Clones and 42 bundled Novel Apps (837 tasks) runnable via --benchmarks mocked-apps with native programmatic verifiers.",
        group="Benchmarks & Environments",
    ),
    PageSpec(
        slug="envs/mobile.html",
        source="site/docs/raw/envs/mobile.md",
        title="Mobile Benchmarks",
        description="Overview of MobileWorld, AndroidWorld, and MobileGym: task scope, Android/mobile runtimes, verification protocols, and MAI-Gym-9B results.",
        group="Benchmarks & Environments",
    ),
    PageSpec(
        slug="envs/computer.html",
        source="site/docs/raw/envs/computer.md",
        title="Computer Benchmarks",
        description="Overview of OSWorld-Verified, OSWorld 2.0, MacAgentBench, and MyPCBench: desktop VM suites, Hybrid Action Space (GUI + CLI), and evaluation protocols.",
        group="Benchmarks & Environments",
    ),
    PageSpec(
        slug="envs/web.html",
        source="site/docs/raw/envs/web.md",
        title="Web Benchmarks",
        description="Overview of WebVoyager, Online-Mind2Web, and WebArena: live-web and self-hosted multi-site browser benchmarks with VLM judges and programmatic evaluators.",
        group="Benchmarks & Environments",
    ),
    # Customization
    PageSpec(
        slug="extending.html",
        source="site/docs/raw/extending.md",
        title="Extending MAI-Gym",
        description="Four extension points (harnesses, tasks, adapters, environment specs) and the 5-operation environment contract.",
        group="Customization",
    ),
    PageSpec(
        slug="coding-agents.html",
        source="skills/README.md",
        title="Coding-Agent Setup",
        description="Reusable setup prompts for Claude Code, Codex, and Cursor to automate benchmark preparation, evaluation, and training.",
        group="Customization",
    ),
    PageSpec(
        slug="development.html",
        source="site/docs/raw/development.md",
        title="Development & Layout",
        description="Repository layout, minimal mobile-agent extraction subset, and local development commands.",
        group="Customization",
    ),
    PageSpec(
        slug="licensing.html",
        source="site/docs/raw/licensing.md",
        title="Licensing & Notices",
        description="MIT project license and third-party notices for Slime (Apache-2.0), TileLang, and uPlot.",
        group="Customization",
    ),
]


def _build_nav_groups(pages: list[PageSpec]) -> list[tuple[str, list[str]]]:
    groups: list[tuple[str, list[str]]] = []
    index: dict[str, list[str]] = {}
    for p in pages:
        if p.group not in index:
            slugs: list[str] = []
            index[p.group] = slugs
            groups.append((p.group, slugs))
        index[p.group].append(p.slug)
    return groups


NAV_GROUPS: list[tuple[str, list[str]]] = _build_nav_groups(PAGES)
FLAT_NAV: list[str] = [p.slug for p in PAGES]
PAGE_BY_SLUG: dict[str, PageSpec] = {p.slug: p for p in PAGES}
SOURCE_TO_SLUG: dict[str, str] = {p.source: p.slug for p in PAGES}


def github_slug(heading: str) -> str:
    s = heading.strip().lower()
    s = re.sub(r"`([^`]*)`", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", s)
    out = [ch for ch in s if ch.isalnum() or ch in (" ", "-", "_")]
    return "".join(out).replace(" ", "-")


def rel_href(from_slug: str, to_slug: str) -> str:
    from_dir = posixpath.dirname(from_slug) or "."
    return posixpath.relpath(to_slug, from_dir)


def asset_prefix(from_slug: str) -> str:
    """Prefix to reach site/docs/ root from a page slug."""
    depth = from_slug.count("/")
    return "../" * depth


def site_root_prefix(from_slug: str) -> str:
    """Prefix to reach site/ root from a page slug inside site/docs/."""
    return "../" * (from_slug.count("/") + 1)


def rewrite_link(raw_target: str, page: PageSpec) -> tuple[str, bool]:
    """Rewrite a Markdown link target into a valid docs-site URL.

    Returns (url, is_external).
    """
    target = raw_target.strip()
    if not target:
        return "#", False
    if target.startswith("#"):
        return target, False
    if target.startswith(("http://", "https://", "mailto:")):
        return target, True

    path_part, sep, frag = target.partition("#")
    suffix = f"#{frag}" if sep else ""

    source_dir = posixpath.dirname(page.source)
    repo_rel = posixpath.normpath(posixpath.join(source_dir, path_part))
    if repo_rel.startswith(".."):
        return f"{REPO_URL}", True

    # Special mappings for root README anchors
    if repo_rel == "README.md":
        if frag in ("train", "4-train-a-gui-agent-with-online-rl-optional"):
            return rel_href(page.slug, "training.html"), False
        if frag in ("quickstart", "getting-started"):
            return rel_href(page.slug, "index.html"), False
        return site_root_prefix(page.slug) + "index.html", False

    # Mapped docs page?
    if repo_rel in SOURCE_TO_SLUG:
        dest_slug = SOURCE_TO_SLUG[repo_rel]
        return rel_href(page.slug, dest_slug) + suffix, False

    # Asset under site/
    if repo_rel.startswith("site/"):
        sub = repo_rel[len("site/") :]
        return site_root_prefix(page.slug) + sub + suffix, False

    # Fallback: link to file/dir on GitHub (validate target exists in repository)
    full_path = ROOT / repo_rel
    if not full_path.exists():
        raise FileNotFoundError(
            f"[{page.source}] Broken repository link: {raw_target!r} -> {full_path}"
        )
    kind = "tree" if full_path.is_dir() else "blob"
    return f"https://github.com/Tongyi-MAI/MAI-Gym/{kind}/main/{repo_rel}{suffix}", True


def highlight_code(code: str, lang: str) -> str:
    """Lightweight deterministic syntax highlighter for bash, yaml, json, python."""
    lang = (lang or "").lower().strip()
    escaped = html.escape(code)

    if lang in ("bash", "sh", "shell", "zsh"):
        lines = []
        for line in escaped.splitlines():
            if re.match(r"^\s*#", line):
                lines.append(f'<span class="tok-comment">{line}</span>')
                continue
            # Highlight inline trailing comment (outside quotes approximation)
            comment = ""
            m_com = re.search(r"(\s+#.*)$", line)
            if m_com and not re.search(r'["\'][^"\']*#', line):
                comment = f'<span class="tok-comment">{m_com.group(1)}</span>'
                line = line[: m_com.start()]
            # Strings
            line = re.sub(
                r"(&quot;[^&]*?&quot;|&#x27;[^&]*?&#x27;)",
                r'<span class="tok-string">\1</span>',
                line,
            )
            # Env vars like $HOME, ${FOO}, VAR=
            line = re.sub(
                r"(\$\{?[A-Za-z_][A-Za-z0-9_]*\}?)",
                r'<span class="tok-var">\1</span>',
                line,
            )
            line = re.sub(
                r"^(\s*)([A-Z_][A-Z0-9_]*)(=)",
                r'\1<span class="tok-var">\2</span>\3',
                line,
            )
            # Flags like --config, -fsS
            line = re.sub(
                r"(?<!\w)(--?[A-Za-z0-9][A-Za-z0-9_-]*)",
                r'<span class="tok-flag">\1</span>',
                line,
            )
            # Leading command keyword
            line = re.sub(
                r"^(\s*)(uv|uvx|git|cd|mkdir|cp|curl|docker|kubectl|python3?|ray|mai-gym|cargo|export|printf|umask|test|cd)\b",
                r'\1<span class="tok-cmd">\2</span>',
                line,
            )
            lines.append(line + comment)
        return "\n".join(lines)

    if lang in ("yaml", "yml"):
        lines = []
        for line in escaped.splitlines():
            if re.match(r"^\s*#", line):
                lines.append(f'<span class="tok-comment">{line}</span>')
                continue
            comment = ""
            m_com = re.search(r"(\s+#.*)$", line)
            if m_com:
                comment = f'<span class="tok-comment">{m_com.group(1)}</span>'
                line = line[: m_com.start()]
            m_kv = re.match(r"^(\s*(?:-\s+)?)([A-Za-z0-9_.-]+)(:)(\s*)(.*)$", line)
            if m_kv:
                prefix, key, colon, sp, val = m_kv.groups()
                if val in ("true", "false", "null"):
                    val_html = f'<span class="tok-kw">{val}</span>'
                elif re.match(r"^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", val, re.I):
                    val_html = f'<span class="tok-num">{val}</span>'
                elif val:
                    val_html = f'<span class="tok-string">{val}</span>'
                else:
                    val_html = ""
                lines.append(
                    f'{prefix}<span class="tok-key">{key}</span>{colon}{sp}{val_html}{comment}'
                )
            else:
                lines.append(line + comment)
        return "\n".join(lines)

    if lang == "json":
        escaped = re.sub(
            r"(&quot;[^&]*?&quot;)(\s*:)",
            r'<span class="tok-key">\1</span>\2',
            escaped,
        )
        escaped = re.sub(
            r"(:\s*)(&quot;[^&]*?&quot;)",
            r'\1<span class="tok-string">\2</span>',
            escaped,
        )
        escaped = re.sub(
            r"(:\s*)(true|false|null)\b",
            r'\1<span class="tok-kw">\2</span>',
            escaped,
        )
        escaped = re.sub(
            r"(:\s*)(-?\d+(?:\.\d+)?)",
            r'\1<span class="tok-num">\2</span>',
            escaped,
        )
        return escaped

    if lang in ("python", "py"):
        escaped = re.sub(
            r"\b(from|import|def|class|return|if|else|elif|for|in|with|as|try|except|raise|True|False|None)\b",
            r'<span class="tok-kw">\1</span>',
            escaped,
        )
        return escaped

    return escaped


def render_inline(text: str, page: PageSpec) -> str:
    """Render inline Markdown (code, images, links, bold, italic, strikethrough)."""
    placeholders: dict[str, str] = {}

    def stash(html_str: str) -> str:
        key = f"\x00PH{len(placeholders)}\x00"
        placeholders[key] = html_str
        return key

    # 1. Inline code spans
    def repl_code(m: re.Match[str]) -> str:
        return stash(f"<code>{html.escape(m.group(1))}</code>")

    text = re.sub(r"`([^`]+)`", repl_code, text)

    # 2. Images ![alt](src)
    def repl_img(m: re.Match[str]) -> str:
        alt = html.escape(m.group(1))
        src, _ = rewrite_link(m.group(2), page)
        return stash(
            f'<figure class="doc-figure"><img src="{html.escape(src)}" alt="{alt}" loading="lazy" data-zoomable>'
            + (f"<figcaption>{alt}</figcaption>" if alt else "")
            + "</figure>"
        )

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl_img, text)

    # 3. Links [label](href)
    def repl_link(m: re.Match[str]) -> str:
        label_html = render_inline(m.group(1), page)
        href, is_ext = rewrite_link(m.group(2), page)
        extra = ' target="_blank" rel="noopener noreferrer"' if is_ext else ""
        ext_icon = (
            '<svg class="ext-ico" viewBox="0 0 12 12" aria-hidden="true">'
            '<path d="M3.5 3h5.5v5.5M9 3 3 9" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round"/></svg>'
            if is_ext
            else ""
        )
        return stash(f'<a href="{html.escape(href)}"{extra}>{label_html}{ext_icon}</a>')

    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl_link, text)

    # Escape remaining raw HTML text
    text = html.escape(text)

    # 4. Bold + italic, bold, italic, strikethrough
    text = re.sub(r"\*\*\*([^*]+)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__([^_]+)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", text)
    text = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", text)

    # Restore newest-first: a stashed link can contain earlier placeholders
    # (e.g. an inline-code label), which must be expanded after the link is.
    for key, val in reversed(placeholders.items()):
        text = text.replace(key, val)
    return text


def strip_markdown_plain(md: str) -> str:
    """Convert Markdown text to clean plain text for search indexing."""
    s = re.sub(r"```.*?```", " ", md, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"[#>*`_~|]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


CALLOUT_META = {
    "NOTE": ("Note", "note"),
    "TIP": ("Tip", "tip"),
    "IMPORTANT": ("Important", "important"),
    "WARNING": ("Warning", "warning"),
    "CAUTION": ("Caution", "warning"),
}


def render_markdown(md_text: str, page: PageSpec) -> tuple[str, str, list[tuple[int, str, str]]]:
    """Render Markdown to (page_h1, body_html, toc_entries).

    toc_entries is a list of (level, id, plain_title) for H2 and H3 headings.
    """
    lines = md_text.splitlines()
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    slug_counts: dict[str, int] = {}
    page_h1 = page.title
    pending_anchors: list[str] = []

    i = 0
    n = len(lines)

    # Skip leading H1 and Markdown-only breadcrumb line immediately following it
    while i < n and not lines[i].strip():
        i += 1
    if i < n and lines[i].startswith("# "):
        page_h1 = lines[i][2:].strip()
        i += 1
        while i < n and not lines[i].strip():
            i += 1
        # Strip manual breadcrumb or language-switcher lines immediately below H1
        # (e.g., `[Home](../README.md) · ...`, `[All environments](../envs.md)`, `English · [简体中文](...)`)
        if i < n and (
            "·" in lines[i]
            and (
                "[Home](" in lines[i]
                or "[Documentation](" in lines[i]
                or "[Docs Hub](" in lines[i]
                or "[Overview](" in lines[i]
                or lines[i].strip().startswith("English ·")
            )
            or lines[i].strip().startswith("[All environments](")
            or lines[i].strip().startswith("[Setup index](")
        ):
            i += 1

    def next_heading_id(raw_title: str) -> str:
        base = github_slug(raw_title) or "section"
        if base in slug_counts:
            slug_counts[base] += 1
            return f"{base}-{slug_counts[base]}"
        slug_counts[base] = 0
        return base

    def flush_pending_anchors() -> str:
        if not pending_anchors:
            return ""
        spans = "".join(
            f'<span id="{html.escape(a)}" class="doc-anchor"></span>' for a in pending_anchors
        )
        pending_anchors.clear()
        return spans

    def is_table_sep(line: str) -> bool:
        s = line.strip()
        if "|" not in s:
            return False
        cells = [c.strip() for c in s.strip("|").split("|")]
        return bool(cells) and all(re.match(r"^:?-{2,}:?$", c) for c in cells)

    def parse_table_row(line: str) -> list[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [c.strip() for c in s.split("|")]

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Explicit <a name="..."></a> anchor lines
        m_anchor = re.match(r'^<a\s+(?:name|id)=["\']([^"\']+)["\']\s*>\s*</a>$', stripped)
        if m_anchor:
            pending_anchors.append(m_anchor.group(1))
            i += 1
            continue

        # Raw centered image block <p align="center"> ... </p>
        if stripped.startswith('<p align="center">'):
            block_lines = [stripped]
            i += 1
            while i < n and "</p>" not in block_lines[-1]:
                block_lines.append(lines[i].strip())
                i += 1
            block = "\n".join(block_lines)
            m_img = re.search(r'<img\s+[^>]*src=["\']([^"\']+)["\'][^>]*>', block)
            if m_img:
                src, _ = rewrite_link(m_img.group(1), page)
                m_alt = re.search(r'alt=["\']([^"\']*)["\']', m_img.group(0))
                alt = html.escape(m_alt.group(1) if m_alt else "")
                out.append(
                    f'{flush_pending_anchors()}<figure class="doc-figure">'
                    f'<img src="{html.escape(src)}" alt="{alt}" loading="lazy" data-zoomable>'
                    + (f"<figcaption>{alt}</figcaption>" if alt else "")
                    + "</figure>"
                )
            continue

        # Standalone Markdown image(s): 2+ back-to-back lines render in a responsive figure grid
        m_img_line = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)
        if m_img_line:
            figures: list[str] = []
            while i < n:
                m_cur = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", lines[i].strip())
                if not m_cur:
                    break
                alt = html.escape(m_cur.group(1))
                src, _ = rewrite_link(m_cur.group(2), page)
                figures.append(
                    f'<figure class="doc-figure">'
                    f'<img src="{html.escape(src)}" alt="{alt}" loading="lazy" data-zoomable>'
                    + (f"<figcaption>{alt}</figcaption>" if alt else "")
                    + "</figure>"
                )
                i += 1
            if len(figures) >= 2:
                out.append(
                    f'{flush_pending_anchors()}<div class="figure-grid">{"".join(figures)}</div>'
                )
            else:
                out.append(f"{flush_pending_anchors()}{figures[0]}")
            continue

        # Horizontal rule
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", stripped):
            out.append("<hr>")
            i += 1
            continue

        # Headings H2 - H4
        m_head = re.match(r"^(#{2,4})\s+(.*)$", stripped)
        if m_head:
            level = len(m_head.group(1))
            raw_title = m_head.group(2).strip()
            hid = next_heading_id(raw_title)
            anchors_html = flush_pending_anchors()
            plain = strip_markdown_plain(raw_title)
            if level in (2, 3):
                toc.append((level, hid, plain))
            out.append(
                f'{anchors_html}<h{level} id="{html.escape(hid)}">'
                f"{render_inline(raw_title, page)}"
                f'<a class="heading-link" href="#{html.escape(hid)}" aria-label="Link to {html.escape(plain)}">#</a>'
                f"</h{level}>"
            )
            i += 1
            continue

        # Fenced code block
        m_fence = re.match(r"^```(\S*)\s*$", stripped)
        if m_fence:
            lang = m_fence.group(1) or "text"
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < n:
                i += 1
            raw_code = "\n".join(code_lines)
            highlighted = highlight_code(raw_code, lang)
            label = html.escape(lang)
            out.append(
                f'{flush_pending_anchors()}<div class="code-block" data-lang="{label}">'
                f'<div class="code-head"><span class="code-lang">{label}</span>'
                f'<button type="button" class="copy-btn" data-copy-code aria-label="Copy code">'
                f'<svg class="i-copy" viewBox="0 0 16 16" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3.5 10.5h-1a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v1" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>'
                f'<svg class="i-check" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.2 8.4 6.5 11.7 13 4.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'
                f"<span>Copy</span></button></div>"
                f'<pre tabindex="0"><code class="lang-{label}">{highlighted}</code></pre></div>'
            )
            continue

        # Blockquote / GitHub Callout Alert
        if stripped.startswith(">"):
            bq_lines: list[str] = []
            while i < n and lines[i].strip().startswith(">"):
                content = re.sub(r"^>\s?", "", lines[i].strip())
                bq_lines.append(content)
                i += 1
            first = bq_lines[0].strip() if bq_lines else ""
            m_alert = re.match(r"^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]$", first)
            if m_alert:
                kind = m_alert.group(1)
                label, css_kind = CALLOUT_META[kind]
                inner_md = "\n".join(bq_lines[1:])
                _, inner_html, _ = render_markdown(inner_md, page)
                out.append(
                    f'{flush_pending_anchors()}<div class="callout callout-{css_kind}" role="note">'
                    f'<div class="callout-title"><span class="callout-dot" aria-hidden="true"></span>{label}</div>'
                    f'<div class="callout-body">{inner_html}</div></div>'
                )
            else:
                inner_md = "\n".join(bq_lines)
                _, inner_html, _ = render_markdown(inner_md, page)
                out.append(f"{flush_pending_anchors()}<blockquote>{inner_html}</blockquote>")
            continue

        # Tables
        if "|" in stripped and i + 1 < n and is_table_sep(lines[i + 1]):
            headers = parse_table_row(lines[i])
            seps = parse_table_row(lines[i + 1])
            aligns: list[str] = []
            for sep in seps:
                if sep.startswith(":") and sep.endswith(":"):
                    aligns.append("center")
                elif sep.endswith(":"):
                    aligns.append("right")
                else:
                    aligns.append("left")
            i += 2
            rows: list[list[str]] = []
            while i < n and lines[i].strip() and "|" in lines[i]:
                rows.append(parse_table_row(lines[i]))
                i += 1
            th_html = "".join(
                f'<th style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{render_inline(cell, page)}</th>'
                for idx, cell in enumerate(headers)
            )
            body_rows = []
            for row in rows:
                tds = "".join(
                    f'<td style="text-align:{aligns[idx] if idx < len(aligns) else "left"}">{render_inline(cell, page)}</td>'
                    for idx, cell in enumerate(row)
                )
                body_rows.append(f"<tr>{tds}</tr>")
            out.append(
                f'{flush_pending_anchors()}<div class="table-wrap"><table><thead><tr>{th_html}</tr></thead>'
                f"<tbody>{''.join(body_rows)}</tbody></table></div>"
            )
            continue

        # Lists (unordered or ordered, supporting indented nested items & code blocks)
        m_ul = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        m_ol = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if m_ul or m_ol:
            tag = "ul" if m_ul else "ol"
            base_indent = len((m_ul or m_ol).group(1))
            items_html: list[str] = []
            while i < n:
                cur = lines[i]
                if not cur.strip():
                    # Look ahead to see if list continues
                    if i + 1 < n and (
                        re.match(rf"^\s{{{base_indent}}}(?:[-*]|\d+\.)\s+", lines[i + 1])
                        or re.match(rf"^\s{{{base_indent + 2},}}\S", lines[i + 1])
                    ):
                        i += 1
                        continue
                    break
                item_m = (
                    re.match(rf"^ {{{base_indent}}}[-*]\s+(.*)$", cur)
                    if tag == "ul"
                    else re.match(rf"^ {{{base_indent}}}\d+\.\s+(.*)$", cur)
                )
                if not item_m:
                    break
                item_lines = [item_m.group(1)]
                i += 1
                # Collect indented continuation lines belonging to this list item
                while i < n:
                    nxt = lines[i]
                    if not nxt.strip():
                        if i + 1 < n and re.match(rf"^\s{{{base_indent + 2},}}\S", lines[i + 1]):
                            item_lines.append("")
                            i += 1
                            continue
                        break
                    if re.match(rf"^\s{{{base_indent + 2},}}\S", nxt):
                        item_lines.append(nxt[base_indent + 2 :])
                        i += 1
                    else:
                        break
                if len(item_lines) == 1:
                    items_html.append(f"<li>{render_inline(item_lines[0], page)}</li>")
                else:
                    # Check if continuation contains sub-list or fenced code block
                    rest_md = "\n".join(item_lines[1:])
                    if "```" in rest_md or re.search(r"^\s*[-*]\s+", rest_md, re.M):
                        _, rest_html, _ = render_markdown(rest_md, page)
                        items_html.append(
                            f"<li><p>{render_inline(item_lines[0], page)}</p>{rest_html}</li>"
                        )
                    else:
                        joined = " ".join(s.strip() for s in item_lines if s.strip())
                        items_html.append(f"<li>{render_inline(joined, page)}</li>")
            out.append(f"{flush_pending_anchors()}<{tag}>{''.join(items_html)}</{tag}>")
            continue

        # Standard paragraph
        para_lines = [stripped]
        i += 1
        while i < n:
            nxt = lines[i]
            ns = nxt.strip()
            if (
                not ns
                or ns.startswith(("#", "```", ">", "---", "***", "<a ", '<p align="center">'))
                or re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", ns)
                or re.match(r"^[-*]\s+", ns)
                or re.match(r"^\d+\.\s+", ns)
                or ("|" in ns and i + 1 < n and is_table_sep(lines[i + 1]))
            ):
                break
            para_lines.append(ns)
            i += 1
        out.append(f"{flush_pending_anchors()}<p>{render_inline(' '.join(para_lines), page)}</p>")

    if pending_anchors:
        out.append(flush_pending_anchors())

    return page_h1, "\n".join(out), toc


def render_page(
    page: PageSpec, h1: str, body_html: str, toc: list[tuple[int, str, str]], raw_rel: str
) -> str:
    docs_prefix = asset_prefix(page.slug)
    root_prefix = site_root_prefix(page.slug)

    # Unified vertical left sidebar (single flat hierarchy across all sections)
    sidebar_sections = []
    for group_name, slugs in NAV_GROUPS:
        items = []
        for s in slugs:
            spec = PAGE_BY_SLUG[s]
            href = rel_href(page.slug, s)
            active = (
                ' class="side-link active" aria-current="page"'
                if s == page.slug
                else ' class="side-link"'
            )
            items.append(
                f'<li><a{active} href="{html.escape(href)}">{html.escape(spec.title)}</a></li>'
            )
        sidebar_sections.append(
            f'<div class="side-group"><p class="side-heading">{html.escape(group_name)}</p>'
            f"<ul>{''.join(items)}</ul></div>"
        )

    # Linear Prev / Next pagination across the unified navigation order
    prev_html = '<span class="pag-spacer"></span>'
    next_html = '<span class="pag-spacer"></span>'
    if page.slug in FLAT_NAV:
        idx = FLAT_NAV.index(page.slug)
        if idx > 0:
            prev_spec = PAGE_BY_SLUG[FLAT_NAV[idx - 1]]
            prev_href = rel_href(page.slug, prev_spec.slug)
            prev_html = (
                f'<a class="pag-link prev" href="{html.escape(prev_href)}">'
                f'<span class="pag-label">Previous</span>'
                f'<span class="pag-title">&larr; {html.escape(prev_spec.title)}</span></a>'
            )
        if idx + 1 < len(FLAT_NAV):
            next_spec = PAGE_BY_SLUG[FLAT_NAV[idx + 1]]
            next_href = rel_href(page.slug, next_spec.slug)
            next_html = (
                f'<a class="pag-link next" href="{html.escape(next_href)}">'
                f'<span class="pag-label">Next</span>'
                f'<span class="pag-title">{html.escape(next_spec.title)} &rarr;</span></a>'
            )

    # Right TOC
    toc_items = []
    for level, hid, plain in toc:
        cls = "toc-link toc-h3" if level == 3 else "toc-link"
        toc_items.append(
            f'<li><a class="{cls}" href="#{html.escape(hid)}">{html.escape(plain)}</a></li>'
        )
    toc_block = (
        f'<ul class="toc-list" id="toc-list">{"".join(toc_items)}</ul>'
        if toc_items
        else '<p class="toc-empty">On this page</p>'
    )

    github_edit_url = f"{REPO_BLOB_BASE}/{page.source}"
    github_issue_url = f"{REPO_URL}/issues/new?title=Docs:%20{html.escape(page.title)}"
    canonical_url = (
        f"{DOCS_BASE_URL}/" if page.slug == "index.html" else f"{DOCS_BASE_URL}/{page.slug}"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{html.escape(page.description)}">
  <link rel="canonical" href="{html.escape(canonical_url)}">
  <meta property="og:site_name" content="MAI-Gym Docs">
  <meta property="og:type" content="website">
  <meta property="og:title" content="{html.escape(page.title)} — MAI-Gym Docs">
  <meta property="og:description" content="{html.escape(page.description)}">
  <meta property="og:url" content="{html.escape(canonical_url)}">
  <meta property="og:image" content="{html.escape(OG_IMAGE_URL)}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{html.escape(page.title)} — MAI-Gym Docs">
  <meta name="twitter:description" content="{html.escape(page.description)}">
  <meta name="twitter:image" content="{html.escape(OG_IMAGE_URL)}">
  <title>{html.escape(page.title)} — MAI-Gym Docs</title>
  <link rel="icon" type="image/png" href="{root_prefix}assets/favicon.png">
  <script>
    (function () {{
      var t;
      try {{ t = localStorage.getItem('maigym.theme'); }} catch (e) {{}}
      document.documentElement.dataset.theme = t === 'day' ? 'day' : 'night';
    }})();
  </script>
  <link rel="preload" href="{root_prefix}assets/fonts/inter-latin-wght-normal.woff2" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="{docs_prefix}assets/docs.css">
  <script>
    window.MAIGYM_DOCS = {{
      root: "{docs_prefix}",
      rawMd: "{html.escape(raw_rel)}",
      slug: "{html.escape(page.slug)}"
    }};
  </script>
  <script src="{docs_prefix}assets/docs.js" defer></script>
</head>
<body>
  <a class="skip-link" href="#doc-content">Skip to main content</a>

  <header class="topbar" id="topbar">
    <div class="topbar-row">
      <div class="topbar-left">
        <button class="mobile-nav-btn" id="mobile-nav-btn" type="button" aria-label="Toggle navigation" aria-expanded="false">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 5.5h14M3 10h14M3 14.5h14"/></svg>
        </button>
        <a class="brand" href="{docs_prefix}index.html" aria-label="MAI-Gym Documentation home">
          <img src="{root_prefix}assets/mark.svg" width="26" height="26" alt="">
          <span class="brand-name">MAI-<span>Gym</span></span>
          <span class="brand-pill">Docs</span>
        </a>
      </div>

      <button class="search-trigger" id="search-trigger" type="button" aria-label="Search documentation">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="8" cy="8" r="5.2"/><path d="m12 12 3.6 3.6"/></svg>
        <span>Search docs...</span>
        <kbd>⌘K</kbd>
      </button>

      <div class="topbar-right">
        <a class="top-link" href="{root_prefix}index.html">Project Page</a>
        <a class="top-link" href="{PAPER_URL}" target="_blank" rel="noopener noreferrer">Paper</a>
        <a class="top-link" href="{MODEL_URL}" target="_blank" rel="noopener noreferrer">Model</a>
        <a class="repo-chip" href="{REPO_URL}" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
          <span>Tongyi-MAI/MAI-Gym</span>
        </a>
        <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle color theme" title="Switch light / dark theme">
          <svg class="i-sun" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="3.6" fill="currentColor"/><path d="M10 1.8v2.1M10 16.1v2.1M1.8 10h2.1M16.1 10h2.1M4.2 4.2l1.5 1.5M14.3 14.3l1.5 1.5M4.2 15.8l1.5-1.5M14.3 5.7l1.5-1.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
          <svg class="i-moon" viewBox="0 0 20 20" aria-hidden="true"><path d="M16.2 12.6A6.8 6.8 0 0 1 7.4 3.8a6.8 6.8 0 1 0 8.8 8.8z" fill="currentColor"/></svg>
        </button>
      </div>
    </div>
  </header>

  <div class="drawer-backdrop" id="drawer-backdrop" hidden></div>

  <div class="doc-shell">
    <aside class="sidebar" id="sidebar" aria-label="Documentation navigation">
      {"".join(sidebar_sections)}
    </aside>

    <main class="doc-main" id="doc-content">
      <article class="doc-article">
        <header class="page-header">
          <div class="page-eyebrow">{html.escape(page.group)}</div>
          <div class="page-title-row">
            <h1>{html.escape(h1)}</h1>
            <button type="button" class="copy-page-btn" id="copy-page-btn" title="Copy page as Markdown">
              <svg class="i-copy" viewBox="0 0 16 16" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3.5 10.5h-1a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v1" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
              <svg class="i-check" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.2 8.4 6.5 11.7 13 4.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
              <span>Copy page</span>
            </button>
          </div>
          <p class="page-lead">{html.escape(page.description)}</p>
        </header>

        <div class="prose">
          {body_html}
        </div>

        <nav class="pagination" aria-label="Pagination">
          {prev_html}
          {next_html}
        </nav>

        <footer class="doc-footer">
          <div class="doc-footer-links">
            <a href="{html.escape(github_edit_url)}" target="_blank" rel="noopener noreferrer">Edit this page on GitHub &nearr;</a>
            <a href="{html.escape(github_issue_url)}" target="_blank" rel="noopener noreferrer">Raise a docs issue &nearr;</a>
            <a href="{docs_prefix}llms.txt">llms.txt</a>
          </div>
          <p class="doc-footer-meta">MAI-Gym · MIT License · Tongyi-MAI</p>
        </footer>
      </article>
    </main>

    <aside class="toc-rail" aria-label="On this page">
      <div class="toc-inner">
        <p class="toc-title">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M2.5 4h11M2.5 8h7M2.5 12h11"/></svg>
          On this page
        </p>
        {toc_block}
        <div class="toc-actions">
          <a href="#top" class="toc-action-link" id="back-to-top">&uarr; Back to top</a>
          <a href="{html.escape(github_edit_url)}" class="toc-action-link" target="_blank" rel="noopener noreferrer">View Markdown &nearr;</a>
        </div>
      </div>
    </aside>
  </div>

  <dialog class="search-dialog" id="search-dialog" aria-label="Search documentation">
    <div class="search-box">
      <div class="search-input-row">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="5.2"/><path d="m12 12 3.6 3.6"/></svg>
        <input id="search-input" type="search" placeholder="Search documentation, CLI flags, benchmarks..." autocomplete="off" spellcheck="false">
        <button type="button" class="search-esc" id="search-close">ESC</button>
      </div>
      <div class="search-results" id="search-results" role="listbox"></div>
      <div class="search-foot">
        <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> navigate</span>
        <span><kbd>Enter</kbd> open</span>
        <span><kbd>Esc</kbd> close</span>
      </div>
    </div>
  </dialog>

  <dialog class="img-dialog" id="img-dialog" aria-label="Image preview">
    <button type="button" class="img-dialog-close" id="img-dialog-close" aria-label="Close preview">&times;</button>
    <img id="img-dialog-src" alt="">
  </dialog>
</body>
</html>
"""


def validate_site(rendered_bodies: dict[str, str]) -> None:
    """Validate internal page links, anchor targets, and local image assets."""
    page_ids: dict[str, set[str]] = {}
    for slug, body_html in rendered_bodies.items():
        ids = set(re.findall(r'\bid="([^"]+)"', body_html))
        ids.add("top")
        ids.add("doc-content")
        page_ids[slug] = ids

    errors: list[str] = []
    for slug, body_html in rendered_bodies.items():
        page_dir = posixpath.dirname(slug) or "."
        # Check local image assets
        for m in re.finditer(r'<img\b[^>]*\bsrc="([^"]+)"', body_html):
            src = html.unescape(m.group(1))
            if src.startswith(("http://", "https://", "data:")):
                continue
            resolved = (OUT_DIR / page_dir / src).resolve()
            if not resolved.is_file():
                errors.append(f"[{slug}] Missing local image: {src} -> {resolved}")

        # Check internal links & anchors
        for m in re.finditer(r'<a\b[^>]*\bhref="([^"]+)"', body_html):
            href = html.unescape(m.group(1))
            if not href or href.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            if href.startswith("#"):
                frag = href[1:]
                if frag and frag not in page_ids[slug]:
                    errors.append(f"[{slug}] Broken same-page anchor: #{frag}")
                continue
            path_part, _, frag = href.partition("#")
            target_rel = posixpath.normpath(posixpath.join(page_dir, path_part))
            if target_rel in PAGE_BY_SLUG:
                if frag and frag not in page_ids[target_rel]:
                    errors.append(f"[{slug}] Broken anchor in {target_rel}: #{frag}")
                continue
            resolved = (OUT_DIR / page_dir / path_part).resolve()
            if not resolved.exists():
                errors.append(f"[{slug}] Broken local link: {href} -> {resolved}")

    if errors:
        raise RuntimeError("Docs validation failed:\n  " + "\n  ".join(errors))


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = OUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Prune stale generated HTML and raw Markdown copies no longer in PAGES
    expected_html = {p.slug for p in PAGES}
    expected_raw = {p.slug.replace(".html", ".md") for p in PAGES}
    for existing in OUT_DIR.rglob("*.html"):
        rel = existing.relative_to(OUT_DIR).as_posix()
        if rel not in expected_html:
            existing.unlink()
    for existing in raw_dir.rglob("*.md"):
        rel = existing.relative_to(raw_dir).as_posix()
        if rel not in expected_raw:
            existing.unlink()

    search_index: list[dict[str, object]] = []
    rendered_bodies: dict[str, str] = {}
    llms_lines = [
        "# MAI-Gym Documentation",
        "",
        "> An open all-in-one platform for GUI agent evaluation and online RL training across mobile, computer, and web environments.",
        "",
        "## Pages",
        "",
    ]

    for page in PAGES:
        src_path = ROOT / page.source
        if not src_path.is_file():
            raise FileNotFoundError(f"Missing documentation source: {src_path}")
        raw_md = src_path.read_text(encoding="utf-8")

        # Write raw markdown copy for Copy Page & AI agents (unless already in raw/)
        raw_slug = page.slug.replace(".html", ".md")
        raw_dest = raw_dir / raw_slug
        raw_dest.parent.mkdir(parents=True, exist_ok=True)
        if src_path.resolve() != raw_dest.resolve():
            raw_dest.write_text(raw_md, encoding="utf-8")
        raw_rel = rel_href(page.slug, f"raw/{raw_slug}")

        h1, body_html, toc = render_markdown(raw_md, page)
        rendered_bodies[page.slug] = body_html
        page_html = render_page(page, h1, body_html, toc, raw_rel)

        dest = OUT_DIR / page.slug
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(page_html, encoding="utf-8")

        # Search index entry
        plain_body = strip_markdown_plain(raw_md)
        search_index.append(
            {
                "slug": page.slug,
                "title": page.title,
                "group": page.group,
                "description": page.description,
                "headings": [{"id": hid, "text": text} for _, hid, text in toc],
                "content": plain_body[:4000],
            }
        )
        llms_lines.append(f"- [{page.title}](raw/{raw_slug}): {page.description}")

    validate_site(rendered_bodies)

    (OUT_DIR / "search-index.json").write_text(
        json.dumps(search_index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "llms.txt").write_text("\n".join(llms_lines) + "\n", encoding="utf-8")
    print(f"Built & validated {len(PAGES)} docs pages into {OUT_DIR.relative_to(ROOT)}/")


def _data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if path.suffix == ".woff2":
        mime = "font/woff2"
    elif path.suffix == ".svg":
        mime = "image/svg+xml"
    elif not mime:
        mime = "application/octet-stream"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _rewrite_html_for_standalone(body_html: str, page: PageSpec) -> str:
    """Inline local images as data URIs and rewrite internal page links for hash routing."""
    page_dir = posixpath.dirname(page.slug) or "."

    def repl_img_src(m: re.Match[str]) -> str:
        raw_src = m.group(1)
        if raw_src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        resolved = (OUT_DIR / page_dir / raw_src).resolve()
        if resolved.is_file():
            return f'src="{_data_uri(resolved)}"'
        return m.group(0)

    body_html = re.sub(r'src="([^"]+)"', repl_img_src, body_html)

    def repl_href(m: re.Match[str]) -> str:
        raw_href = html.unescape(m.group(1))
        if not raw_href or raw_href.startswith(("http://", "https://", "mailto:", "data:")):
            return m.group(0)
        if raw_href.startswith("#"):
            frag = raw_href[1:]
            return f'href="#{html.escape(page.slug)}:{html.escape(frag)}" data-doc-page="{html.escape(page.slug)}" data-doc-hash="{html.escape(frag)}"'
        path_part, _, frag = raw_href.partition("#")
        target = posixpath.normpath(posixpath.join(page_dir, path_part))
        if target in PAGE_BY_SLUG:
            h = f"#{target}:{frag}" if frag else f"#{target}"
            return (
                f'href="{html.escape(h)}" data-doc-page="{html.escape(target)}" '
                f'data-doc-hash="{html.escape(frag)}"'
            )
        # External fallback for site/ files outside the standalone docs bundle (e.g. demo video, project home)
        resolved = (OUT_DIR / page_dir / path_part).resolve()
        if resolved == (SITE_DIR / "index.html").resolve():
            return 'href="https://tongyi-mai.github.io/MAI-Gym/" target="_blank" rel="noopener noreferrer"'
        try:
            rel_repo = resolved.relative_to(ROOT).as_posix()
            return f'href="{REPO_BLOB_BASE}/{html.escape(rel_repo)}" target="_blank" rel="noopener noreferrer"'
        except ValueError:
            return m.group(0)

    return re.sub(r'href="([^"]+)"', repl_href, body_html)


STANDALONE_JS = r"""(() => {
  const bundle = JSON.parse(document.getElementById("maigym-bundle").textContent);
  const pagesBySlug = Object.fromEntries(bundle.pages.map((p) => [p.slug, p]));
  const navGroups = bundle.navGroups;
  const flatNav = navGroups.flatMap((g) => g.slugs);

  let currentSlug = "index.html";
  let tocTargets = [];

  /* ---- Theme Toggle ---- */
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    themeBtn.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "day" ? "night" : "day";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("maigym.theme", next); } catch (e) {}
    });
  }

  /* ---- Mobile Drawer ---- */
  const navBtn = document.getElementById("mobile-nav-btn");
  const sidebarEl = document.getElementById("sidebar");
  const backdrop = document.getElementById("drawer-backdrop");
  function setDrawer(open) {
    if (!sidebarEl || !backdrop || !navBtn) return;
    sidebarEl.classList.toggle("open", open);
    backdrop.hidden = !open;
    navBtn.setAttribute("aria-expanded", String(open));
  }
  if (navBtn) navBtn.addEventListener("click", () => setDrawer(!sidebarEl.classList.contains("open")));
  if (backdrop) backdrop.addEventListener("click", () => setDrawer(false));

  /* ---- Clipboard helper ---- */
  function flashCopied(btn, labelText) {
    btn.classList.add("copied");
    const span = btn.querySelector("span");
    const prev = span ? span.textContent : "";
    if (span) span.textContent = labelText || "Copied";
    setTimeout(() => {
      btn.classList.remove("copied");
      if (span) span.textContent = prev;
    }, 1800);
  }
  async function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  /* ---- Image Zoom Lightbox ---- */
  const imgDialog = document.getElementById("img-dialog");
  const imgDialogSrc = document.getElementById("img-dialog-src");
  const imgDialogClose = document.getElementById("img-dialog-close");
  if (imgDialogClose) imgDialogClose.addEventListener("click", () => imgDialog.close());
  if (imgDialog) {
    imgDialog.addEventListener("click", (e) => {
      if (e.target === imgDialog) imgDialog.close();
    });
  }

  /* ---- Back to top ---- */
  const backTop = document.getElementById("back-to-top");
  if (backTop) {
    backTop.addEventListener("click", (e) => {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  /* ---- Copy Page Markdown ---- */
  const copyPageBtn = document.getElementById("copy-page-btn");
  if (copyPageBtn) {
    copyPageBtn.addEventListener("click", async () => {
      const page = pagesBySlug[currentSlug];
      if (!page) return;
      try {
        await copyText(page.rawMd || "");
        flashCopied(copyPageBtn, "Copied MD");
      } catch (e) {}
    });
  }

  /* ---- TOC Scroll Spy ---- */
  function updateToc() {
    if (!tocTargets.length) return;
    const scrollY = window.scrollY + 140;
    let current = tocTargets[0];
    for (const item of tocTargets) {
      if (item.el.offsetTop <= scrollY) current = item;
      else break;
    }
    for (const item of tocTargets) item.link.classList.remove("active");
    if (current) current.link.classList.add("active");
  }
  let ticking = false;
  window.addEventListener("scroll", () => {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(() => { updateToc(); ticking = false; });
    }
  }, { passive: true });

  /* ---- Render Page ---- */
  function parseHash() {
    const raw = (window.location.hash || "").replace(/^#/, "");
    if (!raw) return { slug: "index.html", anchor: "" };
    const idx = raw.indexOf(":");
    if (idx !== -1) {
      const s = raw.slice(0, idx);
      const a = raw.slice(idx + 1);
      if (pagesBySlug[s]) return { slug: s, anchor: a };
    }
    if (pagesBySlug[raw]) return { slug: raw, anchor: "" };
    return { slug: currentSlug, anchor: raw };
  }

  function navigateTo(slug, anchor = "", push = true) {
    const targetSlug = pagesBySlug[slug] ? slug : "index.html";
    const targetHash = anchor ? `#${targetSlug}:${anchor}` : `#${targetSlug}`;
    if (push && window.location.hash !== targetHash) {
      window.location.hash = targetHash;
      return;
    }
    renderPage(targetSlug, anchor);
  }

  function renderPage(slug, anchor) {
    const page = pagesBySlug[slug] || pagesBySlug["index.html"];
    currentSlug = page.slug;
    document.title = `${page.title} — MAI-Gym Docs`;
    setDrawer(false);

    // Unified vertical sidebar
    const groupsHtml = navGroups
      .map((g) => {
        const lis = g.slugs
          .map((s) => {
            const sp = pagesBySlug[s];
            return `<li><a class="side-link${s === page.slug ? " active" : ""}" href="#${escapeHtml(s)}" data-doc-page="${escapeHtml(s)}">${escapeHtml(sp.title)}</a></li>`;
          })
          .join("");
        return `<div class="side-group"><p class="side-heading">${escapeHtml(g.name)}</p><ul>${lis}</ul></div>`;
      })
      .join("");
    sidebarEl.innerHTML = groupsHtml;
    const activeSide = sidebarEl.querySelector(".side-link.active");
    if (activeSide) activeSide.scrollIntoView({ block: "nearest" });

    // Header & Content
    document.getElementById("page-eyebrow").textContent = page.group;
    document.getElementById("page-h1").textContent = page.h1;
    document.getElementById("page-lead").textContent = page.description;
    const proseEl = document.getElementById("prose-slot");
    proseEl.innerHTML = page.bodyHtml;

    // Bind copy-code & zoomable images
    proseEl.querySelectorAll("[data-copy-code]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const code = btn.closest(".code-block")?.querySelector("pre code");
        if (!code) return;
        try {
          await copyText(code.textContent || "");
          flashCopied(btn, "Copied");
        } catch (e) {}
      });
    });
    proseEl.querySelectorAll("img[data-zoomable]").forEach((img) => {
      img.addEventListener("click", () => {
        imgDialogSrc.src = img.currentSrc || img.src;
        imgDialogSrc.alt = img.alt || "";
        imgDialog.showModal();
      });
    });

    // Pagination
    const idx = flatNav.indexOf(page.slug);
    let prevHtml = '<span class="pag-spacer"></span>';
    let nextHtml = '<span class="pag-spacer"></span>';
    if (idx > 0) {
      const p = pagesBySlug[flatNav[idx - 1]];
      prevHtml = `<a class="pag-link prev" href="#${escapeHtml(p.slug)}" data-doc-page="${escapeHtml(p.slug)}"><span class="pag-label">Previous</span><span class="pag-title">&larr; ${escapeHtml(p.title)}</span></a>`;
    }
    if (idx !== -1 && idx + 1 < flatNav.length) {
      const n = pagesBySlug[flatNav[idx + 1]];
      nextHtml = `<a class="pag-link next" href="#${escapeHtml(n.slug)}" data-doc-page="${escapeHtml(n.slug)}"><span class="pag-label">Next</span><span class="pag-title">${escapeHtml(n.title)} &rarr;</span></a>`;
    }
    document.getElementById("pagination-slot").innerHTML = prevHtml + nextHtml;

    // Footer & TOC links
    document.getElementById("edit-github-link").href = page.githubEditUrl;
    document.getElementById("toc-view-md").href = page.githubEditUrl;
    document.getElementById("issue-github-link").href = page.githubIssueUrl;

    // Right TOC
    const tocSlot = document.getElementById("toc-slot");
    if (page.toc && page.toc.length) {
      tocSlot.innerHTML = `<ul class="toc-list" id="toc-list">${page.toc
        .map(
          (t) =>
            `<li><a class="toc-link${t.level === 3 ? " toc-h3" : ""}" href="#${escapeHtml(page.slug)}:${escapeHtml(t.id)}" data-doc-page="${escapeHtml(page.slug)}" data-doc-hash="${escapeHtml(t.id)}">${escapeHtml(t.text)}</a></li>`
        )
        .join("")}</ul>`;
      tocTargets = Array.from(tocSlot.querySelectorAll(".toc-link"))
        .map((link) => ({ link, el: document.getElementById(link.dataset.docHash || "") }))
        .filter((x) => x.el);
    } else {
      tocSlot.innerHTML = '<p class="toc-empty">Overview</p>';
      tocTargets = [];
    }

    if (anchor) {
      const el = document.getElementById(anchor);
      if (el) {
        requestAnimationFrame(() => el.scrollIntoView({ behavior: "auto", block: "start" }));
      } else {
        window.scrollTo({ top: 0 });
      }
    } else {
      window.scrollTo({ top: 0 });
    }
    updateToc();
  }

  // Delegated click handler for all internal SPA links
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[data-doc-page]");
    if (!a || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    navigateTo(a.dataset.docPage, a.dataset.docHash || "", true);
  });

  window.addEventListener("hashchange", () => {
    const { slug, anchor } = parseHash();
    renderPage(slug, anchor);
  });

  /* ---- Instant Cmd+K Search ---- */
  const searchTrigger = document.getElementById("search-trigger");
  const searchDialog = document.getElementById("search-dialog");
  const searchInput = document.getElementById("search-input");
  const searchResults = document.getElementById("search-results");
  const searchClose = document.getElementById("search-close");
  let selectedIdx = 0;
  let renderedItems = [];

  function highlightMatch(text, terms) {
    let out = escapeHtml(text);
    for (const t of terms) {
      if (t.length < 2) continue;
      const re = new RegExp(`(${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig");
      out = out.replace(re, "<mark>$1</mark>");
    }
    return out;
  }

  function snippetAround(content, terms) {
    const lower = content.toLowerCase();
    let pos = -1;
    for (const t of terms) {
      const p = lower.indexOf(t);
      if (p !== -1 && (pos === -1 || p < pos)) pos = p;
    }
    if (pos === -1) return content.slice(0, 150);
    const start = Math.max(0, pos - 48);
    const end = Math.min(content.length, pos + 110);
    return (start > 0 ? "..." : "") + content.slice(start, end) + (end < content.length ? "..." : "");
  }

  function runSearch(query) {
    const docs = bundle.searchIndex;
    const q = query.trim().toLowerCase();
    const terms = q.split(/\s+/).filter(Boolean);
    let matches = [];
    if (!terms.length) {
      matches = docs.slice(0, 8).map((d) => ({
        slug: d.slug,
        anchor: "",
        title: d.title,
        group: d.group,
        snippet: d.description,
      }));
    } else {
      for (const d of docs) {
        let score = 0;
        const titleL = d.title.toLowerCase();
        const descL = d.description.toLowerCase();
        const contentL = d.content.toLowerCase();
        let bestHeading = null;
        for (const t of terms) {
          if (titleL.includes(t)) score += 10;
          if (descL.includes(t)) score += 5;
          if (contentL.includes(t)) score += 2;
          for (const h of d.headings || []) {
            if (h.text.toLowerCase().includes(t)) {
              score += 7;
              if (!bestHeading) bestHeading = h;
            }
          }
        }
        if (score > 0) {
          matches.push({
            score,
            slug: d.slug,
            anchor: bestHeading ? bestHeading.id : "",
            title: bestHeading ? `${d.title} › ${bestHeading.text}` : d.title,
            group: d.group,
            snippet: snippetAround(d.content, terms),
          });
        }
      }
      matches.sort((a, b) => b.score - a.score);
      matches = matches.slice(0, 10);
    }

    selectedIdx = 0;
    renderedItems = matches;
    if (!matches.length) {
      searchResults.innerHTML = `<div class="search-empty">No results for "${escapeHtml(query)}"</div>`;
      return;
    }
    searchResults.innerHTML = matches
      .map(
        (m, i) => `
      <a class="search-item${i === 0 ? " selected" : ""}" href="#${escapeHtml(m.slug)}${m.anchor ? ":" + escapeHtml(m.anchor) : ""}" data-search-idx="${i}" role="option">
        <div class="search-item-head">
          <span class="search-item-title">${highlightMatch(m.title, terms)}</span>
          <span class="search-item-group">${escapeHtml(m.group)}</span>
        </div>
        <div class="search-item-snippet">${highlightMatch(m.snippet, terms)}</div>
      </a>`
      )
      .join("");
  }

  searchResults.addEventListener("click", (e) => {
    const item = e.target.closest("[data-search-idx]");
    if (!item) return;
    e.preventDefault();
    const m = renderedItems[Number(item.dataset.searchIdx)];
    if (m) {
      searchDialog.close();
      navigateTo(m.slug, m.anchor, true);
    }
  });

  function openSearch() {
    if (!searchDialog.open) searchDialog.showModal();
    searchInput.value = "";
    searchInput.focus();
    runSearch("");
  }
  if (searchTrigger) searchTrigger.addEventListener("click", openSearch);
  if (searchClose) searchClose.addEventListener("click", () => searchDialog.close());
  if (searchDialog) {
    searchDialog.addEventListener("click", (e) => { if (e.target === searchDialog) searchDialog.close(); });
  }
  if (searchInput) {
    searchInput.addEventListener("input", () => runSearch(searchInput.value));
    searchInput.addEventListener("keydown", (e) => {
      const items = searchResults.querySelectorAll(".search-item");
      if (!items.length) return;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        selectedIdx = (selectedIdx + 1) % items.length;
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        selectedIdx = (selectedIdx - 1 + items.length) % items.length;
      } else if (e.key === "Enter" && renderedItems[selectedIdx]) {
        e.preventDefault();
        const m = renderedItems[selectedIdx];
        searchDialog.close();
        navigateTo(m.slug, m.anchor, true);
        return;
      } else {
        return;
      }
      items.forEach((el, idx) => el.classList.toggle("selected", idx === selectedIdx));
      items[selectedIdx]?.scrollIntoView({ block: "nearest" });
    });
  }

  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      openSearch();
    } else if (
      e.key === "/" &&
      !searchDialog?.open &&
      document.activeElement?.tagName !== "INPUT" &&
      document.activeElement?.tagName !== "TEXTAREA"
    ) {
      e.preventDefault();
      openSearch();
    } else if (e.key === "Escape") {
      setDrawer(false);
    }
  });

  // Initial render
  const initial = parseHash();
  renderPage(initial.slug, initial.anchor);
})();"""


def build_standalone(output_path: Path) -> None:
    """Bundle the entire Mintlify-style docs site into one self-contained HTML file."""
    css = (OUT_DIR / "assets" / "docs.css").read_text(encoding="utf-8")
    inter_uri = _data_uri(SITE_DIR / "assets/fonts/inter-latin-wght-normal.woff2")
    jb_uri = _data_uri(SITE_DIR / "assets/fonts/jetbrains-mono-latin-wght-normal.woff2")
    css = css.replace(
        'url("../../assets/fonts/inter-latin-wght-normal.woff2")', f'url("{inter_uri}")'
    )
    css = css.replace(
        'url("../../assets/fonts/jetbrains-mono-latin-wght-normal.woff2")', f'url("{jb_uri}")'
    )

    favicon_uri = _data_uri(SITE_DIR / "assets/favicon.png")
    mark_uri = _data_uri(SITE_DIR / "assets/mark.svg")

    bundled_pages: list[dict[str, object]] = []
    search_index: list[dict[str, object]] = []

    for page in PAGES:
        src_path = ROOT / page.source
        raw_md = src_path.read_text(encoding="utf-8")
        h1, body_html, toc = render_markdown(raw_md, page)
        body_html = _rewrite_html_for_standalone(body_html, page)
        bundled_pages.append(
            {
                "slug": page.slug,
                "title": page.title,
                "h1": h1,
                "description": page.description,
                "group": page.group,
                "bodyHtml": body_html,
                "rawMd": raw_md,
                "toc": [{"level": lv, "id": hid, "text": txt} for lv, hid, txt in toc],
                "githubEditUrl": f"{REPO_BLOB_BASE}/{page.source}",
                "githubIssueUrl": f"{REPO_URL}/issues/new?title=Docs:%20{page.title}",
            }
        )
        search_index.append(
            {
                "slug": page.slug,
                "title": page.title,
                "group": page.group,
                "description": page.description,
                "headings": [{"id": hid, "text": text} for _, hid, text in toc],
                "content": strip_markdown_plain(raw_md)[:4000],
            }
        )

    bundle_data = {
        "navGroups": [{"name": gname, "slugs": slugs} for gname, slugs in NAV_GROUPS],
        "pages": bundled_pages,
        "searchIndex": search_index,
    }
    bundle_json = json.dumps(bundle_data, ensure_ascii=False).replace("</", "<\\/")

    standalone_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="MAI-Gym Documentation (Standalone Offline Bundle)">
  <title>Quickstart — MAI-Gym Docs</title>
  <link rel="icon" type="image/png" href="{favicon_uri}">
  <script>
    (function () {{
      var t;
      try {{ t = localStorage.getItem('maigym.theme'); }} catch (e) {{}}
      document.documentElement.dataset.theme = t === 'day' ? 'day' : 'night';
    }})();
  </script>
  <style>{css}</style>
</head>
<body>
  <a class="skip-link" href="#doc-content">Skip to main content</a>

  <header class="topbar" id="topbar">
    <div class="topbar-row">
      <div class="topbar-left">
        <button class="mobile-nav-btn" id="mobile-nav-btn" type="button" aria-label="Toggle navigation" aria-expanded="false">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M3 5.5h14M3 10h14M3 14.5h14"/></svg>
        </button>
        <a class="brand" href="#index.html" data-doc-page="index.html" aria-label="MAI-Gym Documentation home">
          <img src="{mark_uri}" width="26" height="26" alt="">
          <span class="brand-name">MAI-<span>Gym</span></span>
          <span class="brand-pill">Docs</span>
        </a>
      </div>

      <button class="search-trigger" id="search-trigger" type="button" aria-label="Search documentation">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="8" cy="8" r="5.2"/><path d="m12 12 3.6 3.6"/></svg>
        <span>Search docs...</span>
        <kbd>⌘K</kbd>
      </button>

      <div class="topbar-right">
        <a class="top-link" href="https://tongyi-mai.github.io/MAI-Gym/" target="_blank" rel="noopener noreferrer">Project Page</a>
        <a class="top-link" href="{PAPER_URL}" target="_blank" rel="noopener noreferrer">Paper</a>
        <a class="top-link" href="{MODEL_URL}" target="_blank" rel="noopener noreferrer">Model</a>
        <a class="repo-chip" href="{REPO_URL}" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>
          <span>Tongyi-MAI/MAI-Gym</span>
        </a>
        <button class="theme-toggle" id="theme-toggle" type="button" aria-label="Toggle color theme" title="Switch light / dark theme">
          <svg class="i-sun" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="3.6" fill="currentColor"/><path d="M10 1.8v2.1M10 16.1v2.1M1.8 10h2.1M16.1 10h2.1M4.2 4.2l1.5 1.5M14.3 14.3l1.5 1.5M4.2 15.8l1.5-1.5M14.3 5.7l1.5-1.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>
          <svg class="i-moon" viewBox="0 0 20 20" aria-hidden="true"><path d="M16.2 12.6A6.8 6.8 0 0 1 7.4 3.8a6.8 6.8 0 1 0 8.8 8.8z" fill="currentColor"/></svg>
        </button>
      </div>
    </div>
  </header>

  <div class="drawer-backdrop" id="drawer-backdrop" hidden></div>

  <div class="doc-shell">
    <aside class="sidebar" id="sidebar" aria-label="Documentation navigation"></aside>

    <main class="doc-main" id="doc-content">
      <article class="doc-article">
        <header class="page-header">
          <div class="page-eyebrow" id="page-eyebrow"></div>
          <div class="page-title-row">
            <h1 id="page-h1"></h1>
            <button type="button" class="copy-page-btn" id="copy-page-btn" title="Copy page as Markdown">
              <svg class="i-copy" viewBox="0 0 16 16" aria-hidden="true"><rect x="5.5" y="5.5" width="8" height="8" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M3.5 10.5h-1a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h7a1 1 0 0 1 1 1v1" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round"/></svg>
              <svg class="i-check" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.2 8.4 6.5 11.7 13 4.8" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
              <span>Copy page</span>
            </button>
          </div>
          <p class="page-lead" id="page-lead"></p>
        </header>

        <div class="prose" id="prose-slot"></div>

        <nav class="pagination" id="pagination-slot" aria-label="Pagination"></nav>

        <footer class="doc-footer">
          <div class="doc-footer-links">
            <a id="edit-github-link" href="{REPO_URL}" target="_blank" rel="noopener noreferrer">Edit this page on GitHub &nearr;</a>
            <a id="issue-github-link" href="{REPO_URL}/issues/new" target="_blank" rel="noopener noreferrer">Raise a docs issue &nearr;</a>
          </div>
          <p class="doc-footer-meta">MAI-Gym · MIT License · Standalone Review Bundle</p>
        </footer>
      </article>
    </main>

    <aside class="toc-rail" aria-label="On this page">
      <div class="toc-inner">
        <p class="toc-title">
          <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M2.5 4h11M2.5 8h7M2.5 12h11"/></svg>
          On this page
        </p>
        <div id="toc-slot"></div>
        <div class="toc-actions">
          <a href="#top" class="toc-action-link" id="back-to-top">&uarr; Back to top</a>
          <a id="toc-view-md" href="{REPO_URL}" class="toc-action-link" target="_blank" rel="noopener noreferrer">View Markdown &nearr;</a>
        </div>
      </div>
    </aside>
  </div>

  <dialog class="search-dialog" id="search-dialog" aria-label="Search documentation">
    <div class="search-box">
      <div class="search-input-row">
        <svg viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="5.2"/><path d="m12 12 3.6 3.6"/></svg>
        <input id="search-input" type="search" placeholder="Search documentation, CLI flags, benchmarks..." autocomplete="off" spellcheck="false">
        <button type="button" class="search-esc" id="search-close">ESC</button>
      </div>
      <div class="search-results" id="search-results" role="listbox"></div>
      <div class="search-foot">
        <span><kbd>&uarr;</kbd><kbd>&darr;</kbd> navigate</span>
        <span><kbd>Enter</kbd> open</span>
        <span><kbd>Esc</kbd> close</span>
      </div>
    </div>
  </dialog>

  <dialog class="img-dialog" id="img-dialog" aria-label="Image preview">
    <button type="button" class="img-dialog-close" id="img-dialog-close" aria-label="Close preview">&times;</button>
    <img id="img-dialog-src" alt="">
  </dialog>

  <script id="maigym-bundle" type="application/json">{bundle_json}</script>
  <script>{STANDALONE_JS}</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(standalone_html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(
        f"Built standalone single-file docs ({len(PAGES)} pages, {size_kb:.1f} KB) -> {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MAI-Gym static documentation.")
    parser.add_argument(
        "--standalone",
        type=Path,
        default=None,
        help="Also compile a single self-contained offline HTML file to this path.",
    )
    args = parser.parse_args()
    build()
    if args.standalone is not None:
        build_standalone(args.standalone)


if __name__ == "__main__":
    main()
