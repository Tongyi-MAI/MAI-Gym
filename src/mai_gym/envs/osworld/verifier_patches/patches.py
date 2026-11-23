from __future__ import annotations

import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections.abc import Callable
from copy import copy
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import (
    drop_bash_history_metrics,
    set_all_options,
)

CONFIG_PATCHES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


METRIC_OVERRIDES: dict[str, dict[str, Callable[..., Any]]] = {}


PATCH_META: dict[str, dict[str, Any]] = {}


def _empty_url_path(url: str) -> bool:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    return parsed.path in ("", "/") and not parsed.params


def _host_key(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        import tldextract

        extracted = tldextract.extract(host)
        return (
            f"{extracted.subdomain}.{extracted.domain}" if extracted.subdomain else extracted.domain
        )
    except Exception:
        parts = [p for p in host.split(".") if p]
        if len(parts) >= 3 and parts[-2] in {"com", "co", "org", "net"}:
            domain_parts = parts[:-2]
        elif len(parts) >= 2:
            domain_parts = parts[:-1]
        else:
            domain_parts = parts
        return ".".join(domain_parts)


def _normalized_path(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    path = parsed.path or ""
    return "" if path == "/" else path.rstrip("/")


def _urls_match_for_tab(expected: str, actual: str) -> bool:
    if _host_key(expected) != _host_key(actual):
        return False
    if _empty_url_path(expected):
        return True
    return _normalized_path(expected) == _normalized_path(actual)


def _is_expected_tabs_allow_redirect_query(
    open_tabs: list[dict[str, str]],
    rule: dict[str, Any],
) -> float:

    if not open_tabs or rule.get("type") != "url":
        return 0.0

    expected_urls = list(rule.get("urls") or [])
    actual_urls = [tab.get("url") for tab in open_tabs if tab.get("url")]
    if len(expected_urls) != len(actual_urls):
        return 0.0

    remaining = actual_urls[:]
    for expected in expected_urls:
        for idx, actual in enumerate(remaining):
            if _urls_match_for_tab(expected, actual):
                remaining.pop(idx)
                break
        else:
            return 0.0
    return 1.0


def _is_expected_tabs_e135df7c(
    open_tabs: list[dict[str, str]],
    rule: dict[str, Any],
) -> float:

    if not open_tabs or rule.get("type") != "url":
        return 0.0
    expected_urls = list(rule.get("urls") or [])
    actual_urls = [tab.get("url") for tab in open_tabs if tab.get("url")]
    if len(expected_urls) != len(actual_urls):
        return 0.0

    remaining = actual_urls[:]
    for expected in expected_urls:
        expected_parsed = urlparse(expected)
        for idx, actual in enumerate(remaining):
            if expected_parsed.scheme == "file":
                if actual == expected:
                    remaining.pop(idx)
                    break
                continue
            if _host_key(expected) == _host_key(actual):
                remaining.pop(idx)
                break
        else:
            return 0.0
    return 1.0


def _is_expected_url_pattern_match_f3b19d1e(
    active_tab_info: dict[str, Any] | None,
    rule: dict[str, Any],
) -> float:

    url = (active_tab_info or {}).get("url") or ""
    title = (active_tab_info or {}).get("title") or ""
    haystack = f"{url} {title}"
    for pattern in rule.get("expected") or []:
        if re.search(pattern, haystack):
            return 1.0
    if "help.ticketek.com.au" in url and "Ticket-Delivery" in url and "Ticket Delivery" in title:
        return 1.0
    return 0.0


_CJK_ADDRESS_STRIP_RE = re.compile(r"[\s,，。；;:：、.()（）\\-—_]+")
_CJK_ADDRESS_PREFIX_RE = re.compile(r"^\d+")


def _normalize_cjk_address(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _CJK_ADDRESS_PREFIX_RE.sub("", text)
    return _CJK_ADDRESS_STRIP_RE.sub("", text)


def _fuzzy_place_math_7ff48d5b(result_file: str, expected: dict[str, Any]) -> float:

    if not result_file:
        return 0.0
    try:
        import docx

        document = docx.Document(result_file)
    except Exception:
        return 0.0

    paragraphs = [
        _normalize_cjk_address(p.text) for p in document.paragraphs if p.text and p.text.strip()
    ]
    candidates = [
        _normalize_cjk_address(x) for x in (expected.get("expected") or []) if isinstance(x, str)
    ]
    matched: set[str] = set()
    for paragraph in paragraphs:
        for candidate in candidates:
            if candidate and candidate in paragraph:
                matched.add(candidate)
                break
    return 1.0 if len(matched) >= 5 else 0.0


_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
    }
)

_SIMPLE_QUOTED_SHEET_REF_RE = re.compile(r"'([A-Za-z_][A-Za-z0-9_]*)'!")


def _normalize_pptx_quotes(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    from pptx import Presentation

    prs = Presentation(dst)
    changed = False
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    normalized = run.text.translate(_QUOTE_TRANSLATION)
                    if normalized != run.text:
                        run.text = normalized
                        changed = True
    if changed:
        prs.save(dst)


def _compare_pptx_files_normalized_quotes(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:

    try:
        from desktop_env.evaluators.metrics.slides import compare_pptx_files

        with tempfile.TemporaryDirectory(prefix="mai_gym_pptx_quotes_") as tmp:
            left = str(Path(tmp) / "result.pptx")
            right = str(Path(tmp) / "expected.pptx")
            _normalize_pptx_quotes(file1_path, left)
            _normalize_pptx_quotes(file2_path, right)
            return float(compare_pptx_files(left, right, **options))
    except Exception:
        return 0.0


_DBLP_VOLATILE_RE = re.compile(r"^\s*timestamp\s*=\s*\{.*?\},?\s*$", re.IGNORECASE)


def _normalize_bibtex_for_compare(text: str, *, ignore_blanks: bool) -> str:
    lines = [line for line in text.splitlines() if not _DBLP_VOLATILE_RE.match(line)]
    text = "\n".join(lines).strip()
    if ignore_blanks:
        text = re.sub(r"[\t\n]", " ", text).strip()
        text = re.sub(r"\s+", " ", text)
    return text


def _compare_text_file_ignore_dblp_timestamp(
    actual: str,
    expected: str,
    **options: Any,
) -> float:

    if not actual:
        return 0.0
    try:
        actual_text = Path(actual).read_text()
        expected_text = Path(expected).read_text()
    except Exception:
        return 0.0

    ignore_blanks = bool(options.get("ignore_blanks", False))
    actual_text = _normalize_bibtex_for_compare(
        actual_text,
        ignore_blanks=ignore_blanks,
    )
    expected_text = _normalize_bibtex_for_compare(
        expected_text,
        ignore_blanks=ignore_blanks,
    )
    if options.get("ignore_case", False):
        actual_text = actual_text.lower()
        expected_text = expected_text.lower()
    return 1.0 if actual_text == expected_text else 0.0


def _normalize_simple_quoted_sheet_refs_in_xlsx(src: str, dst: str) -> None:

    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith(".xml"):
                text = data.decode("utf-8", errors="ignore")
                normalized = _SIMPLE_QUOTED_SHEET_REF_RE.sub(r"\1!", text)
                if normalized != text:
                    data = normalized.encode("utf-8")
            zout.writestr(item, data)


def _compare_table_normalize_chart_refs(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:

    try:
        from desktop_env.evaluators.metrics.table import compare_table

        with tempfile.TemporaryDirectory(prefix="mai_gym_chart_refs_") as tmp:
            left = str(Path(tmp) / "result.xlsx")
            right = str(Path(tmp) / "expected.xlsx")
            _normalize_simple_quoted_sheet_refs_in_xlsx(file1_path, left)
            _normalize_simple_quoted_sheet_refs_in_xlsx(file2_path, right)
            return float(compare_table(left, right, **options))
    except Exception:
        return 0.0


def _patch_2b9493d7_drop_history_probe(ev: dict[str, Any]) -> dict[str, Any]:

    ev["result"] = {
        "type": "vm_command_line",
        "command": [
            "/bin/bash",
            "-c",
            (
                'output=$(ps aux | grep "[s]office"); '
                'if [ -z "$output" ]; then '
                'echo "no libreoffice is running"; '
                'else echo "libreoffice is still running"; fi'
            ),
        ],
    }
    ev["expected"] = {
        "type": "rule",
        "rules": {
            "include": ["no libreoffice is running"],
            "exclude": ["libreoffice is still running"],
        },
    }
    return ev


def _patch_aa4b5023_desktop_result_path(ev: dict[str, Any]) -> dict[str, Any]:

    result = ev.get("result")
    if isinstance(result, dict) and result.get("type") == "vm_file":
        result["path"] = "/home/user/Desktop/1984_Apple_Macintosh_Commercial.mp4"
    return ev


def _patch_4ed5abd0_ignore_textbox_geometry(ev: dict[str, Any]) -> dict[str, Any]:

    return set_all_options(ev, examine_shape=False)


def _patch_59f21cfb_vlc_process_probe(ev: dict[str, Any]) -> dict[str, Any]:

    ev["result"] = {
        "type": "vm_command_line",
        "command": [
            "/bin/bash",
            "-lc",
            (
                "echo __PGREP__; "
                "pgrep -a vlc || true; "
                "echo __LSOF__; "
                "lsof -Fn -c vlc 2>/dev/null || true; "
                "echo __WMCTRL__; "
                "wmctrl -l 2>/dev/null | grep -i vlc || true"
            ),
        ],
    }
    return ev


def _normalize_xlsx_red_alpha(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    import openpyxl

    workbook = openpyxl.load_workbook(dst)
    changed = False
    for worksheet in workbook.worksheets:
        for row in worksheet.iter_rows():
            for cell in row:
                fill = cell.fill
                if not fill:
                    continue
                new_fill = copy(fill)
                cell_changed = False
                for color in (new_fill.fgColor, new_fill.bgColor):
                    rgb = getattr(color, "rgb", None)
                    if isinstance(rgb, str) and len(rgb) == 8 and rgb.startswith("00"):
                        color.rgb = "FF" + rgb[-6:]
                        cell_changed = True
                        changed = True
                if cell_changed:
                    cell.fill = new_fill
    if changed:
        workbook.save(dst)


def _compare_table_normalize_red_alpha(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:

    try:
        from desktop_env.evaluators.metrics.table import compare_table

        with tempfile.TemporaryDirectory(prefix="mai_gym_xlsx_red_alpha_") as tmp:
            left = str(Path(tmp) / "result.xlsx")
            _normalize_xlsx_red_alpha(file1_path, left)
            return float(compare_table(left, file2_path, **options))
    except Exception:
        return 0.0


def _normalize_authors_email_header(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    import openpyxl

    workbook = openpyxl.load_workbook(dst)
    worksheet = workbook.worksheets[0]
    value = worksheet["B1"].value
    if isinstance(value, str) and value.strip().lower() == "email":
        worksheet["B1"] = "Mail"
        workbook.save(dst)


def _compare_table_b5062e3e_email_header_alias(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:

    try:
        from desktop_env.evaluators.metrics.table import compare_table

        with tempfile.TemporaryDirectory(prefix="mai_gym_authors_header_") as tmp:
            left = str(Path(tmp) / "authors.xlsx")
            _normalize_authors_email_header(file1_path, left)
            return float(compare_table(left, file2_path, **options))
    except Exception:
        return 0.0


def _normalize_authors_workbook(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    import openpyxl

    workbook = openpyxl.load_workbook(dst)
    worksheet = workbook.worksheets[0]
    if worksheet.title != "Sheet1":
        if "Sheet1" in workbook.sheetnames:
            workbook.remove(workbook["Sheet1"])
        worksheet.title = "Sheet1"

    value = worksheet["B1"].value
    if isinstance(value, str) and value.strip().lower() in {
        "email",
        "e-mail",
        "mail",
    }:
        worksheet["B1"] = "Mail"
    workbook.save(dst)


def _compare_table_b5062e3e_authors_alias(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:

    try:
        from desktop_env.evaluators.metrics.table import compare_table

        with tempfile.TemporaryDirectory(prefix="mai_gym_authors_b5062e3e_") as tmp:
            left = str(Path(tmp) / "authors.xlsx")
            _normalize_authors_workbook(file1_path, left)
            return float(compare_table(left, file2_path, **options))
    except Exception:
        return 0.0


def _compare_config_line_membership(actual: str, rules: dict[str, Any], **options: Any) -> float:

    if not actual:
        return 0.0
    expected = str(rules.get("expected", "")).strip()
    if not expected:
        return 0.0
    try:
        actual_text = Path(actual).read_text(encoding="utf-8")
    except Exception:
        return 0.0
    if actual_text.strip() == expected:
        return 1.0
    reported = [line.strip() for line in actual_text.splitlines() if line.strip()]
    return 1.0 if expected in reported else 0.0


def _looks_like_existing_file(actual: str | None) -> bool:

    if not actual:
        return False
    text = str(actual)
    if "\n" in text or len(text) > 4096:
        return False
    try:
        return Path(text).exists()
    except (OSError, ValueError):
        return False


def _is_vlc_playing_59f21cfb(actual: str | None, rule: dict[str, Any]) -> float:

    expected = str(rule.get("file_name", "")).strip()
    if not expected:
        return 0.0

    if _looks_like_existing_file(actual):
        try:
            from desktop_env.evaluators.metrics.vlc import is_vlc_playing

            if float(is_vlc_playing(str(actual), rule)) == 1.0:
                return 1.0
        except Exception:
            pass
        try:
            actual_text = Path(str(actual)).read_text(errors="ignore")
        except Exception:
            actual_text = ""
    else:
        actual_text = actual or ""

    expected_path = f"/home/user/Desktop/{expected}"
    lines = [line.strip() for line in actual_text.splitlines() if line.strip()]
    process_lines = [line for line in lines if "vlc" in line.lower() or line.startswith("n/")]
    if not process_lines:
        return 0.0
    evidence = "\n".join(process_lines)
    return 1.0 if expected_path in evidence or expected in evidence else 0.0


def _html_visible_lines_from_bytes(data: bytes) -> list[str]:
    try:
        import lxml.html

        root = lxml.html.fromstring(data)
        body = root.find("body")
        root = body if body is not None else root
        nodes = root.xpath(".//*[self::p or self::h1 or self::h2 or self::h3 or self::li]")
        lines = [" ".join(node.text_content().split()) for node in nodes]
        lines = [line for line in lines if line]
        if lines:
            return lines
        text = " ".join(root.text_content().split())
        return [text] if text else []
    except Exception:
        text = " ".join(data.decode(errors="ignore").split())
        return [text] if text else []


def _epub_html_by_basename(path: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        for name in archive.namelist():
            if not name.lower().endswith((".html", ".xhtml")):
                continue
            basename = Path(name).name
            lines = _html_visible_lines_from_bytes(archive.read(name))
            if lines:
                out[basename] = lines
    return out


def _compare_epub_content_level(result: str, expected: str, **options: Any) -> float:

    if not result:
        return 0.0
    try:
        from desktop_env.evaluators.metrics.others import compare_epub

        original = float(compare_epub(result, expected))
    except Exception:
        original = 0.0

    try:
        result_html = _epub_html_by_basename(result)
        expected_html = _epub_html_by_basename(expected)
        common = sorted(set(result_html) & set(expected_html))
        body_common = [
            name for name in common if name.lower() not in {"cover.html", "cover.xhtml"}
        ] or common
        if not body_common:
            return original
        scores = [
            SequenceMatcher(None, result_html[name], expected_html[name]).ratio()
            for name in body_common
        ]
        content_score = sum(scores) / len(scores)
        if content_score >= 0.995 and min(scores) >= 0.99:
            return 1.0
        return max(original, content_score)
    except Exception:
        return original


A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P_NS = "{http://schemas.openxmlformats.org/presentationml/2006/main}"


def _paragraph_text_xml(paragraph: Any) -> str:
    return "".join(t.text or "" for t in paragraph.iter(f"{A_NS}t"))


def _trim_trailing_empty_paragraphs_in_slide_xml(xml_bytes: bytes) -> tuple[bytes, int]:
    from xml.etree import ElementTree as ET

    root = ET.fromstring(xml_bytes)
    removed = 0
    for tx_body in root.iter(f"{P_NS}txBody"):
        paragraphs = [child for child in list(tx_body) if child.tag == f"{A_NS}p"]
        while len(paragraphs) > 1 and _paragraph_text_xml(paragraphs[-1]).strip() == "":
            tx_body.remove(paragraphs.pop())
            removed += 1
    if removed == 0:
        return xml_bytes, 0
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), removed


def _copy_pptx_trim_trailing_empty_paragraphs(src: str, dst: str) -> int:
    removed_total = 0
    with zipfile.ZipFile(src, "r") as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
                data, removed = _trim_trailing_empty_paragraphs_in_slide_xml(data)
                removed_total += removed
            zout.writestr(item, data)
    return removed_total


def _compare_pptx_trim_trailing_empty_paragraphs(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:
    try:
        from desktop_env.evaluators.metrics.slides import compare_pptx_files

        with tempfile.TemporaryDirectory(prefix="mai_gym_pptx_trim_empty_") as tmp:
            left = str(Path(tmp) / "result.pptx")
            right = str(Path(tmp) / "expected.pptx")
            _copy_pptx_trim_trailing_empty_paragraphs(file1_path, left)
            _copy_pptx_trim_trailing_empty_paragraphs(file2_path, right)
            return float(compare_pptx_files(left, right, **options))
    except Exception:
        return 0.0


def _normalize_empty_paragraph_alignment(src: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    from pptx import Presentation

    prs = Presentation(dst)
    changed = False
    for slide in prs.slides:
        for shape in slide.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            for paragraph in shape.text_frame.paragraphs:
                if (paragraph.text or "").strip() == "" and paragraph.alignment is not None:
                    paragraph.alignment = None
                    changed = True
    if changed:
        prs.save(dst)


def _compare_pptx_skip_empty_alignment(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:
    try:
        from desktop_env.evaluators.metrics.slides import compare_pptx_files

        with tempfile.TemporaryDirectory(prefix="mai_gym_pptx_empty_align_") as tmp:
            left = str(Path(tmp) / "result.pptx")
            right = str(Path(tmp) / "expected.pptx")
            _normalize_empty_paragraph_alignment(file1_path, left)
            _normalize_empty_paragraph_alignment(file2_path, right)
            return float(compare_pptx_files(left, right, **options))
    except Exception:
        return 0.0


def _normalize_text_shape_heights_to_expected(src: str, expected: str, dst: str) -> None:
    shutil.copyfile(src, dst)
    from pptx import Presentation

    result_prs = Presentation(dst)
    expected_prs = Presentation(expected)
    changed = False
    for result_slide, expected_slide in zip(result_prs.slides, expected_prs.slides, strict=False):
        for result_shape, expected_shape in zip(
            result_slide.shapes, expected_slide.shapes, strict=False
        ):
            if (
                getattr(result_shape, "has_text_frame", False)
                and getattr(expected_shape, "has_text_frame", False)
                and result_shape.height != expected_shape.height
            ):
                result_shape.height = expected_shape.height
                changed = True
    if changed:
        result_prs.save(dst)


def _compare_pptx_text_height_tolerant(
    file1_path: str,
    file2_path: str,
    **options: Any,
) -> float:
    if options.get("examine_modify_height", False):
        try:
            from desktop_env.evaluators.metrics.slides import compare_pptx_files

            return float(compare_pptx_files(file1_path, file2_path, **options))
        except Exception:
            return 0.0
    try:
        from desktop_env.evaluators.metrics.slides import compare_pptx_files

        with tempfile.TemporaryDirectory(prefix="mai_gym_pptx_text_height_") as tmp:
            left = str(Path(tmp) / "result.pptx")
            _normalize_text_shape_heights_to_expected(file1_path, file2_path, left)
            return float(compare_pptx_files(left, file2_path, **options))
    except Exception:
        return 0.0


METRIC_OVERRIDES["06fe7178-4491-4589-810f-2e2bc9502122"] = {
    "is_expected_tabs": _is_expected_tabs_allow_redirect_query,
}
PATCH_META["06fe7178-4491-4589-810f-2e2bc9502122"] = {
    "domain": "chrome",
    "category": "stale_expectation",
    "status": "active",
    "rationale": "Tab-restore task: the expected URLs are bare origins, while "
    "Chrome may restore/preload redirected URLs with query strings. Keep exact "
    "tab count and one-to-one domain/path matching, but ignore query/fragment "
    "and compare origin-only when the expected URL is a bare domain.",
    "osworld_ref": "metrics/chrome.py:is_expected_tabs + metrics/utils.py:compare_urls(full=True)",
}

METRIC_OVERRIDES["0a211154-fda0-48d0-9274-eaac4ce5486d"] = {
    "compare_pptx_files": _compare_pptx_files_normalized_quotes,
}
PATCH_META["0a211154-fda0-48d0-9274-eaac4ce5486d"] = {
    "domain": "libreoffice_impress",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": 'Instruction asks for ASCII "Let\'s start" but the gold PPTX '
    "contains the typographic apostrophe variant. Normalize equivalent quote "
    "glyphs before delegating to the original PPTX comparator.",
    "osworld_ref": "metrics/slides.py:compare_pptx_files text comparison",
}

METRIC_OVERRIDES["12382c62-0cd1-4bf2-bdc8-1d20bf9b2371"] = {
    "compare_table": _compare_table_normalize_chart_refs,
}
PATCH_META["12382c62-0cd1-4bf2-bdc8-1d20bf9b2371"] = {
    "domain": "libreoffice_calc",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The clustered column chart type/title/data ranges match the "
    "gold, but openpyxl/LibreOffice may serialize simple sheet references as "
    "'Sheet1'!A1 while the gold uses Sheet1!A1. Normalize only redundant quotes "
    "around simple sheet identifiers before delegating to compare_table.",
    "osworld_ref": "metrics/table.py chart branch + utils.load_charts key compare",
}

METRIC_OVERRIDES["df67aebb-fb3a-44fd-b75b-51b6012df509"] = {
    "compare_text_file": _compare_text_file_ignore_dblp_timestamp,
}
PATCH_META["df67aebb-fb3a-44fd-b75b-51b6012df509"] = {
    "domain": "multi_apps",
    "category": "stale_expectation",
    "status": "active",
    "rationale": "The references.bib entries match the required published DBLP "
    "records except for DBLP's volatile timestamp metadata. Strip timestamp "
    "lines and preserve the original whitespace/case options for all other "
    "content.",
    "osworld_ref": "metrics/vscode.py:compare_text_file",
}

METRIC_OVERRIDES["7ff48d5b-2df2-49da-b500-a5150ffc7f18"] = {
    "fuzzy_place_math": _fuzzy_place_math_7ff48d5b,
}
PATCH_META["7ff48d5b-2df2-49da-b500-a5150ffc7f18"] = {
    "domain": "multi_apps",
    "category": "metric_bug",
    "status": "active",
    "rationale": "The instruction asks for five Futian self-service endorsement "
    "machine addresses in Chinese, and the result document contains five "
    "valid expected addresses. Upstream fuzzy_place_math hard-gates on exactly "
    "three whitespace tokens, which contradicts the task and breaks on CJK "
    "addresses. Match address paragraphs against the expected candidate list.",
    "osworld_ref": "metrics/general.py:fuzzy_place_math len(... ) != 3 gate",
}

METRIC_OVERRIDES["e135df7c-7687-4ac0-a5f0-76b74438b53e"] = {
    "is_expected_tabs": _is_expected_tabs_e135df7c,
}
PATCH_META["e135df7c-7687-4ac0-a5f0-76b74438b53e"] = {
    "domain": "multi_apps",
    "category": "stale_expectation",
    "status": "active",
    "rationale": "The converted HTML file is checked separately by "
    "compare_htmls. is_expected_tabs also requires three pre-opened live-site "
    "tabs to keep exact original URLs; LinkedIn can redirect to login before "
    "evaluation. Require the local HTML tab exactly and compare pre-opened web "
    "tabs by host.",
    "osworld_ref": "metrics/chrome.py:is_expected_tabs + live-site redirects",
}

METRIC_OVERRIDES["f3b19d1e-2d48-44e9-b4e1-defcae1a0197"] = {
    "is_expected_url_pattern_match": _is_expected_url_pattern_match_f3b19d1e,
}
PATCH_META["f3b19d1e-2d48-44e9-b4e1-defcae1a0197"] = {
    "domain": "chrome",
    "category": "stale_expectation",
    "status": "active",
    "rationale": "The final page is Ticketek's official help-center category "
    "'Ticket Delivery and Ticket Sharing'. Upstream expects a brittle legacy "
    "URL substring 'Ticket-Delivery-FAQs'. Accept the live official "
    "Ticket-Delivery category.",
    "osworld_ref": "metrics/chrome.py:is_expected_url_pattern_match",
}

CONFIG_PATCHES["2b9493d7-49b8-493a-a71b-56cd1f4d6908"] = _patch_2b9493d7_drop_history_probe
PATCH_META["2b9493d7-49b8-493a-a71b-56cd1f4d6908"] = {
    "domain": "multi_apps",
    "category": "bash_history",
    "status": "active",
    "rationale": "The real success criterion is that no LibreOffice process is "
    "running. The rollout used computer.bash(), which does not write the "
    "interactive terminal's ~/.bash_history. Keep the ps-based live process "
    "check and remove only the provenance proxy.",
    "osworld_ref": "metrics/general.py:check_include_exclude fused ps/history cmd",
}

METRIC_OVERRIDES["8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14"] = {
    "compare_table": _compare_table_normalize_red_alpha,
}
PATCH_META["8b1ce5f2-59d2-4dcc-b0b0-666a714b9a14"] = {
    "domain": "libreoffice_calc",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The instruction asks for red #ff0000 weekend cells. The "
    "result marks exactly the weekend cells red, but openpyxl serializes the "
    "static fill as 00FF0000 while the gold conditional-format style is "
    "FFFF0000. Normalize equivalent ARGB alpha bytes before compare_table.",
    "osworld_ref": "metrics/table.py style/bgcolor exact ARGB compare",
}

CONFIG_PATCHES["aa4b5023-aef6-4ed9-bdc9-705f59ab9ad6"] = _patch_aa4b5023_desktop_result_path
PATCH_META["aa4b5023-aef6-4ed9-bdc9-705f59ab9ad6"] = {
    "domain": "vlc",
    "category": "path_mismatch",
    "status": "active",
    "rationale": "The instruction asks to save the flipped video on the main "
    "screen/Desktop, and the source file is on the Desktop. The evaluator reads "
    "from the home directory. Fetch the result from the Desktop path.",
    "osworld_ref": "getters/vm_file result.path for compare_videos",
}

METRIC_OVERRIDES["42d25c08-fb87-4927-8b65-93631280a26f"] = {
    "compare_epub": _compare_epub_content_level,
}
PATCH_META["42d25c08-fb87-4927-8b65-93631280a26f"] = {
    "domain": "multi_apps",
    "category": "path_packaging",
    "status": "active",
    "rationale": "The task grades the readable EPUB content. Upstream "
    "compare_epub aligns extracted files by sorted position and compares raw "
    "HTML lines, so optional cover files and empty paragraph serialization can "
    "drag a correct EPUB down to a low score. Fall back to same-named HTML "
    "chapter visible-text similarity, ignoring empty paragraphs while keeping "
    "chapter order/content constraints.",
    "osworld_ref": "metrics/others.py:compare_epub sorted zip/raw HTML compare",
}

CONFIG_PATCHES["4ed5abd0-8b5d-47bd-839f-cacfa15ca37a"] = _patch_4ed5abd0_ignore_textbox_geometry
PATCH_META["4ed5abd0-8b5d-47bd-839f-cacfa15ca37a"] = {
    "domain": "libreoffice_impress",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The task grades title text color and underline on slides 2, 3, "
    "and 5. The result text runs match the gold formatting; LibreOffice GUI "
    "editing only auto-fit changed textbox heights. Disable shape-geometry "
    "checking for this task while retaining text/font checks.",
    "osworld_ref": "metrics/slides.py:compare_pptx_files examine_shape",
}

METRIC_OVERRIDES["53ad5833-3455-407b-bbc6-45b4c79ab8fb"] = {
    "compare_config": _compare_config_line_membership,
}
PATCH_META["53ad5833-3455-407b-bbc6-45b4c79ab8fb"] = {
    "domain": "vs_code",
    "category": "live_app_state",
    "status": "active",
    "rationale": "The OpenProject probe can report multiple VS Code windows "
    "when a stale empty window coexists with the correct project window. The "
    "task only requires the folder named 'project' to be open in some window; "
    "accept exact per-line membership instead of byte-exact or loose substring "
    "matching.",
    "osworld_ref": "metrics/vscode.py:compare_config OpenProject.txt",
}

CONFIG_PATCHES["59f21cfb-0120-4326-b255-a5b827b38967"] = _patch_59f21cfb_vlc_process_probe
METRIC_OVERRIDES["59f21cfb-0120-4326-b255-a5b827b38967"] = {
    "is_vlc_playing": _is_vlc_playing_59f21cfb,
}
PATCH_META["59f21cfb-0120-4326-b255-a5b827b38967"] = {
    "domain": "vlc",
    "category": "live_app_state",
    "status": "active",
    "rationale": "The user task asks VLC to play a desktop video. Upstream "
    "checks only VLC's HTTP RC status.xml, but a valid GUI/CLI VLC launch may "
    "not enable that HTTP interface. Probe VLC process/open-file state and "
    "require the target desktop video filename/path to appear.",
    "osworld_ref": "metrics/vlc.py:is_vlc_playing + getters/vlc.py:vlc_playing_info",
}

METRIC_OVERRIDES["7dbc52a6-11e0-4c9a-a2cb-1e36cfda80d8"] = {
    "compare_pptx_files": _compare_pptx_skip_empty_alignment,
}
PATCH_META["7dbc52a6-11e0-4c9a-a2cb-1e36cfda80d8"] = {
    "domain": "libreoffice_impress",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The result satisfies the notes and bold-title requirements; "
    "upstream compare_pptx_files fails on alignment metadata for empty "
    "paragraph/placeholders. Normalize alignment only for empty paragraphs, "
    "leaving visible text and formatting checks intact.",
    "osworld_ref": "metrics/slides.py:compare_pptx_files empty paragraph alignment",
}

METRIC_OVERRIDES["a53f80cd-4a90-4490-8310-097b011433f6"] = {
    "compare_pptx_files": _compare_pptx_text_height_tolerant,
}
PATCH_META["a53f80cd-4a90-4490-8310-097b011433f6"] = {
    "domain": "libreoffice_impress",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The task grades title font color/bold and deletion of "
    "personal-info shapes. LibreOffice GUI autofit can grow text-placeholder "
    "height after style changes while python-pptx keeps the seed height. For "
    "this task, normalize only text-shape heights to expected before "
    "delegating; left/top/width, shape count, non-text shapes, text and font "
    "checks remain strict.",
    "osworld_ref": "metrics/slides.py:compare_pptx_files text shape geometry",
}

METRIC_OVERRIDES["b5062e3e-641c-4e3a-907b-ac864d2e7652"] = {
    "compare_table": _compare_table_b5062e3e_authors_alias,
}
PATCH_META["b5062e3e-641c-4e3a-907b-ac864d2e7652"] = {
    "domain": "multi_apps",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The result table content and sorting match the gold. Upstream "
    "compare_table reads a literal sheet named 'Sheet1' and the gold header is "
    "'Mail', while the result naturally uses sheet 'Authors' and header "
    "'Email'. Normalize only the first-sheet name and e-mail header alias for "
    "this task, then delegate all value/range checks to compare_table.",
    "osworld_ref": "metrics/table.py:compare_table sheet name + sheet_fuzzy header",
}

METRIC_OVERRIDES["eb303e01-261e-4972-8c07-c9b4e7a4922a"] = {
    "compare_pptx_files": _compare_pptx_trim_trailing_empty_paragraphs,
}
PATCH_META["eb303e01-261e-4972-8c07-c9b4e7a4922a"] = {
    "domain": "multi_apps",
    "category": "over_strict_formatting",
    "status": "active",
    "rationale": "The task grades slide speaker notes. The result/gold visible "
    "slide-body text matches, but upstream compare_pptx_files fails because "
    "the result has an extra trailing empty paragraph in a text body. Trim only "
    "trailing empty text paragraphs before delegating, preserving all non-empty "
    "text, formatting and notes checks.",
    "osworld_ref": "metrics/slides.py:compare_pptx_files paragraph-count check",
}


CONFIG_PATCHES["ee9a3c83-f437-4879-8918-be5efbb9fac7"] = drop_bash_history_metrics
PATCH_META["ee9a3c83-f437-4879-8918-be5efbb9fac7"] = {
    "domain": "multi_apps",
    "category": "bash_history",
    "status": "active",
    "rationale": "Drops the history-grep metric; keeps compare_csv against the "
    "gold CSV. The CSV content check is the real success criterion.",
    "osworld_ref": "metrics/general.py:28 check_include_exclude over a "
    "vm_command_line `cat ~/.bash_history | grep ...`",
}


CONFIG_PATCHES["510f64c8-9bcc-4be1-8d30-638705850618"] = drop_bash_history_metrics
PATCH_META["510f64c8-9bcc-4be1-8d30-638705850618"] = {
    "domain": "multi_apps",
    "category": "bash_history",
    "status": "active",
    "rationale": "Drops history-grep; keeps compare_config(OpenProject.txt) "
    "which already confirms VS Code opened ~/Desktop/project.",
    "osworld_ref": "metrics/general.py:28 + metrics/vscode.py:186 compare_config",
}


CONFIG_PATCHES["f7dfbef3-7697-431c-883a-db8583a4e4f9"] = drop_bash_history_metrics
PATCH_META["f7dfbef3-7697-431c-883a-db8583a4e4f9"] = {
    "domain": "multi_apps",
    "category": "bash_history",
    "status": "active",
    "rationale": "Drops history-grep; keeps compare_archive (ground-truth=0.996 "
    "on cached result — the .doc->PDF conversion is correct). Safe: a wrong "
    "conversion still fails compare_archive.",
    "osworld_ref": "metrics/general.py:28 + metrics/general.py compare_archive",
}


_TODO: dict[str, dict[str, Any]] = {
    "2b9493d7-49b8-493a-a71b-56cd1f4d6908": {
        "domain": "multi_apps",
        "category": "bash_history",
        "rationale": "ONE vm_command_line probes BOTH `ps aux|grep soffice` "
        "(real: LibreOffice killed) AND `cat ~/.bash_history|grep kill` "
        "(bogus). Cannot drop the whole metric. Fix = rewrite the command to "
        "keep only the ps probe, OR enable MAI_GYM_BASH_WRITE_HISTORY.",
        "osworld_ref": "metrics/general.py:28 check_include_exclude (fused cmd)",
        "suggested": "command rewrite OR harness history fix",
    },
    "7f35355e-02a6-45b5-b140-f0be698bcf85": {
        "domain": "multi_apps",
        "category": "bash_history",
        "rationale": "compare_result_files reads /home/user/Desktop/result.txt "
        "vs gold; the produced result was correct. Verify whether failure is "
        "the history proxy or a result.txt path/format issue before patching.",
        "osworld_ref": "metrics/vscode.py:437 compare_result_files",
        "suggested": "verify; likely no patch needed once history written",
    },
    "3680a5ee-6870-426a-a997-eba929a0d25c": {
        "domain": "multi_apps",
        "category": "live_app_state",
        "rationale": "check_include_exclude requires soffice be launched from a "
        "tty (`ps ... | grep -E 'pts/|tty'`) — bash background launch fails. "
        "Fix = relax to 'soffice running' (drop the tty requirement).",
        "osworld_ref": "metrics/general.py:28 over `ps aux|grep [s]office|awk ...`",
        "suggested": "command rewrite (drop pts/tty gate)",
    },
    "9f3bb592-209d-43bc-bb47-d77d9df56504": {
        "domain": "multi_apps",
        "category": "live_app_state",
        "rationale": "exact_match on stdout of a gold subtitles_script.py; "
        "subtitles were removed correctly but script path/timing differed. "
        "Verify the produced video meets the gold script before patching.",
        "osworld_ref": "metrics/general.py:41 exact_match over vm_command_line",
        "suggested": "verify; possibly result path fix",
    },
    "59f21cfb-0120-4326-b255-a5b827b38967": {
        "domain": "vlc",
        "category": "live_app_state",
        "rationale": "is_vlc_playing queries http://vm_ip:8080/requests/status.xml "
        "(VLC RC HTTP iface). A bash-launched vlc has no --extraintf http, so "
        "the iface is absent though the video plays. Fix = METRIC_OVERRIDE that "
        "checks a process/window playing the file, OR launch vlc with "
        "--extraintf http --http-port 8080 in the harness.",
        "osworld_ref": "metrics/vlc.py:21 is_vlc_playing / getters/vlc.py:11",
        "suggested": "METRIC_OVERRIDE is_vlc_playing (process-based)",
    },
    "f0b971a1-6831-4b9b-a50e-22a6e47f45ba": {
        "domain": "chrome",
        "category": "stale_expectation",
        "rationale": "is_expected_active_tab expects URL .../scores/2019/post4, "
        "which 307-redirects to .../scores/2019/super-bowl-sunday — the literal "
        "'post4' slug can never appear in a real address bar. Fix = update "
        "expected.rules.url to the redirect target (or compare with the "
        "redirected URL). VERIFY the canonical target first.",
        "osworld_ref": "metrics/chrome.py:18 is_expected_active_tab + "
        "metrics/utils.py:783 compare_urls(full=True)",
        "suggested": "config patch: set expected url to super-bowl-sunday",
    },
    "58565672-7bfe-48ab-b828-db349231de6b": {
        "domain": "multi_apps",
        "category": "stale_expectation",
        "rationale": "is_expected_tabs compares open-tab URLs exactly; trips on "
        "redirect / trailing slash. Fix = compare normalized URLs.",
        "osworld_ref": "metrics/chrome.py:131 is_expected_tabs",
        "suggested": "METRIC_OVERRIDE is_expected_tabs (normalized compare)",
    },
    "eb303e01-261e-4972-8c07-c9b4e7a4922a": {
        "domain": "multi_apps",
        "category": "stale_expectation",
        "rationale": "compare_pptx_files: the per-shape paragraph-count check at "
        "slides.py ~505 is UNCONDITIONAL (not gated by examine_text), so it "
        "fails legitimate layouts even with examine_shape=False. Fix = "
        "METRIC_OVERRIDE that gates the paragraph-count check behind "
        "examine_text. REVIEW: ensure it doesn't loosen genuinely-text tasks.",
        "osworld_ref": "metrics/slides.py:~505 compare_pptx_files",
        "suggested": "METRIC_OVERRIDE compare_pptx_files (guard para-count)",
    },
    "b5062e3e-641c-4e3a-907b-ac864d2e7652": {
        "domain": "multi_apps",
        "category": "stale_expectation",
        "rationale": "compare_table sheet_fuzzy with sheet_idx 'RNSheet1'/'ENSheet1'. "
        "_parse_sheet_idx treats 'RN'/'EN' as a literal sheet NAME ('Sheet1'); "
        "verify whether the produced workbook's sheet name matches. If the bug "
        "is a wrong-sheet read, METRIC_OVERRIDE compare_table or fix the "
        "sheet_idx in the config.",
        "osworld_ref": "metrics/table.py:44 _parse_sheet_idx",
        "suggested": "verify sheet naming; config sheet_idx fix or override",
    },
    "aa4b5023-aef6-4ed9-bdc9-705f59ab9ad6": {
        "domain": "vlc",
        "category": "stale_expectation",
        "rationale": "compare_videos result getter hard-codes "
        "/home/user/<file>.mp4 (home root) but the instruction says save it on "
        "the Desktop. Fix = update the result getter path to the Desktop "
        "location (config: result[i].path).",
        "osworld_ref": "metrics/vlc.py:296 compare_videos + getters/file.py:74 get_vm_file",
        "suggested": "config patch: fix result path to ~/Desktop/...",
    },
    "88fe4b2d-3040-4c70-9a70-546a47764b48": {
        "domain": "libreoffice_writer",
        "category": "over_strict_formatting",
        "rationale": "REVERTED from active: ignore_blanks=True is UNSAFE here "
        "(this task grades inserted empty lines; collapsing whitespace passes a "
        "do-nothing answer). Correct fix = METRIC_OVERRIDE comparing per-paragraph "
        ".rstrip() while KEEPING paragraph-count/structure. Re-verify the score "
        "correction too (run_2 had 44 vs 40 paragraphs).",
        "osworld_ref": "metrics/docs.py:159 compare_docx_files (ignore_blanks=False)",
        "suggested": "METRIC_OVERRIDE (rstrip-only, preserve structure); NOT ignore_blanks",
    },
    "5bc63fb9-276a-4439-a7c1-9dc76401737f": {
        "domain": "multi_apps",
        "category": "over_strict_formatting",
        "rationale": "compare_docx_files_and_ignore_new_lines with "
        "ignore_blanks=False does exact per-paragraph string match. Relax "
        "ignore_blanks=True. (Note: only run_1/run_3 produced correct content.)",
        "osworld_ref": "metrics/docs.py:980",
        "suggested": "set_all_options(ignore_blanks=True)",
    },
    "236833a3-5704-47fc-888c-4f298f09f799": {
        "domain": "multi_apps",
        "category": "over_strict_formatting",
        "rationale": "compare_docx_files; mostly-correct content. Confirm whether "
        "remaining diffs are formatting-only before relaxing.",
        "osworld_ref": "metrics/docs.py:159 compare_docx_files",
        "suggested": "verify; maybe enable fuzzy_match",
    },
    "7ff48d5b-2df2-49da-b500-a5150ffc7f18": {
        "domain": "multi_apps",
        "category": "over_strict_formatting",
        "rationale": "fuzzy_place_math hard-gates `len(fuzzy_score_list)!=3 -> 0`; "
        "splitting CJK addresses by whitespace is unstable. Fix = METRIC_OVERRIDE "
        "with robust tokenization / drop the exact-count gate.",
        "osworld_ref": "metrics/general.py:101 fuzzy_place_math",
        "suggested": "METRIC_OVERRIDE fuzzy_place_math",
    },
    "df67aebb-fb3a-44fd-b75b-51b6012df509": {
        "domain": "multi_apps",
        "category": "over_strict_formatting",
        "rationale": "compare_text_file on references.bib; under ignore_blanks=True "
        "it matches. Ensure options.ignore_blanks=True (set if absent).",
        "osworld_ref": "metrics/vscode.py:77 compare_text_file",
        "suggested": "set_all_options(ignore_blanks=True)",
    },
    "42d25c08-fb87-4927-8b65-93631280a26f": {
        "domain": "multi_apps",
        "category": "path_packaging",
        "rationale": "compare_epub does structural toc/content equality; a "
        "valid epub with equivalent-but-not-identical packaging fails. Fix = "
        "METRIC_OVERRIDE comparing rendered text rather than zip structure.",
        "osworld_ref": "metrics/others.py:72 compare_epub",
        "suggested": "METRIC_OVERRIDE compare_epub (content-level)",
    },
    "bf4e9888-f10f-47af-8dba-76413038b73c": {
        "domain": "libreoffice_impress",
        "category": "path_packaging",
        "rationale": "compare_pptx_files result getter reads a FIXED template "
        "path, not the path the agent saved to. Fix = update result[i].path to "
        "the actual save location. (Only 2/3 runs were correct.)",
        "osworld_ref": "metrics/slides.py:190 + getters/file.py get_vm_file",
        "suggested": "config patch: fix result path",
    },
    "5812b315-e7bd-4265-b51f-863c02174c28": {
        "domain": "os",
        "category": "path_packaging",
        "rationale": "check_include_exclude expects a probe string only emitted "
        "by an interactive login shell. The SSH user was created correctly. "
        "Fix = run the verification probe in a login shell, or relax the "
        "expected string.",
        "osworld_ref": "metrics/general.py:28 check_include_exclude over vm_command_line",
        "suggested": "command rewrite (login shell) OR relax expected",
    },
}
for _uid, _meta in _TODO.items():
    _meta.setdefault("status", "todo")
    PATCH_META.setdefault(_uid, _meta)
