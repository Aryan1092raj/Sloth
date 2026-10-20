"""Turn a pile of files into one digest."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from . import collect, fallback
from .config import Config
from .llm import Analyzer

SPACE = " "
CHECKBOX = f"- [{SPACE}]{SPACE}"
SEP = chr(44) + SPACE


def two_lines(summary: str) -> str:
    """The digest promises two lines. Make sure there are two."""
    lines = [line.strip() for line in (summary or "").splitlines() if line.strip()]
    if len(lines) >= 2:
        return "\n".join(lines[:2])
    if not lines:
        return ""
    sentences = fallback.split_sentences(lines[0])
    if len(sentences) >= 2:
        return "\n".join(sentences[:2])
    return lines[0]


def split_chunks(text: str, size: int, overlap: int = 300) -> list[str]:
    """Split long text into overlapping chunks so nothing is silently dropped."""
    if size <= 0 or len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    pieces: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            pieces.append(piece)
        if start + size >= len(text):
            break
        start += step
    return pieces


def merge_chunks(results: list[dict]) -> dict:
    """Combine the analyses of the chunks of one file into one record."""
    if not results:
        return {"source": "", "summary": "", "actions": [], "topics": [], "kind": "chat"}
    summaries = [r.get("summary", "").strip() for r in results if r.get("summary", "").strip()]
    actions = fallback.dedupe([a for r in results for a in r.get("actions", [])])
    topics = fallback.dedupe([t for r in results for t in r.get("topics", [])])[:8]
    return {
        "source": results[0].get("source", ""),
        "summary": " ".join(summaries),
        "actions": actions,
        "topics": topics,
        "kind": results[0].get("kind", "chat"),
    }


def analyze_files(cfg: Config, files: list[Path], analyzer: Analyzer) -> list[dict]:
    results = []
    for path in files:
        text = collect.read_text(path, cfg.max_file_bytes)
        if not text.strip():
            continue
        source = str(path)
        if cfg.chunk:
            chunks = split_chunks(text, cfg.max_chars)
        else:
            chunks = [collect.truncate(text, cfg.max_chars)]
        chunk_results = [analyzer.analyze(chunk, source) for chunk in chunks]
        results.append(merge_chunks(chunk_results) if len(chunk_results) > 1 else chunk_results[0])
    return results


def checklist(results: list[dict], merged_actions: list[str]) -> list[str]:
    """To-do rows: per-thread actions first with their source, then anything only the merge found."""
    rows: list[tuple[str, str | None]] = []
    keys: list[str] = []

    def add(text: str, label: str | None) -> None:
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if not key:
            return
        for index, other in enumerate(keys):
            if other == key or other in key or key in other:
                if len(key) > len(other):
                    rows[index] = (text, label)
                    keys[index] = key
                return
        rows.append((text, label))
        keys.append(key)

    for item in results:
        label = Path(str(item.get("source", ""))).name or None
        for action in item.get("actions", []):
            add(action, label)
    for action in merged_actions:
        add(action, None)
    return [text + (f" ({label})" if label else "") for text, label in rows]


def build_digest(cfg: Config, files: list[Path], analyzer: Analyzer) -> dict:
    results = analyze_files(cfg, files, analyzer)
    merged = analyzer.merge(results[-60:])
    merged["results"] = results
    merged["sources"] = [str(p) for p in files]
    merged["checklist"] = checklist(results, merged.get("actions", []))
    return merged


def render(result: dict, *, when: str, mode: str, sources: int) -> str:
    summary = (result.get("summary") or "").strip()
    items = result.get("checklist") or result.get("actions") or []
    topics = result.get("topics") or []
    results = result.get("results") or []
    lines = [f"# Loafer digest - {when}", ""]
    lines += ["## Two lines", "", two_lines(summary) or "_nothing worth reporting_", ""]
    lines += ["## To do", ""]
    lines += [f"{CHECKBOX}{item}" for item in items] if items else ["_nothing to do_"]
    lines += ["", "## Threads", ""]
    if results:
        for item in results:
            name = Path(str(item.get("source", "?"))).name
            body = (item.get("summary") or "").strip() or "_no summary_"
            item_actions = item.get("actions") or []
            lines += [f"### {name}", "", body, ""]
            lines += [f"{CHECKBOX}{action}" for action in item_actions] if item_actions else ["_no actions_"]
            lines += [""]
    else:
        lines += ["_no threads_", ""]
    lines += ["Topics: " + SEP.join(topics) if topics else "Topics: _none_", ""]
    lines += [f"<!-- {sources} source(s) | model: {mode} | generated by loafer -->", ""]
    return "\n".join(lines)


def write_digest(out_dir: Path, text: str, name: str) -> Path:
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
