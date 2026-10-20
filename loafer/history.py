"""Search the digests you already wrote."""

from __future__ import annotations

from pathlib import Path

from . import todo


def search(out_dir: Path, query: str, limit: int = 50) -> list[tuple[Path, int, str]]:
    """Every line in every digest that contains the query."""
    matches: list[tuple[Path, int, str]] = []
    needle = query.lower()
    for path in todo.digest_files(out_dir):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            if needle in line.lower():
                matches.append((path, number, line.strip()))
                if len(matches) >= limit:
                    return matches
    return matches


def inventory(out_dir: Path) -> list[dict]:
    """One record per digest: name, when it was written, how many items are still open."""
    records = []
    for path in todo.digest_files(out_dir):
        try:
            text = path.read_text(encoding="utf-8")
            stamp = path.stat().st_mtime
        except OSError:
            continue
        records.append({
            "path": path,
            "name": path.name,
            "written": stamp,
            "open": len(todo.open_items(text, todo.load_done(out_dir))),
        })
    return records
