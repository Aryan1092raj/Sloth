"""Search the digests you already wrote."""

from __future__ import annotations

from datetime import datetime
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


def source_of(item: str) -> str:
    """The thread an item came from, if the digest recorded one."""
    text = item.strip()
    if text.endswith(")") and " (" in text:
        return text[text.rfind(" (") + 2: -1]
    return ""


def recurring(out_dir: Path, min_appearances: int = 2, limit: int = 20) -> list[dict]:
    """Promises that keep coming back. The sloth nags about these."""
    appearances: dict[str, dict] = {}
    for path in todo.digest_files(out_dir):
        try:
            text = path.read_text(encoding="utf-8")
            stamp = path.stat().st_mtime
        except OSError:
            continue
        # daily digests are named YYYY-MM-DD.md, that is the day they are about
        stem = path.stem
        day = stem if len(stem) == 10 and stem[4] == "-" and stem[7] == "-" else datetime.fromtimestamp(stamp).strftime("%Y-%m-%d")
        for item in todo.open_items(text):
            base = todo.base_item(item)
            key = base.lower()
            if not key:
                continue
            entry = appearances.setdefault(key, {"item": base, "days": [], "files": set()})
            if day not in entry["days"]:
                entry["days"].append(day)
            source = source_of(item)
            if source:
                entry["files"].add(source)
    rows = [entry for entry in appearances.values() if len(entry["days"]) >= min_appearances]
    rows.sort(key=lambda entry: (-len(entry["days"]), entry["item"]))
    return [
        {
            "item": entry["item"],
            "days": len(entry["days"]),
            "first": entry["days"][0],
            "last": entry["days"][-1],
            "files": sorted(entry["files"]),
        }
        for entry in rows[:limit]
    ]
