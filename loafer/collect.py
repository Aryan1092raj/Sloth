"""Collect text files worth reading. Boring, predictable, no surprises."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

TEXT_SUFFIXES = {".txt", ".md", ".log", ".csv", ".json", ".chat", ".eml"}
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", ".loafer"}

DURATIONS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_since(value: str, now: datetime | None = None) -> datetime | None:
    """Turn '2h', 'yesterday', 'today', 'now' or an ISO date into a cutoff time."""
    if not value:
        return None
    text = value.strip().lower()
    now = now or datetime.now()
    if text == "now":
        return now
    if text == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if text == "yesterday":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    match = re.fullmatch(r"(\d+)\s*([smhdw])", text)
    if match:
        return now - timedelta(seconds=int(match.group(1)) * DURATIONS[match.group(2)])
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(
            f"cannot read --since '{value}'. use 30m, 6h, 2d, today, yesterday, now or YYYY-MM-DD"
        ) from None


def collect_files(
    root: Path,
    exclude_dirs: list[Path] | None = None,
    max_bytes: int = 400_000,
    since: datetime | None = None,
) -> list[Path]:
    """Text files under root, sorted by path. Skips hidden dirs, big files, old files and excluded dirs."""
    root = Path(root).expanduser()
    if not root.exists():
        return []
    excluded = [Path(p).expanduser().resolve() for p in (exclude_dirs or [])]

    def is_excluded(path: Path) -> bool:
        resolved = path.resolve()
        return any(resolved == ex or ex in resolved.parents for ex in excluded)

    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.relative_to(root).parts[:-1]):
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if is_excluded(path):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > max_bytes:
            continue
        if since is not None and datetime.fromtimestamp(stat.st_mtime) < since:
            continue
        found.append(path)
    return found


def read_text(path: Path, max_bytes: int = 400_000) -> str:
    """Best-effort text read. Bad bytes become '?', never an exception."""
    try:
        data = Path(path).read_bytes()[:max_bytes]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"
