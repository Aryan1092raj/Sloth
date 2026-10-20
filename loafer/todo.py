"""Todo handling: read the digest, tick items off, remember what is done.

The tick state lives in a sidecar file, not in the digest, because `watch` rewrites the
daily digest whenever new mail arrives and would wipe ticks written into it.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

SPACE = " "
UNCHECKED = f"- [{SPACE}]{SPACE}"
CHECKED = f"- [{SPACE}x]{SPACE}"
DONE_FILE = ".loafer-done.json"


def digest_files(out_dir: Path) -> list[Path]:
    """Digest files, oldest first. Daily files are named YYYY-MM-DD.md."""
    out_dir = Path(out_dir).expanduser()
    if not out_dir.exists():
        return []
    files = [p for p in out_dir.glob("*.md") if p.is_file()]
    return sorted(files, key=lambda p: (p.name.startswith("digest-"), p.name))


def latest_digest(out_dir: Path) -> Path | None:
    """Today's daily digest if it exists, otherwise the most recently written one."""
    out_dir = Path(out_dir).expanduser()
    files = digest_files(out_dir)
    if not files:
        return None
    today = out_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"
    if today.exists():
        return today
    try:
        return max(files, key=lambda p: p.stat().st_mtime)
    except OSError:
        return files[-1]


def done_path(out_dir: Path) -> Path:
    return Path(out_dir).expanduser() / DONE_FILE


def load_done(out_dir: Path) -> set[str]:
    path = done_path(out_dir)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return set()
    items = data.get("done") if isinstance(data, dict) else data
    return {str(item).lower() for item in items or []}


def save_done(out_dir: Path, items: list[str]) -> set[str]:
    done = load_done(out_dir) | {item.lower() for item in items}
    path = done_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"done": sorted(done)}, indent=2), encoding="utf-8")
    return done


def open_items(text: str, done: set[str] | None = None) -> list[str]:
    """Every unchecked item, in order, minus the ones already ticked off."""
    done = done or set()
    items: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(UNCHECKED.strip()):
            continue
        item = stripped[len(UNCHECKED.strip()):].strip()
        key = item.lower()
        if key in done or key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items


def tick(text: str, needle: str) -> tuple[str, list[str]]:
    """Tick every open item that contains the needle. Returns the new text and what got ticked."""
    needle = needle.lower()
    ticked: list[str] = []
    out_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(UNCHECKED.strip()) and needle in stripped.lower():
            item = stripped[len(UNCHECKED.strip()):].strip()
            ticked.append(item)
            out_lines.append(CHECKED + item)
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), ticked


def tick_file(path: Path, needle: str) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    new_text, ticked = tick(text, needle)
    if ticked:
        path.write_text(new_text, encoding="utf-8")
    return ticked

def export_open_items(out_dir: Path, path: Path, items: list[str] | None = None) -> Path:
    """Write the open items to a plain markdown todo file, for whatever tool you like."""
    if items is None:
        latest = latest_digest(out_dir)
        items = [] if latest is None else open_items(latest.read_text(encoding="utf-8"), load_done(out_dir))
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = [f"# Todo - {datetime.now().strftime('%Y-%m-%d')}", ""]
    body += [f"{UNCHECKED}{item}" for item in items] if items else ["_nothing open_"]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path
