"""The overnight loop: wake up, read new stuff, sleep again."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

from . import collect, digest
from .config import Config
from .llm import Analyzer


class State:
    """Tiny JSON store: what we already read, and what we already concluded."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: dict = {"seen": {}, "results": [], "digest": None}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except (ValueError, OSError):
                # ValueError covers json.JSONDecodeError and UnicodeDecodeError
                pass

    def is_new(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except OSError:
            return False
        key = str(path)
        seen = self.data["seen"].get(key)
        return not seen or seen.get("mtime") != stat.st_mtime or seen.get("size") != stat.st_size

    def mark(self, path: Path) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        self.data["seen"][str(path)] = {"mtime": stat.st_mtime, "size": stat.st_size}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        tmp.replace(self.path)


def new_files(cfg: Config, state: State) -> list[Path]:
    since = collect.parse_since(cfg.since) if cfg.since else None
    files = collect.collect_files(
        cfg.input_dir,
        exclude_dirs=[cfg.out_dir],
        max_bytes=cfg.max_file_bytes,
        since=since,
    )
    return [f for f in files if state.is_new(f)]


def run_once(cfg: Config, analyzer: Analyzer, state: State) -> dict:
    """One pass over the inbox. Returns the merged digest."""
    files = new_files(cfg, state)
    results = state.data.get("results", [])
    if files:
        fresh = digest.analyze_files(cfg, files, analyzer)
        results = results + fresh
        for path in files:
            state.mark(path)
    merged = analyzer.merge(results[-40:]) if results else {
        "source": "digest", "summary": "", "actions": [], "topics": [], "kind": "digest"
    }
    state.data["results"] = results[-200:]
    state.data["digest"] = merged
    state.save()
    merged["sources"] = len(files)
    merged["new_files"] = [str(p) for p in files]
    return merged


def write_daily(cfg: Config, merged: dict, analyzer: Analyzer) -> Path:
    now = datetime.now()
    text = digest.render(
        merged,
        when=now.strftime("%Y-%m-%d %H:%M"),
        mode=analyzer.describe(),
        sources=len(merged.get("results", [])),
    )
    return digest.write_digest(cfg.out_dir, text, f"{now.strftime('%Y-%m-%d')}.md")


IN_CLOSE_WRITE = 0x00000008
IN_MOVED_TO = 0x00000080
IN_CREATE = 0x00000100
IN_NONBLOCK = 0o4000


def inotify_watch(paths: list[Path]) -> int | None:
    """An inotify file descriptor for the folders, or None if the kernel says no."""
    try:
        import ctypes
        import ctypes.util

        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
        fd = libc.inotify_init1(IN_NONBLOCK)
        if fd < 0:
            return None
        mask = IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE
        watched = 0
        for path in paths:
            if path.is_dir() and libc.inotify_add_watch(fd, str(path).encode(), mask) >= 0:
                watched += 1
        if not watched:
            libc.close(fd)
            return None
        return fd
    except Exception:
        # ponytail: any inotify problem means we fall back to polling, which always works
        return None


def wait_for_change(fd: int, timeout: float) -> bool:
    """Sleep until something happens in a watched folder, or until the timeout."""
    import select

    try:
        ready, _, _ = select.select([fd], [], [], timeout)
    except (OSError, ValueError):
        return False
    if ready:
        try:
            os.read(fd, 4096)  # drain the events, the content does not matter
        except OSError:
            pass
    return bool(ready)


def watch(cfg: Config, analyzer: Analyzer, *, once: bool = False, use_inotify: bool = True) -> int:
    state = State(cfg.state_file)
    folders = [cfg.input_dir] + [p for p in cfg.input_dir.rglob("*") if p.is_dir()] if cfg.input_dir.exists() else []
    fd = inotify_watch(folders) if use_inotify else None
    mode = "inotify" if fd is not None else "polling"
    print(f"loafer watching {cfg.input_dir} ({mode}, {cfg.interval}s fallback, ctrl-c to stop)", flush=True)
    while True:
        merged = run_once(cfg, analyzer, state)
        path = write_daily(cfg, merged, analyzer)
        count = len(merged.get("new_files", []))
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {count} new file(s) -> {path}", flush=True)
        if once:
            return 0
        try:
            if fd is not None:
                wait_for_change(fd, cfg.interval)
            else:
                time.sleep(cfg.interval)
        except KeyboardInterrupt:
            print("sleeping now.", flush=True)
            return 0
