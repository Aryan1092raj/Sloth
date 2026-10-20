"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import __version__, collect, digest, grocery, history, todo, watch as watch_mod
from .config import QUALITY_PRESETS, load_config
from .llm import BackendError, OllamaBackend, pick_analyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loafer",
        description="Loafer - a local sloth that reads your chats while you sleep.",
    )
    parser.add_argument("--version", action="version", version=f"loafer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    dg = sub.add_parser("digest", help="read the inbox once and write a digest")
    dg.add_argument("--input", type=Path, help="folder of chat/email exports")
    dg.add_argument("--out", type=Path, help="where digests are written")
    dg.add_argument("--model", help="model name (ollama tag or llama.cpp model id)")
    dg.add_argument("--quality", choices=sorted(QUALITY_PRESETS), help="fast, balanced or deep")
    dg.add_argument("--backend", choices=["auto", "ollama", "llamacpp", "heuristic"])
    dg.add_argument("--file", action="append", default=[], help="digest this exact file (repeatable)")
    dg.add_argument("--since", help="only files touched since: 30m, 6h, 2d, today, yesterday, YYYY-MM-DD")
    dg.add_argument("--json", action="store_true", help="print raw json instead of markdown")

    wt = sub.add_parser("watch", help="keep an eye on the inbox and refresh the daily digest")
    wt.add_argument("--input", type=Path)
    wt.add_argument("--out", type=Path)
    wt.add_argument("--model")
    wt.add_argument("--quality", choices=sorted(QUALITY_PRESETS))
    wt.add_argument("--backend", choices=["auto", "ollama", "llamacpp", "heuristic"])
    wt.add_argument("--interval", type=int, help="seconds between checks")
    wt.add_argument("--since", help="ignore anything older than this")
    wt.add_argument("--poll", action="store_true", help="poll instead of using inotify")
    wt.add_argument("--once", action="store_true", help="do a single pass and exit")

    gr = sub.add_parser("grocery", help="fridge photo -> shopping list")
    gr.add_argument("image", type=Path)
    gr.add_argument("--model", help="vision model (default qwen2.5vl:3b)")
    gr.add_argument("--out", type=Path, help="write the list to this file too")

    td = sub.add_parser("todo", help="show what is still open in the latest digest")
    td.add_argument("--out", type=Path)
    td.add_argument("--from", dest="source", type=Path, help="use this digest file instead of the latest")
    td.add_argument("--export", type=Path, help="write the open items to this markdown file")

    dn = sub.add_parser("done", help="tick items off in the latest digest")
    dn.add_argument("needle", help="text that identifies the item")
    dn.add_argument("--out", type=Path)
    dn.add_argument("--from", dest="source", type=Path, help="use this digest file instead of the latest")

    hs = sub.add_parser("history", help="list past digests, or search them")
    hs.add_argument("query", nargs="?", help="text to search for")
    hs.add_argument("--out", type=Path)
    hs.add_argument("--limit", type=int, default=50)

    doc = sub.add_parser("doctor", help="check that the local model and paths are ready")
    doc.add_argument("--json", action="store_true", help="print diagnostics as json")
    return parser


def _digest_file(cfg, source: Path | None) -> Path | None:
    if source is not None:
        path = Path(source).expanduser()
        return path if path.exists() else None
    return todo.latest_digest(cfg.out_dir)


def cmd_digest(args) -> int:
    cfg = load_config(
        input_dir=args.input,
        out_dir=args.out,
        model=args.model,
        backend=args.backend,
        quality=args.quality,
    )
    since = collect.parse_since(args.since) if args.since else None
    analyzer = pick_analyzer(cfg)
    if args.file:
        files = [Path(f).expanduser() for f in args.file]
    else:
        files = collect.collect_files(
            cfg.input_dir,
            exclude_dirs=[cfg.out_dir],
            max_bytes=cfg.max_file_bytes,
            since=since,
        )
    if not files:
        print(f"nothing to read in {cfg.input_dir}", file=sys.stderr)
        return 0
    merged = digest.build_digest(cfg, files, analyzer)
    if args.json:
        print(json.dumps(merged, indent=2, ensure_ascii=False))
        return 0
    text = digest.render(
        merged,
        when=datetime.now().strftime("%Y-%m-%d %H:%M"),
        mode=analyzer.describe(),
        sources=len(files),
    )
    path = digest.write_digest(cfg.out_dir, text, f"digest-{datetime.now().strftime('%Y-%m-%d-%H%M')}.md")
    print(text)
    print(f"written to {path}")
    return 0


def cmd_watch(args) -> int:
    cfg = load_config(
        input_dir=args.input,
        out_dir=args.out,
        model=args.model,
        backend=args.backend,
        quality=args.quality,
        interval=args.interval,
    )
    analyzer = pick_analyzer(cfg)
    return watch_mod.watch(cfg, analyzer, once=args.once, use_inotify=not args.poll)


def cmd_grocery(args) -> int:
    cfg = load_config(vl_model=args.model)
    result = grocery.photo_to_items(cfg, args.image)
    text = grocery.render_list(result["present"], result["missing"])
    print(text)
    if args.out:
        path = Path(args.out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"written to {path}")
    return 0


def cmd_todo(args) -> int:
    cfg = load_config(out_dir=args.out)
    path = _digest_file(cfg, args.source)
    if path is None:
        print("no digest yet. run: loafer digest --input DIR")
        return 0
    items = todo.open_items(path.read_text(encoding="utf-8"), todo.load_done(cfg.out_dir))
    print(f"open items in {path}")
    if not items:
        print("nothing open. the sloth is proud of you.")
        return 0
    for item in items:
        print(f"{todo.UNCHECKED}{item}")
    if args.export:
        target = todo.export_open_items(cfg.out_dir, args.export, items)
        print(f"written to {target}")
    return 0


def cmd_done(args) -> int:
    cfg = load_config(out_dir=args.out)
    path = _digest_file(cfg, args.source)
    if path is None:
        print("no digest yet. run: loafer digest --input DIR")
        return 0
    text = path.read_text(encoding="utf-8")
    done = todo.load_done(cfg.out_dir)
    pending = [item for item in todo.open_items(text, done) if args.needle.lower() in item.lower()]
    if not pending:
        print(f"nothing open matched '{args.needle}' in {path}")
        return 0
    todo.tick_file(path, args.needle)
    todo.save_done(cfg.out_dir, pending)
    print(f"ticked {len(pending)} item(s) in {path}")
    for item in pending:
        print(f"  {item}")
    return 0


def cmd_history(args) -> int:
    cfg = load_config(out_dir=args.out)
    if args.query:
        matches = history.search(cfg.out_dir, args.query, limit=args.limit)
        if not matches:
            print(f"nothing in {cfg.out_dir} matches '{args.query}'")
            return 0
        for path, number, line in matches:
            print(f"{path.name}:{number}: {line}")
        print(f"{len(matches)} match(es)")
        return 0
    records = history.inventory(cfg.out_dir)
    if not records:
        print(f"no digests in {cfg.out_dir} yet")
        return 0
    for record in records:
        stamp = datetime.fromtimestamp(record["written"]).strftime("%Y-%m-%d %H:%M")
        print(f"{record['name']:<30} {stamp}  {record['open']} open")
    return 0


def cmd_doctor(args) -> int:
    cfg = load_config()
    models: list[str] = []
    backend = OllamaBackend(cfg.ollama_host, cfg.model)
    try:
        models = backend.models()
    except BackendError:
        models = []
    active = None
    try:
        active = pick_analyzer(cfg).describe()
    except BackendError:
        active = None
    files = collect.collect_files(cfg.input_dir, exclude_dirs=[cfg.out_dir], max_bytes=cfg.max_file_bytes)
    report = {
        "loafer": __version__,
        "python": sys.version.split()[0],
        "inbox": str(cfg.input_dir),
        "inbox_exists": cfg.input_dir.exists(),
        "digests": str(cfg.out_dir),
        "backend": cfg.backend,
        "quality": cfg.quality,
        "active": active,
        "ollama": cfg.ollama_host,
        "ollama_reachable": bool(models),
        "models": models,
        "readable_files": len(files),
    }
    if args.json:
        print(json.dumps(report, indent=2))
        return 0
    print(f"loafer {__version__}")
    print(f"python      {report['python']}")
    print(f"inbox       {cfg.input_dir} ({'exists' if cfg.input_dir.exists() else 'missing'})")
    print(f"digests     {cfg.out_dir}")
    print(f"backend     {cfg.backend} (quality {cfg.quality})")
    print(f"active      {active or 'none'}")
    print(f"ollama      {cfg.ollama_host} -> {', '.join(models) if models else 'no models pulled'}")
    print(f"readable    {len(files)} file(s) waiting")
    for path in files[:10]:
        print(f"            {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "digest": cmd_digest,
        "watch": cmd_watch,
        "grocery": cmd_grocery,
        "todo": cmd_todo,
        "done": cmd_done,
        "history": cmd_history,
        "doctor": cmd_doctor,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except (BackendError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
