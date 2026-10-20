"""Ask a question about your chats.

Retrieval is plain BM25 over windows of text, computed with the standard library.
No embeddings, no vector database, no index to keep in sync. Then the local model
reads the top windows and answers, and every answer comes with its sources.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from . import collect, fallback
from .config import Config

WINDOW = 600
OVERLAP = 120
TOP_K = 6



def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in fallback.STOPWORDS and len(w) > 1]


def windows(text: str, size: int = WINDOW, overlap: int = OVERLAP) -> list[tuple[int, str]]:
    """(first line number, text) windows over a document."""
    lines = text.splitlines()
    out: list[tuple[int, str]] = []
    start_line = 0
    while start_line < len(lines) and len(out) < 2000:
        chunk_lines: list[str] = []
        chars = 0
        end_line = start_line
        while end_line < len(lines) and chars < size:
            line = lines[end_line]
            chunk_lines.append(line)
            chars += len(line) + 1
            end_line += 1
        chunk = "\n".join(chunk_lines).strip()
        if chunk:
            out.append((start_line + 1, chunk))
        if end_line >= len(lines):
            break
        step = max(1, size - overlap) // 40 or 1
        next_start = end_line - step
        if next_start <= start_line:
            next_start = end_line  # never stand still, or this loops forever
        start_line = next_start
    return out


def collect_documents(cfg: Config) -> list[tuple[Path, int, str]]:
    """Windows from the inbox and from the digests already written."""
    docs: list[tuple[Path, int, str]] = []
    folders = [cfg.input_dir] + [cfg.out_dir]
    for folder in folders:
        for path in collect.collect_files(folder, exclude_dirs=[cfg.out_dir], max_bytes=cfg.max_file_bytes):
            text = collect.read_text(path, cfg.max_file_bytes)
            if not text.strip():
                continue
            for number, chunk in windows(text):
                docs.append((path, number, chunk))
    return docs


def rank(question: str, docs: list[tuple[Path, int, str]], top_k: int = TOP_K) -> list[tuple[Path, int, str, float]]:
    """BM25 over the windows. Simple, local, no dependencies."""
    query = tokenize(question)
    if not query or not docs:
        return []
    token_lists = [tokenize(doc[2]) for doc in docs]
    lengths = [len(tokens) for tokens in token_lists]
    average = sum(lengths) / max(1, len(lengths))
    document_freq = Counter()
    for tokens in token_lists:
        document_freq.update(set(tokens))
    total = len(docs)
    scored: list[tuple[Path, int, str, float]] = []
    for index, tokens in enumerate(token_lists):
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0.0
        for term in query:
            if term not in counts:
                continue
            idf = math.log(1 + (total - document_freq[term] + 0.5) / (document_freq[term] + 0.5))
            score += idf * (counts[term] * 2.5) / (counts[term] + 1.5 * (1 - 0.75 + 0.75 * lengths[index] / average))
        scored.append((docs[index][0], docs[index][1], docs[index][2], score))
    scored.sort(key=lambda item: item[3], reverse=True)
    # keep the best windows even when they scored nothing, a small model reads better
    # with context than with an empty prompt
    return scored[:top_k]


def format_blocks(hits: list[tuple[Path, int, str, float]]) -> str:
    blocks = []
    for path, number, text, _ in hits:
        blocks.append(f"### {path.name} (from line {number})\n{text}")
    return "\n\n".join(blocks)


REFUSALS = (
    "do not contain",
    "don't contain",
    "does not contain",
    "cannot answer",
    "can't answer",
    "no mention",
    "not mentioned",
    "insufficient information",
    "don't have enough information",
    "no information",
)


def is_refusal(answer: str) -> bool:
    """Small models sometimes refuse an answer that is right there. Catch that."""
    lowered = answer.lower()
    return any(phrase in lowered for phrase in REFUSALS)


def ask(cfg: Config, question: str, analyzer, top_k: int = TOP_K) -> dict:
    """Answer a question from the local corpus. Returns the answer plus its sources."""
    docs = collect_documents(cfg)
    hits = rank(question, docs, top_k=top_k)
    if not hits:
        return {"answer": "nothing in the inbox matches that question.", "sources": [], "mode": "none"}
    blocks = format_blocks(hits)
    prompt = (
        f"QUESTION: {question}\n\n"
        f"EXCERPTS:\n{blocks}\n\n"
        "Rules:\n"
        "- answer from the excerpts only, never from general knowledge\n"
        "- if the answer is not obvious, quote the most relevant line\n"
        "- name the source file when you use an excerpt\n"
        "- two or three sentences is enough"
    )
    try:
        answer = analyzer.answer(prompt)
    except Exception:
        answer = ""
    if not answer.strip() or is_refusal(answer):
        # ponytail: no model, a bad reply, or a refusal means show the excerpts, never invent
        answer = "Top matching excerpts:\n" + "\n\n".join(
            f"- {path.name}:{number}: {text[:200]}" for path, number, text, _ in hits
        )
    return {
        "answer": answer.strip(),
        "sources": [{"file": str(path), "line": number, "score": round(score, 3)} for path, number, _, score in hits],
        "mode": analyzer.describe(),
    }
