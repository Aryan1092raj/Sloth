"""No-model fallback: extractive summarisation with the standard library only.

Used when no local model is reachable. It is dumb on purpose - it never invents
facts, it only picks sentences that are actually in the text.
"""

from __future__ import annotations

import math
import re
from collections import Counter

STOPWORDS = set(
    """a an the and or but if then else for while of to in on at by with from as is are was were be been being
    this that these those it its i you he she they we me my your our their them us not no yes do does did done
    have has had will would can could should shall may might must just so than too very re ve ll s t m o y
    about after before again all any both each few more most other some such only own same up down out off over
    under here there when where why how what who which am get got go going one two also ok okay yeah hi hello
    thanks thank please hi hey""".split()
)

ACTION_CUES = (
    "todo",
    "please",
    "can you",
    "could you",
    "need to",
    "needs to",
    "you should",
    "we should",
    "i'll",
    "i will",
    "remind",
    "don't forget",
    "do not forget",
    "deadline",
    "due by",
    "urgent",
    "asap",
    "must ",
    "let's",
    "action:",
    "- [ ]",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in STOPWORDS and len(w) > 2]


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return [p for p in parts if len(p) > 2]


def top_terms(text: str, limit: int = 6) -> list[str]:
    counts = Counter(content_words(text))
    return [term for term, _ in counts.most_common(limit)]


def extractive_summary(text: str, max_sentences: int = 2) -> str:
    """Two best sentences, in the order they appeared."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    counts = Counter(content_words(text))
    scored = []
    for idx, sentence in enumerate(sentences):
        words = content_words(sentence)
        if not words:
            continue
        score = sum(counts[w] for w in words) / math.sqrt(len(words))
        scored.append((score, idx, sentence))
    if not scored:
        return sentences[0][:200]
    scored.sort(reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda item: item[1])
    return " ".join(sentence for _, _, sentence in chosen)


def find_actions(text: str, limit: int = 12) -> list[str]:
    actions: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*>•").strip()
        if not line or len(line) > 400:
            continue
        lowered = line.lower()
        if any(cue in lowered for cue in ACTION_CUES):
            cleaned = re.sub(r"^\[[^\]]*\]\s*", "", line)  # drop "[09:12] Ana:" prefixes
            cleaned = re.sub(r"^\w+:\s*", "", cleaned) if cleaned.count(":") == 1 else cleaned
            actions.append(cleaned[:140])
    return dedupe(actions)[:limit]


def dedupe(items: list[str]) -> list[str]:
    """Drop repeats, and drop the vague version of something more specific."""
    out: list[str] = []
    keys: list[str] = []
    for item in items:
        key = re.sub(r"\W+", " ", item.lower()).strip()
        if not key:
            continue
        if any(key == other for other in keys):
            continue
        overlaps = [i for i, other in enumerate(keys) if other and (other in key or key in other)]
        if overlaps:
            for i in overlaps:
                if len(key) > len(keys[i]):
                    out[i] = item
                    keys[i] = key
            continue
        out.append(item)
        keys.append(key)
    return out


def analyze(text: str, source: str = "") -> dict:
    return {
        "source": source,
        "summary": extractive_summary(text),
        "actions": find_actions(text),
        "topics": top_terms(text),
        "kind": "heuristic",
    }


def merge(results: list[dict]) -> dict:
    summaries = [r.get("summary", "").strip() for r in results if r.get("summary", "").strip()]
    actions = dedupe([a for r in results for a in r.get("actions", [])])
    topics = Counter(t for r in results for t in r.get("topics", []))
    return {
        "source": "digest",
        "summary": "\n".join(summaries[:2]),
        "actions": actions[:12],
        "topics": [t for t, _ in topics.most_common(8)],
        "kind": "heuristic",
    }
