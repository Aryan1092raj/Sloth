"""Local model backends + the analyzers that talk to them.

Everything here talks to localhost. No API keys, no telemetry, no egress.
"""

from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from string import Template
from typing import Any

from . import fallback
from .config import MODEL_PREFERENCE, Config

ANSWER_SYSTEM = (
    "You answer questions using only the excerpts you are given. "
    "If the excerpts do not contain the answer, say so plainly."
)

SYSTEM_PROMPT = (
    "You extract structured facts from chat logs and notes. "
    "Reply with ONE JSON object and nothing else. No prose, no markdown fences, no preamble."
)

PER_FILE_PROMPT = Template(
    """Summarise this conversation fragment.

Return JSON only, exactly this shape:
{"summary": "one or two sentences", "actions": ["short imperative task"], "topics": ["topic"], "kind": "chat|email|note|code|other"}

Rules:
- actions only when a person promised to do something or explicitly asked someone to.
- ignore small talk, greetings, emoji, reactions.
- keep each action under 90 characters.
- name the object and keep the deadline if one was stated, for example "send the Q3 numbers to the board by Wednesday".

--- source: $source ---
$text
"""
)

MERGE_PROMPT = Template(
    """Combine these per-thread results into one digest.

Return JSON only, exactly this shape:
{"summary": "exactly two short lines separated by a newline", "actions": ["short imperative task"], "topics": ["topic"]}

Rules: remove duplicates, keep the most important actions first, no more than 10 actions.
- keep the concrete wording from the threads, do not generalise.

$blocks
"""
)


class BackendError(RuntimeError):
    pass


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def hostname_of(url: str) -> str:
    parsed = urllib.parse.urlparse(url if "://" in url else "http://" + url)
    return (parsed.hostname or "").lower()


def ensure_local(url: str) -> None:
    """Refuse to send anything off this machine, unless the user insists.

    Loafer never downloads a model and never phones home. The only host it is willing
    to talk to is the local model server. Point it at a remote API by accident and it
    says no instead of quietly uploading your chat logs.
    """
    host = hostname_of(url)
    if host in LOCAL_HOSTS or host.startswith("127."):
        return
    override = os.environ.get("LOAFER_ALLOW_REMOTE", "").strip().lower()
    if override not in ("", "0", "false", "no"):
        return
    raise BackendError(
        f"refusing to send your data to '{host}'. Loafer only talks to localhost. "
        "Set LOAFER_ALLOW_REMOTE=1 if you really mean it."
    )


def _request(url: str, payload: dict | None = None, *, timeout: float = 180.0) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
        return json.loads(body) if body.strip() else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise BackendError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise BackendError(f"cannot reach {url}: {exc.reason}") from exc
    except (http.client.HTTPException, OSError) as exc:
        raise BackendError(f"cannot reach {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise BackendError(f"bad JSON from {url}: {exc}") from exc


class OllamaBackend:
    """Ollama - open-source model runner, MIT licensed."""

    name = "ollama"

    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    def models(self) -> list[str]:
        data = _request(f"{self.host}/api/tags", timeout=5)
        return [m.get("name", "") for m in data.get("models", [])]

    def available(self) -> bool:
        try:
            names = self.models()
        except BackendError:
            return False
        if not names:
            return False
        if self.model in ("", "auto"):
            return True
        return any(n == self.model or n.split(":")[0] == self.model.split(":")[0] for n in names)

    def pick_model(self) -> str:
        """The configured model if it is installed, else the best installed one.

        A preset is a preference, not a requirement. If it is not on disk we use the
        closest thing that is, so the tool still works on a machine with less VRAM.
        """
        names = self.models()
        if not names:
            return self.model
        if self.model and self.model != "auto":
            for name in names:
                if name == self.model:
                    return name
        for preferred in list(MODEL_PREFERENCE) + ([self.model] if self.model else []):
            for name in names:
                if name == preferred or name.split(":")[0] == preferred.split(":")[0]:
                    return name
        return names[0]

    def chat(self, system: str, user: str, *, images: list[str] | None = None,
             max_tokens: int = 400, temperature: float = 0.2) -> str:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if images:
            payload["images"] = images
        try:
            data = _request(f"{self.host}/api/chat", payload)
        except BackendError as exc:
            if "think" in str(exc):
                payload.pop("think", None)
                data = _request(f"{self.host}/api/chat", payload)
            else:
                raise
        return (data.get("message") or {}).get("content", "")


class LlamaCppBackend:
    """llama.cpp server - OpenAI-compatible endpoint in front of a GGUF file."""

    name = "llamacpp"

    def __init__(self, base_url: str, model: str):
        self.base = base_url.rstrip("/")
        self.model = model or "local"

    def models(self) -> list[str]:
        data = _request(f"{self.base}/v1/models", timeout=5)
        return [m.get("id", "") for m in data.get("data", [])]

    def available(self) -> bool:
        try:
            return bool(self.models())
        except BackendError:
            return False

    def pick_model(self) -> str:
        names = self.models()
        return names[0] if names else self.model

    def chat(self, system: str, user: str, *, images: list[str] | None = None,
             max_tokens: int = 400, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if images:
            messages[-1]["content"] = [{"type": "text", "text": user}, *images]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = _request(f"{self.base}/v1/chat/completions", payload)
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content", "")


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model reply. None if there is none."""
    if not text:
        return None
    blob = text.strip()
    if "```" in blob:
        for chunk in blob.split("```"):
            chunk = chunk.strip()
            if chunk.startswith("json"):
                blob = chunk[4:].strip()
                break
    start = blob.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(blob[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(blob[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def normalize(data: dict, source: str, kind: str = "llm") -> dict:
    def as_list(value) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    return {
        "source": source,
        "summary": str(data.get("summary", "")).strip(),
        "actions": as_list(data.get("actions")),
        "topics": as_list(data.get("topics")),
        "kind": str(data.get("kind", kind)) or kind,
    }


class Analyzer:
    """Turns text into {summary, actions, topics}."""

    name = "none"

    def analyze(self, text: str, source: str) -> dict:
        raise NotImplementedError

    def merge(self, results: list[dict]) -> dict:
        raise NotImplementedError

    def answer(self, prompt: str) -> str:
        """Answer a question from retrieved excerpts. Empty string means no answer."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


class LLMAnalyzer(Analyzer):
    name = "llm"

    def __init__(self, backend, model: str, temperature: float = 0.2, max_tokens: int = 400):
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def describe(self) -> str:
        return f"{self.backend.name}:{self.model}"

    def analyze(self, text: str, source: str) -> dict:
        raw = self.backend.chat(
            SYSTEM_PROMPT,
            PER_FILE_PROMPT.safe_substitute(source=source, text=text),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        data = extract_json(raw)
        if data is None:
            # ponytail: model drifted out of format, fall back to extractive instead of crashing
            return fallback.analyze(text, source)
        result = normalize(data, source)
        if not result["summary"].strip():
            result["summary"] = fallback.extractive_summary(text)
        return result

    def merge(self, results: list[dict]) -> dict:
        if not results:
            return {"source": "digest", "summary": "", "actions": [], "topics": [], "kind": "digest"}
        blocks = "\n\n".join(
            f"### {r.get('source', '?')}\n"
            f"summary: {r.get('summary', '')}\n"
            f"actions: {', '.join(r.get('actions', [])) or 'none'}\n"
            f"topics: {', '.join(r.get('topics', [])) or 'none'}"
            for r in results
        )
        raw = self.backend.chat(
            SYSTEM_PROMPT,
            MERGE_PROMPT.safe_substitute(blocks=blocks),
            max_tokens=self.max_tokens + 100,
            temperature=self.temperature,
        )
        data = extract_json(raw)
        if data is None:
            return fallback.merge(results)
        merged = normalize(data, "digest", kind="digest")
        # keep every action the per-thread pass found, model merge only adds meaning
        thread_actions = [a for r in results for a in r.get("actions", [])]
        merged["actions"] = fallback.dedupe(thread_actions + merged["actions"])
        return merged


    def answer(self, prompt: str) -> str:
        raw = self.backend.chat(
            ANSWER_SYSTEM,
            prompt,
            max_tokens=self.max_tokens,
            temperature=0.1,
        )
        return raw.strip()


class HeuristicAnalyzer(Analyzer):
    name = "heuristic"

    def analyze(self, text: str, source: str) -> dict:
        return fallback.analyze(text, source)

    def merge(self, results: list[dict]) -> dict:
        return fallback.merge(results)

    def answer(self, prompt: str) -> str:
        # no model, no answer: the caller falls back to showing the excerpts
        return ""


def pick_analyzer(cfg: Config) -> Analyzer:
    """auto -> ollama -> llamacpp -> heuristic. Never raises, never blocks on the network for long."""
    choice = (cfg.backend or "auto").lower()
    if choice == "heuristic":
        return HeuristicAnalyzer()

    if choice in ("auto", "ollama"):
        ensure_local(cfg.ollama_host)
        backend = OllamaBackend(cfg.ollama_host, cfg.model)
        if backend.available():
            model = backend.pick_model()
            backend.model = model
            return LLMAnalyzer(backend, model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        if choice == "ollama":
            raise BackendError(f"no ollama model reachable at {cfg.ollama_host}")

    if choice in ("auto", "llamacpp"):
        ensure_local(cfg.llamacpp_url)
        backend = LlamaCppBackend(cfg.llamacpp_url, cfg.model)
        if backend.available():
            model = backend.pick_model()
            backend.model = model
            return LLMAnalyzer(backend, model, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
        if choice == "llamacpp":
            raise BackendError(f"no llama.cpp server at {cfg.llamacpp_url}")

    return HeuristicAnalyzer()
