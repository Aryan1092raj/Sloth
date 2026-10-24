"""Tests for loafer. Standard library only, no network, no fixtures to install.

    python3 -m unittest discover -s tests -v
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loafer import ask, collect, digest, fallback, history, todo, watch  # noqa: E402
from loafer.config import load_config  # noqa: E402
from loafer.llm import HeuristicAnalyzer, LLMAnalyzer, extract_json, normalize  # noqa: E402


class FakeBackend:
    name = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, user, **kwargs):
        self.calls.append((system, user))
        return self.replies.pop(0) if self.replies else "{}"


SAMPLE = """[09:12] Ana: please chase finance today
[09:14] Ben: ok, I will ping Priya now
[09:40] Carla: the invoice must go out before Friday
"""


class TestExtractJson(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(extract_json('{"summary": "hi"}'), {"summary": "hi"})

    def test_object_inside_prose(self):
        raw = 'Sure! Here you go:\n```json\n{"summary": "ok", "actions": ["do it"]}\n```\nHope that helps.'
        self.assertEqual(extract_json(raw)["actions"], ["do it"])

    def test_nested_braces(self):
        raw = '{"summary": "a {weird} thing", "actions": []}'
        self.assertEqual(extract_json(raw)["summary"], "a {weird} thing")

    def test_no_json(self):
        self.assertIsNone(extract_json("I refuse to answer in JSON."))
        self.assertIsNone(extract_json(""))

    def test_normalize_coerces_shapes(self):
        data = normalize({"summary": " s ", "actions": "one", "topics": ["t", ""]}, "src")
        self.assertEqual(data["actions"], ["one"])
        self.assertEqual(data["topics"], ["t"])
        self.assertEqual(data["source"], "src")


class TestFallback(unittest.TestCase):
    def test_finds_actions(self):
        result = fallback.analyze(SAMPLE, "sample")
        self.assertTrue(result["actions"])
        self.assertTrue(any("chase finance" in a.lower() for a in result["actions"]))

    def test_summary_is_two_sentences_max(self):
        summary = fallback.extractive_summary(SAMPLE)
        self.assertLessEqual(len(fallback.split_sentences(summary)), 2)

    def test_merge_dedupes_actions(self):
        merged = fallback.merge([
            {"summary": "one", "actions": ["ping priya"], "topics": ["invoice"]},
            {"summary": "two", "actions": ["ping priya"], "topics": ["invoice"]},
        ])
        self.assertEqual(merged["actions"], ["ping priya"])


class TestCollect(unittest.TestCase):
    def test_skips_hidden_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "inbox"
            out = Path(tmp) / "digests"
            (root / ".hidden").mkdir(parents=True)
            (root / "a.txt").write_text("hello")
            (root / ".hidden" / "b.txt").write_text("nope")
            (root / "image.png").write_bytes(b"\x89PNG")
            out.mkdir()
            (out / "old.md").write_text("digest")

            files = collect.collect_files(root, exclude_dirs=[out])
            names = [p.name for p in files]
            self.assertIn("a.txt", names)
            self.assertNotIn("b.txt", names)
            self.assertNotIn("old.md", names)
            self.assertNotIn("image.png", names)

    def test_missing_dir_is_empty(self):
        self.assertEqual(collect.collect_files(Path("/nope/does/not/exist")), [])

    def test_truncate_marks_cut(self):
        text = collect.truncate("x" * 100, 10)
        self.assertIn("truncated", text)


class TestDigest(unittest.TestCase):
    def test_render_has_sections(self):
        text = digest.render(
            {"summary": "line one\nline two", "actions": ["pay invoice"], "topics": ["invoice"]},
            when="2026-10-20 07:00",
            mode="ollama:qwen3:0.6b",
            sources=2,
        )
        self.assertIn("## Two lines", text)
        self.assertIn("- [ ] pay invoice", text)
        self.assertIn("invoice", text)

    def test_llm_analyzer_parses_model_json(self):
        backend = FakeBackend(['{"summary": "s", "actions": ["a"], "topics": ["t"], "kind": "chat"}'])
        analyzer = LLMAnalyzer(backend, "fake:1")
        result = analyzer.analyze(SAMPLE, "sample")
        self.assertEqual(result["actions"], ["a"])
        self.assertEqual(result["kind"], "chat")

    def test_llm_analyzer_degrades_on_garbage(self):
        backend = FakeBackend(["not json at all"])
        analyzer = LLMAnalyzer(backend, "fake:1")
        result = analyzer.analyze(SAMPLE, "sample")
        self.assertTrue(result["actions"])

    def test_merge_keeps_per_thread_actions(self):
        backend = FakeBackend([
            '{"summary": "s", "actions": ["model action"], "topics": ["t"]}',
            '{"summary": "s2", "actions": ["model action"], "topics": ["t"]}',
        ])
        analyzer = LLMAnalyzer(backend, "fake:1")
        merged = analyzer.merge([
            {"source": "a", "summary": "s", "actions": ["thread action"], "topics": []},
        ])
        self.assertIn("thread action", merged["actions"])


class TestState(unittest.TestCase):
    def test_tracks_new_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp) / "inbox"
            inbox.mkdir()
            path = inbox / "a.txt"
            path.write_text("hello")
            cfg = load_config(input_dir=inbox, out_dir=Path(tmp) / "out")
            state = watch.State(cfg.state_file)
            self.assertTrue(state.is_new(path))
            state.mark(path)
            state.save()
            again = watch.State(cfg.state_file)
            self.assertFalse(again.is_new(path))


class TestSince(unittest.TestCase):
    def test_durations_and_words(self):
        now = datetime(2026, 10, 20, 23, 30)
        self.assertEqual(collect.parse_since("30m", now), datetime(2026, 10, 20, 23, 0))
        self.assertEqual(collect.parse_since("6h", now), datetime(2026, 10, 20, 17, 30))
        self.assertEqual(collect.parse_since("2d", now), datetime(2026, 10, 18, 23, 30))
        self.assertEqual(collect.parse_since("today", now), datetime(2026, 10, 20, 0, 0))
        self.assertEqual(collect.parse_since("yesterday", now), datetime(2026, 10, 19, 0, 0))
        self.assertEqual(collect.parse_since("2026-10-19", now), datetime(2026, 10, 19, 0, 0))

    def test_bad_value_raises(self):
        with self.assertRaises(ValueError):
            collect.parse_since("whenever", datetime(2026, 10, 20))

    def test_collect_respects_since(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.txt"
            new = root / "new.txt"
            old.write_text("old")
            new.write_text("new")
            stamp = datetime(2026, 10, 1).timestamp()
            os.utime(old, (stamp, stamp))
            files = collect.collect_files(root, since=datetime(2026, 10, 19))
            self.assertEqual([p.name for p in files], ["new.txt"])


class TestTodo(unittest.TestCase):
    def test_open_items_and_tick(self):
        text = "## To do\n\n- [ ] pay the invoice\n- [ ] book the table\n- [x] done already\n"
        items = todo.open_items(text)
        self.assertEqual(items, ["pay the invoice", "book the table"])
        new_text, ticked = todo.tick(text, "invoice")
        self.assertEqual(ticked, ["pay the invoice"])
        self.assertIn("x", new_text)
        self.assertNotIn("pay the invoice", todo.open_items(new_text))

    def test_tick_file_writes_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "2026-10-20.md"
            path.write_text("- [ ] renew the domain\n")
            ticked = todo.tick_file(path, "domain")
            self.assertEqual(ticked, ["renew the domain"])
            self.assertIn("x", path.read_text())

    def test_done_sidecar_survives_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-20.md").write_text("- [ ] pay the invoice\n- [ ] book the table\n")
            todo.save_done(out, ["pay the invoice"])
            # watch rewrites the digest, the sidecar still remembers
            (out / "2026-10-20.md").write_text("- [ ] pay the invoice\n- [ ] book the table\n")
            items = todo.open_items((out / "2026-10-20.md").read_text(), todo.load_done(out))
            self.assertEqual(items, ["book the table"])

    def test_latest_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-19.md").write_text("- [ ] old\n")
            (out / "2026-10-20.md").write_text("- [ ] new\n")
            self.assertEqual(todo.latest_digest(out).name, "2026-10-20.md")


class TestChunking(unittest.TestCase):
    def test_split_chunks_covers_everything(self):
        text = "".join(f"line {i} of a long conversation.\n" for i in range(400))
        chunks = digest.split_chunks(text, 1000, overlap=100)
        self.assertGreater(len(chunks), 1)
        self.assertIn("line 399", chunks[-1])
        self.assertTrue(all(chunk.strip() for chunk in chunks))

    def test_short_text_is_one_chunk(self):
        self.assertEqual(digest.split_chunks("hello", 1000), ["hello"])

    def test_merge_chunks_keeps_every_action(self):
        merged = digest.merge_chunks([
            {"source": "a.txt", "summary": "one", "actions": ["do a"], "topics": ["x"], "kind": "chat"},
            {"source": "a.txt", "summary": "two", "actions": ["do b", "do a"], "topics": ["y"], "kind": "chat"},
        ])
        self.assertEqual(merged["actions"], ["do a", "do b"])
        self.assertIn("one", merged["summary"])
        self.assertEqual(merged["source"], "a.txt")

    def test_long_file_is_not_truncated_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.txt"
            path.write_text("please renew the domain\n" * 300)
            cfg = load_config(input_dir=Path(tmp), out_dir=Path(tmp) / "out", max_chars=2000)
            results = digest.analyze_files(cfg, [path], HeuristicAnalyzer())
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["actions"])


class TestChecklist(unittest.TestCase):
    def test_actions_carry_their_source(self):
        rows = digest.checklist([
            {
                "source": "/tmp/team-chat.txt",
                "actions": ["renew the domain"],
                "action_lines": {"renew the domain": 9},
            },
        ], [])
        self.assertEqual(rows, ["renew the domain (team-chat.txt:L9)"])

    def test_action_line_finds_the_source_sentence(self):
        line = digest._action_line(
            "Ana: please chase finance today\nBen: I will ping Priya now\n",
            "chase finance",
        )
        self.assertEqual(line, 1)

    def test_action_line_ignores_weak_matches(self):
        self.assertIsNone(digest._action_line("hello there\n", "send the report"))

    def test_specific_action_wins_over_vague_one(self):
        rows = digest.checklist(
            [{"source": "/tmp/a.txt", "actions": ["send Q3 numbers"]}],
            ["send Q3 numbers to the board by Wednesday"],
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("by Wednesday", rows[0])


class TestHistory(unittest.TestCase):
    def test_search_finds_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-20.md").write_text("# digest\n- [ ] renew the domain\n")
            matches = history.search(out, "renew")
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0][1], 2)

    def test_search_misses_are_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-20.md").write_text("# digest\n")
            self.assertEqual(history.search(out, "nothing here"), [])

    def test_inventory_counts_open_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-20.md").write_text("- [ ] one\n- [ ] two\n")
            records = history.inventory(out)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["open"], 2)


class TestTodoExport(unittest.TestCase):
    def test_export_writes_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-20.md").write_text("- [ ] renew the domain\n")
            target = Path(tmp) / "todo.md"
            todo.export_open_items(out, target)
            text = target.read_text()
            self.assertIn("renew the domain", text)
            self.assertIn("# Todo", text)


class TestPresets(unittest.TestCase):
    def test_quality_presets_change_the_model_and_budget(self):
        fast = load_config(quality="fast")
        deep = load_config(quality="deep")
        self.assertNotEqual(fast.model, deep.model)
        self.assertGreater(deep.max_tokens, fast.max_tokens)

    def test_explicit_model_beats_the_preset(self):
        cfg = load_config(quality="deep", model="qwen2.5vl:3b")
        self.assertEqual(cfg.model, "qwen2.5vl:3b")


class TestLocalOnly(unittest.TestCase):
    def test_local_hosts_are_allowed(self):
        from loafer.llm import ensure_local
        ensure_local("http://127.0.0.1:11434")
        ensure_local("http://localhost:8080/v1")
        ensure_local("http://127.5.5.5:11434")

    def test_remote_hosts_are_refused(self):
        from loafer.llm import BackendError, ensure_local
        with self.assertRaises(BackendError):
            ensure_local("https://api.openai.com")
        with self.assertRaises(BackendError):
            ensure_local("https://huggingface.co")

    def test_remote_can_be_allowed_on_purpose(self):
        import os
        from loafer.llm import ensure_local
        old = os.environ.get("LOAFER_ALLOW_REMOTE")
        os.environ["LOAFER_ALLOW_REMOTE"] = "1"
        try:
            ensure_local("https://api.example.com")
        finally:
            if old is None:
                os.environ.pop("LOAFER_ALLOW_REMOTE", None)
            else:
                os.environ["LOAFER_ALLOW_REMOTE"] = old


class TestAsk(unittest.TestCase):
    def test_windows_cover_the_document(self):
        text = "".join(f"line {i} of a conversation\n" for i in range(60))
        found = ask.windows(text)
        self.assertGreater(len(found), 1)
        self.assertEqual(found[0][0], 1)
        self.assertIn("line 59", found[-1][1])

    def test_windows_terminate_on_a_short_document(self):
        self.assertEqual(len(ask.windows("one line")), 1)

    def test_rank_finds_the_relevant_chunk(self):
        docs = [
            (Path("a.txt"), 1, "we need to renew the domain before it expires"),
            (Path("b.txt"), 1, "the weather is nice today"),
        ]
        hits = ask.rank("renew the domain", docs)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0].name, "a.txt")

    def test_rank_with_nothing_to_find(self):
        self.assertEqual(ask.rank("", [(Path("a.txt"), 1, "text")]), [])

    def test_heuristic_analyzer_has_no_answer(self):
        self.assertEqual(HeuristicAnalyzer().answer("anything"), "")

    def test_refusals_are_detected(self):
        self.assertTrue(ask.is_refusal("The excerpts do not contain the answer."))
        self.assertTrue(ask.is_refusal("there is no mention of that"))
        self.assertFalse(ask.is_refusal("We promised to renew the domain."))


class TestRecurring(unittest.TestCase):
    def test_promises_that_keep_coming_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for day in ("2026-10-18", "2026-10-19", "2026-10-20"):
                (out / f"{day}.md").write_text("- [ ] renew the domain (team-chat.txt)\n")
            (out / "2026-10-20.md").write_text(
                "- [ ] renew the domain (team-chat.txt)\n- [ ] one off (chat.txt)\n"
            )
            rows = history.recurring(out, min_appearances=2)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["item"], "renew the domain")
            self.assertEqual(rows[0]["days"], 3)
            self.assertEqual(rows[0]["files"], ["team-chat.txt"])

    def test_nothing_recurring(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "2026-10-20.md").write_text("- [ ] one off (chat.txt)\n")
            self.assertEqual(history.recurring(out, min_appearances=2), [])


class TestSlackFormat(unittest.TestCase):
    def test_headings_become_bold(self):
        text = digest.to_slack("# Title\n## Section\n### Sub\n- item")
        self.assertIn("*Title*", text)
        self.assertIn("*Section*", text)
        self.assertIn("*Sub*", text)
        self.assertNotIn("##", text)


if __name__ == "__main__":
    unittest.main()
