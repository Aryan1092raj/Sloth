# Loafer

**A local sloth that reads your chats while you sleep.**

You have 400 unread messages across three group chats and two email threads. Somewhere in
there are the two things you promised to do by Friday. Loafer is a CLI that reads that pile
of exports overnight with an open-weight model running on your own laptop, and hands you a
two-line summary plus a checklist in the morning.

It is named after the animal that does nothing all day and somehow gets the job done.

---

## What it does

```bash
loafer digest --input ~/exports/chat    # read a folder of chat/email exports, write one digest
loafer watch  --input ~/exports/chat    # keep one digest file for the whole night, wake on new mail
loafer todo                             # what is still open in the latest digest
loafer done "renew domain"              # tick an item off, and remember that you did
loafer ask "who owes me the invoice?"    # ask a question about your chats, get sources back
loafer nag                              # the promises that keep coming back
loafer history "invoice"                # search every digest you have ever written
loafer grocery fridge.jpg                # photo of the fridge -> shopping list of what is missing
loafer init                             # create the folders and the config file
loafer doctor                           # what is running locally, and is it reachable
```

No web UI. No account. No API key. No pip install needed - the whole thing is the Python 3
standard library.

## Why open-source AI, for this project

The inputs here are the most private text a person owns: chat logs with friends, email
threads with their manager, a photo of the inside of their fridge. Everything a cloud
assistant needs to summarise that is exactly what you would not want to upload.

| Question | Answer |
| --- | --- |
| Runs on a laptop with no internet? | Yes. After the model is pulled once, `loafer` never touches the network. It talks to `localhost`. |
| Keeps data off a server you do not control? | Yes. Chat exports and photos are read from disk, sent to a local model, and never leave the machine. No telemetry, no analytics, no account. |
| Can you swap models? | Yes. `LOAFER_MODEL=llama3.2:1b`, or point it at any llama.cpp server with a GGUF file. The model is a swappable component, not a hard dependency. |
| Can you fine-tune it? | Yes. Serve your own fine-tuned weights with Ollama or llama.cpp and point `LOAFER_MODEL` at them. A digest that knows your group's shorthand is a fine-tune away. |
| Does it cost anything to run? | Nothing. No API keys, no tokens, no quota. 20 exports take about 12 seconds on a laptop GPU. |
| Does it ever download anything? | Never. No auto-pull, no model registry calls, no telemetry. Loafer refuses to talk to any host except localhost unless you set `LOAFER_ALLOW_REMOTE=1`. |
| Where did the open approach beat a closed one? | A closed API needs your private logs uploaded, needs a paid key, needs to be online exactly when you want to review your morning digest, and pins you to one model's judgement. A local open-weight model can be wrong in exactly the way you can inspect, retune, or replace. |

There is a second reason, and it is the reason this project exists at all: a 0.6B open model
is small enough that the boring part of the job - reading, summarising, listing what was
promised - can be done *for free, offline, every night*. The quality bar for that job is
lower than people assume, and the privacy bar is higher than most tools accept.

## Install

Requirements: Python 3.9+ and a local model runtime. [Ollama](https://ollama.com) (MIT) is
the easy one; [llama.cpp](https://github.com/ggml-org/llama.cpp) works too.

```bash
git clone https://github.com/Aryan1092raj/Sloth.git
cd Sloth

ollama serve                       # in one terminal
ollama pull qwen3:0.6b             # 522 MB, Apache-2.0 weights
python3 -m loafer doctor           # check the wiring
```

No dependencies to install. `python3 -m loafer` is the entry point. If you would rather have
a `loafer` command on your PATH:

```bash
pipx install .                     # or: pip install .
```

**Loafer never downloads anything.** There is no auto-pull, no model registry call, no
telemetry, no update check. You choose the model and pull it yourself; Loafer uses what is
on disk and tells you if it is missing. The only host it will talk to is `localhost`, and
it refuses anything else unless you set `LOAFER_ALLOW_REMOTE=1`:

```bash
$ LOAFER_OLLAMA_HOST=https://api.openai.com loafer digest --input exports
error: refusing to send your data to 'api.openai.com'. Loafer only talks to localhost.
```

For the vision command you also want a vision model:

```bash
ollama pull qwen2.5vl:3b           # 3.2 GB, or use a 7B+ model for better results
```

## Use it

```bash
# one shot over a folder of exports
python3 -m loafer digest --input examples --out ~/loafer/digests

# only what landed since you went to bed
python3 -m loafer digest --input ~/exports/chat --since 8h
python3 -m loafer digest --input ~/exports/chat --since yesterday

# trade speed for care
python3 -m loafer digest --input ~/exports/chat --quality deep

# leave it running overnight, it only reads new or changed files
python3 -m loafer watch --input ~/exports/chat --out ~/loafer/digests --interval 300

# photo of the fridge -> what to buy
python3 -m loafer grocery fridge.jpg

# close the loop: what is still open, then tick it off
python3 -m loafer todo
python3 -m loafer done "renew domain"

# send the open items to whatever todo tool you use
python3 -m loafer todo --export ~/todo.md

# look back
python3 -m loafer history
python3 -m loafer history "invoice"
```

`--since` takes a duration (`30m`, `6h`, `2d`), a word (`today`, `yesterday`, `now`) or a
date (`2026-10-19`). It filters on file mtime, so it works on any export folder.

Real output, generated by this repo's `examples/` on an RTX 4050 laptop with `qwen3:0.6b`:

```markdown
# Loafer digest - 2026-10-20 22:17

## Two lines

Priya is confirming the finance approval for the October invoice pending and the board pack needs Q3 numbers by Wednesday.
The template is shared, so the user should use it as instructed.

## To do

- [ ] confirm finance approval (email-thread.txt)
- [ ] send Q3 numbers by Wednesday (email-thread.txt)
- [ ] check availability (team-chat.txt)
- [ ] book table (team-chat.txt)
- [ ] renew domain (team-chat.txt)

## Threads

### email-thread.txt

Priya is confirming the finance approval for the October invoice pending and the board pack
needs Q3 numbers by Wednesday. The template is shared, so the user should use it as instructed.

### team-chat.txt

Ana and Ben are discussing the payment window and finance approval, while Carla updates the
venue and Dan books a table. Ana also mentions renewing the domain, and Dan confirms the
renewal will be handled this week.

- [ ] send Q3 numbers

Topics: finance, board pack
```

`watch` writes the same file shape to `~/loafer/digests/YYYY-MM-DD.md` and keeps rewriting
it, so one file at 7am holds everything that landed overnight. On Linux it uses inotify, so
it reacts to a new export in under a second instead of waiting out the interval; anywhere
else it falls back to polling.

### The loop

A checklist is only useful if you can act on it, so the digest doubles as a todo list:

```bash
$ loafer todo
open items in ~/loafer/digests/2026-10-20.md
- [ ] confirm finance approval (email-thread.txt)
- [ ] send Q3 numbers by Wednesday (email-thread.txt)
- [ ] book table (team-chat.txt)

$ loafer done "book table"
ticked 1 item(s) in ~/loafer/digests/2026-10-20.md
  book table
```

Ticks live in `~/loafer/digests/.loafer-done.json`, not in the digest, because `watch`
rewrites the daily digest whenever new mail lands and would wipe them.

### Ask your chats

Retrieval is plain BM25 over windows of text, computed with the standard library. No
embeddings, no vector database, no index to keep in sync. The local model reads the top
windows and answers, and every answer comes with its sources:

```bash
$ loafer ask "what did we promise about the domain and the invoice?"
The answer is that we promised to renew the domain and handle the invoice renewal this
week. The source files are team-chat.txt.

Sources:
  team-chat.txt:9
  email-thread.txt:1
```

With no model reachable, the same command prints the matching excerpts instead of an
answer. Retrieval still works, nothing is invented.

### The sloth nags you

Promises that keep coming back are the real problem. `loafer nag` compares every digest and
tells you what has been asked for more than once:

```bash
$ loafer nag
renew the domain
    3 day(s), first 2026-10-18, last 2026-10-20  [team-chat.txt]
send Q3 numbers by Wednesday
    3 day(s), first 2026-10-18, last 2026-10-20  [email-thread.txt]
```

### Paste it into Slack

```bash
$ loafer digest --input exports --format slack
*Loafer digest - 2026-10-20 23:41*
*Two lines*
Priya is informing Ben about the finance approval pending and the board pack deadline.
*To do*
- [ ] follow up with finance approval (email-thread.txt)
```

### Search your own history

Every digest is a markdown file, so every digest is searchable:

```bash
$ loafer history
2026-10-20.md                     2026-10-20 23:33  11 open
digest-2026-10-19-0812.md         2026-10-19 08:12   4 open

$ loafer history "invoice"
digest-2026-10-19-0812.md:19: Priya is confirming the finance approval for the October invoice.
1 match(es)
```

No vector database, no embeddings, no index to maintain. `grep` would do the same thing;
this just knows where your digests live.

## How it works

```
  exports/*.txt
        |
        v
   collect.py  -- find text files, skip hidden dirs, skip the output dir, cap the size,
        |        filter by --since on mtime
        v
   digest.py   -- split long files into overlapping chunks, one prompt per chunk
        |
        v
   llm.py      -- one JSON-only prompt per piece, then one merge prompt
        |         (Ollama / llama.cpp on localhost)
        |         |  malformed reply?
        |         v
        |     fallback.py -- stdlib extractive summary + cue-based action detection
        v
   digest.py   -- two-line summary, checklist with sources, per-thread breakdown -> markdown
        |
        v
   ~/loafer/digests/2026-10-20.md
```

Four rules the code follows:

1. **Degrade, never crash.** If the model is unreachable, the backend is a pure-stdlib
   extractive summariser. If the model replies with prose instead of JSON, the parser
   falls back the same way. You always get a digest.
2. **The model is a component, not a dependency.** `pick_analyzer()` tries Ollama, then any
   llama.cpp OpenAI-compatible server, then the stdlib fallback. `LLMAnalyzer` and
   `HeuristicAnalyzer` implement the same four methods.
3. **Nothing is silently dropped.** A long export is split into overlapping chunks and every
   chunk is analysed, instead of truncating at the first 6000 characters.
4. **Local only.** Loafer refuses any host that is not localhost, and it never downloads
   anything - no model pull, no registry call, no telemetry.
5. **No dependencies.** Standard library only: `urllib`, `json`, `pathlib`, `argparse`,
   `re`, `collections`, `math`, `select`, `ctypes`. Nothing to `pip install`, nothing to
   break on a new laptop.
6. **Boring on purpose.** JSON files instead of a database, one file per day instead of a
   feed, BM25 instead of a vector store, `grep` instead of a search index.

### Quality presets

A preset picks a model, a temperature and a token budget. The model is a *preference*: if
it is not installed locally, Loafer uses the closest model that is, so the command still
works on a machine with less VRAM.

| Preset | Prefers | Tokens | Chunking | For |
| --- | --- | --- | --- | --- |
| `fast` | `qwen3:0.6b` | 300 | no | a quick look, CPU only, huge inboxes |
| `balanced` (default) | `qwen3:1.7b` | 400 | yes | the normal case |
| `deep` | `qwen3:4b` | 800 | yes | slow, careful, when the wording matters |

```bash
loafer digest --input exports --quality deep
LOAFER_QUALITY=fast loafer watch --input exports
```

### Configuration

Precedence: CLI flag, then environment variable, then `~/.config/loafer/config.json`,
then the default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `LOAFER_INPUT` | `~/loafer/inbox` | folder to read |
| `LOAFER_OUT` | `~/loafer/digests` | where digests are written |
| `LOAFER_MODEL` | from the preset | model tag, or llama.cpp model id |
| `LOAFER_QUALITY` | `balanced` | `fast`, `balanced`, `deep` |
| `LOAFER_BACKEND` | `auto` | `auto`, `ollama`, `llamacpp`, `heuristic` |
| `LOAFER_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama endpoint |
| `LOAFER_LLAMACPP_URL` | `http://127.0.0.1:8080/v1` | llama.cpp endpoint |
| `LOAFER_VL_MODEL` | `qwen2.5vl:3b` | vision model for `grocery` |
| `LOAFER_STAPLES` | 12 staples | comma separated list for `grocery` |
| `LOAFER_INTERVAL` | `300` | seconds between polls in `watch` |
| `LOAFER_SINCE` | none | ignore anything older than this |
| `LOAFER_ALLOW_REMOTE` | unset | allow a non-localhost model host (off by default) |

`loafer doctor --json` prints the same picture for scripts.

### Swapping the model

```bash
# a different Ollama model
LOAFER_MODEL=llama3.2:1b python3 -m loafer digest --input exports

# any GGUF served by llama.cpp
llama-server -m model.gguf --port 8080
LOAFER_BACKEND=llamacpp LOAFER_MODEL=local python3 -m loafer digest --input exports

# no model at all, pure stdlib
python3 -m loafer digest --backend heuristic --input exports
```

Fine-tuning works the same way: serve your fine-tuned weights and point `LOAFER_MODEL` at
them. Nothing in the code knows or cares which weights are behind the endpoint.

## Honest limits

- **A 0.6B model is lossy.** It paraphrases, it occasionally invents an action that nobody
  promised, and it will not catch a promise buried in the last three lines of a thread. The
  digest is a reading aid, not a source of truth. A 7B model is better; the prompt and the
  parsing do not change when you swap it.
- **`grocery` needs a 7B+ vision model to be trustworthy.** With a 3B vision model on a
  cluttered, sideways fridge photo, detection is unreliable. The shopping list is always
  `staples minus what the model confirmed`, and the model confirms too little, so treat the
  list as a starting point, not a verified stock take.
- **Exports, not live capture.** WhatsApp, Telegram, Slack and Discord all export text;
  Loafer reads those files. It does not talk to the apps.
- **One merge pass.** Everything is merged in a single prompt, so a very large inbox is
  merged from the 60 most recent records. Raise the cap in `digest.build_digest` if you
  want more.
- **Prompt injection.** A chat export is untrusted input. A sufficiently determined export
  could try to talk the model into ignoring the instructions. Loafer's answer is that the
  output is a reading aid and the parse is JSON-only, but this is not fully defended.

## Repo layout

```text
Sloth/
  loafer/            the package: cli, config, collect, llm, fallback, digest, watch, todo, history, ask, grocery
  tests/             46 unittest cases, stdlib only, no network
  examples/          a chat export and an email thread to try it on
  pyproject.toml     zero dependencies, installs a `loafer` command
  README.md          this file
```

The design documents (`PRD.md`, `TRD.md`, `architecture.md`) live in the working copy and are
kept out of git on purpose; this README is the published surface.

## Roadmap

- write ticked items back into a plain markdown todo file on every digest, not on demand
- an `--email` mode that reads a local maildir export instead of a folder of text files
- `nag` should also notice promises that appear in a chat but never in a digest
- optional round-trip: mark a promise as done in the source app if it exposes an API

## License

MIT.
