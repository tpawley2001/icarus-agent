# What breaks when you build an agent for local models

Icarus is a CLI coding agent that runs entirely against models on your own
machine. The architecture is unremarkable — a tool registry, an agent loop,
progressive-disclosure skills, persistent sessions. That part is well-trodden.

What is *not* well-trodden is everything that stops working when the model
behind the loop is an 8B GGUF on a consumer GPU instead of a hosted frontier
model. This is a writeup of the failures found while building it, and the
measurements used to fix them. Every number here was measured on a 2×RTX 3060
box (8 GB + 12 GB) running llama-swap in front of llama.cpp.

---

## 1. Models leak their tool-call syntax as plain text

An agent needs tool calls. The OpenAI API has a `tools` parameter, llama.cpp
supports it, and for models with a tool-aware chat template it works fine.

The problem is the models *without* one. Handing them a `tools` array produces
either a 500 or confident prose describing the tool call it would like to make.
So Icarus probes each model once, caches the result, and falls back to a
documented protocol in the system prompt: reply with a fenced JSON block.

That fallback failed on the first live test, and the failure was informative.
Gemma returned:

```
<|tool_call>call:read_file{path:<|"|>notes.txt<|"|>}<tool_call|>
```

That is Gemma's *native* tool-call format, emitted as literal text. llama.cpp
only activates the matching parser when you pass `tools` — and we deliberately
had not, because we were using the text protocol. So the model fell back to
what it was trained on, and nothing was there to read it.

Every family leaks a different shape:

| Family | Emitted form |
|---|---|
| Qwen / Hermes | `<tool_call>{"name":…,"arguments":{…}}</tool_call>` |
| Gemma | `<\|tool_call>call:name{key:<\|"\|>value<\|"\|>}<tool_call\|>` |
| Llama 3.1 | `<function=name>{…}</function>` |
| Mistral | `[TOOL_CALLS] [{"name":…}]` |

Note Gemma's is not even JSON: unquoted keys, and `<|"|>` as a quote delimiter.

**The fix**: parse all of them, plus the documented fenced form. A fallback that
only understands its own syntax is useless on precisely the models that need a
fallback. There is also a runtime downgrade — if a model advertised as
tool-capable starts emitting text calls, Icarus notices mid-turn, switches mode,
and persists that to the capability cache.

---

## 2. Reasoning models return empty replies

Several models served with `--jinja` return **empty** `content` unless thinking
is explicitly disabled — the entire response goes into a reasoning channel the
client never sees. You get a successful HTTP 200 with nothing in it.

Different builds honour different kill-switches, so Icarus sends all of them
(`enable_thinking`, `chat_template_kwargs`, `reasoning_effort`), strips `<think>`
blocks that leak through anyway, and suppresses them *during* streaming so you
don't watch reasoning scroll past.

The subtlety: some models return nothing **with thinking off**. For those,
obeying the user's "thinking off" preference produces blank replies. Icarus
detects this at probe time and forces thinking back on for that model, saying
so rather than silently disagreeing.

---

## 3. Context is the binding constraint, and the math is not obvious

Hosted agents assume 200K tokens. Here, 8K is common and 32K is a luxury.
llama.cpp allocates the KV cache **up front at server launch**, so the window
is fixed until you relaunch — and its cost varies enormously.

Measured KV cost per token across models on this box:

| Model | KV per token | at 32K |
|---|---|---|
| qwen3.6:35b-a3b | 80 KB | 2.5 GB |
| qwen3:latest | 144 KB | 4.5 GB |
| gemma4:12b (naive) | 1,536 KB | 48 GB |

A **19× spread**. Bumping the third one blindly is how you OOM-thrash a
machine. So Icarus reads the GGUF header directly and costs every proposed
change against the model's real geometry.

### Sliding-window attention changes the answer by 5.8×

The first version of the planner reported "largest safe context: 0" for
`gemma4:12b` — a model that demonstrably runs at 16K. The naive calculation
(all layers × context) is wrong for Gemma, which uses interleaved local/global
attention: 5 of every 6 layers keep only a fixed window of KV, and use smaller
head dimensions while doing it (`key_length_swa` 256 vs `key_length` 512).

The GGUF header carries what's needed — `attention.sliding_window`,
`key_length_swa`, `value_length_swa`, and `shared_kv_layers` (Gemma 3n shares KV
across 18 of its 42 layers, which allocate nothing).

Costing global and windowed layers separately:

```
per-token   = n_global × n_head_kv × (k_len + v_len) × 2
fixed       = n_local  × n_head_kv × (k_swa + v_swa) × window × 2
```

For gemma4:e4b that is 16 KB/token instead of 168 KB/token — **5.8× lower**.
Without this correction the planner refuses resizes that are perfectly safe on
half the models in a typical library.

### Validation

Predicted KV growth for an 8K→32K change on gemma4:e4b: **384 MiB**.
Measured, by unloading, resizing, reloading and reading `nvidia-smi`: **384 MiB**.
Zero error.

### File size is not weight footprint

`gemma4:e4b` is a 5.03 GB file that occupies **3.7 GB** of VRAM — llama.cpp
keeps large embedding tables on the host. Planning against file size
over-estimates by ~40%, which is safe but wastes headroom. Icarus measures the
real figure whenever a model happens to be loaded and plans against it
thereafter.

### "Free VRAM" is the wrong baseline

`nvidia-smi` free memory changes minute to minute, because llama-swap evicts
models on a TTL. Another llama-server's allocation is *reclaimable* — it will
be evicted to make room. What is not reclaimable is everything else sharing the
card: an image-generation server, a TTS server, a video transcode.

Budgeting against **total minus non-evictable** makes the answer stable.

One trap: a llama-server mid-teardown still appears in `nvidia-smi` after its
PID is gone. Treating an unknown PID as non-evictable zeroed the budget and
blocked valid resizes. Unknown PIDs are reclaimable — a process that no longer
exists is not holding memory for long.

---

## 4. Token counting without a tokenizer

Budgeting a turn requires knowing how many tokens the history is. Shipping a
tokenizer per model family is a lot of dependency for an estimate.

Instead: start at 3.6 characters per token, then calibrate from the
`prompt_tokens` the server reports on every reply. Within two turns the ratio
is accurate for the running model, and it re-calibrates automatically when you
switch models. A rolling 8-sample window, clamped to [1.5, 8.0] so one odd turn
can't wreck the budget.

---

## 5. Small models need smaller tool surfaces and louder tool results

An 8B model handed forty tools picks the wrong one. Icarus ships **ten**.
Anything larger belongs in a skill, loaded on demand.

Tool results matter as much as tool definitions. Asked to count lines in a
3-line file, a model called `read_file` with `limit: 1`, got one line back plus
"2 more lines", and concluded *"the file has at least 3 lines"* — then stopped
and asked permission to continue.

The fix was not a better prompt. It was making a partial read self-describing:
every result now leads with the file's true total length. Same model, same
question, after the change: *"The file has 3 lines. The last line is gamma."*

The general principle: with a small model, a tool result that is technically
complete but requires an inference to use will produce a hedge instead of an
answer.

---

## 6. Cold loads are minutes, not seconds

A short client timeout is the single most common cause of phantom failures on
the first request after a model switch — llama-swap's own health-check timeout
is 600 seconds. Icarus uses minutes-scale timeouts and shows a running clock,
because a loading 35B is indistinguishable from a hang otherwise.

Related: after writing a config change and unloading, an immediate request
returns `HTTP 500 "matrix is shutting down"`. Anything that unloads must wait
for the teardown to finish, and a one-shot retry on transient 5xx turns a hard
failure into a pause.

---

## 7. Loopback is not the same as local

Icarus' premise is that nothing leaves the machine. Autodetection scans
loopback for inference servers — and on the development box it found three: a
llama-swap, an Ollama, and a third endpoint with 99 models including
`gemini-3.5-flash`, `deepseek-v4-pro`, and a pile of `:free` suffixes.

That third one was a cloud router: an ordinary OpenAI-compatible endpoint on
127.0.0.1 that forwards every token to hosted providers. Auto-selecting it
would have quietly broken the one promise the tool makes.

Detection now flags endpoints whose model IDs look like hosted catalogues
(`:free`, `@cf/`, `openai/`, `claude-*`, `gemini*`), labels them
`CLOUD ROUTER — not local`, and never selects them implicitly.

---

## The general lesson

Most of these are not "the model is dumber" problems. They are **interface**
problems: fixed-at-launch context, inconsistent tool-call syntax, reasoning
hidden behind template flags, allocation costs that vary 19× between models
you might switch between mid-conversation.

A hosted API hides all of it. Building against local models means the agent has
to know what it is actually running on — which is why Icarus reads GGUF headers
and llama-swap's launch commands instead of assuming.
