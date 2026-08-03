# Icarus

A command-line agent that runs entirely on your own machine, against your own
models, served by [llama-swap](https://github.com/mostlygeek/llama-swap).

Drawn from hermes' architecture — tool registry, progressive-disclosure skills,
persistent sessions, todo/memory tools — but rebuilt around the constraints that
actually bite when the model is local: small context windows, inconsistent tool
support, cold model loads, and finite VRAM.

**Nothing leaves this machine.** The only network destination Icarus contacts is
the llama-swap base URL, which defaults to `127.0.0.1`. There is no account, no
telemetry, and no cloud fallback.

```
icarus                          # interactive
icarus "summarize this repo"    # one-shot
git log --oneline | icarus "what changed this week?"
```

## Install

```bash
git clone https://github.com/tpawley2001/icarus-agent.git
cd icarus
./install.sh
```

No venv, no build step, no root. Python 3.9+ and PyYAML are the only
requirements — everything else is standard library.

The installer **autodetects the inference stack you already run**, writes a
config pointing at it, links `icarus` onto your PATH, and verifies by listing
your models:

```
Inference servers
  llama-swap at 127.0.0.1:9292        31 models   full support ← will use
      config: ~/llama-swap/config.yaml
  ollama at 127.0.0.1:11434           53 models   basic (OpenAI-compatible)
  cloud-router at 127.0.0.1:5001      99 models   CLOUD ROUTER — not local

Hardware
  cuda: NVIDIA GeForce RTX 3060, 8192 MiB
  cuda: NVIDIA GeForce RTX 3060, 12288 MiB
```

Detection is signature-based rather than port-based — several of these products
share default ports and people move them, so each candidate is asked something
only one product answers a particular way (`/running` for llama-swap,
`/api/tags` for Ollama, `/props` for a bare llama.cpp server).

Recognised: **llama-swap**, **Ollama**, **LM Studio**, **vLLM**, **llama.cpp
server**, **KoboldCpp**, **Jan**, **LocalAI**, **text-generation-webui**,
**TabbyAPI**, and any other OpenAI-compatible endpoint. llama-swap gets the full
feature set (`/ctx` resizing, VRAM planning, exact context windows); everything
else works as a plain endpoint.

**Loopback is not the same as local.** A cloud router presents an ordinary
OpenAI endpoint on `127.0.0.1` and forwards every token to hosted providers.
Icarus flags those from their model IDs and never selects one implicitly —
otherwise autodetection would quietly break the one promise this tool makes.

```bash
./install.sh --detect-only    # just show what's here
./install.sh --yes            # no prompts
./install.sh --prefix ~/bin   # where to link the launcher
icarus --detect               # re-run detection later
```

## Why it's built for local models

**Context windows are the binding constraint.** Icarus reads llama-swap's own
config to learn each model's real `--ctx-size` *before* loading it, budgets
every turn against that number, and compacts when it approaches the ceiling —
squeezing old tool output first, then summarizing, then dropping. It calibrates
its own chars-per-token ratio from the `prompt_tokens` the server reports, so
budgeting gets more accurate as a session goes on rather than relying on a
guess.

**Tool calling is inconsistent.** Icarus probes each model once and caches the
result. Models with native tool support get an OpenAI `tools` array. Models
without get a documented fenced-block protocol described in the system prompt.
Either way the agent loop sees identical data.

That fallback parses the formats models *actually* emit, not just the one it
asked for — a model with a tool template but no `tools` array will happily leak
its trained special tokens as plain text, differently per family:

| Family | Leaked form |
|---|---|
| Icarus protocol | ` ```icarus {"tool":…,"args":{…}} ``` ` |
| Qwen / Hermes | `<tool_call>{"name":…}</tool_call>` |
| Gemma | `<\|tool_call>call:name{key:<\|"\|>val<\|"\|>}<tool_call\|>` |
| Llama 3.1 | `<function=name>{…}</function>` |
| Mistral | `[TOOL_CALLS] [{…}]` |

If a model advertised as native starts emitting text calls, Icarus notices
mid-turn, downgrades it, and persists that to the capability cache.

**Reasoning models hide their answers.** Served with `--jinja`, several models
return *empty* content unless thinking is explicitly disabled. Icarus sends every
known kill-switch, strips `<think>` blocks that leak through, and suppresses them
during streaming so you never watch reasoning scroll past. Models that return
nothing with thinking off are detected at probe time and have it forced back on.

**Cold loads are slow.** Timeouts are minutes, not seconds, and the spinner shows
a running clock so a loading 35B doesn't look like a hang.

**VRAM is finite.** Switching models unloads the previous one first, avoiding the
moment of double residency that OOMs the big models on a two-GPU box.

## Switching models mid-conversation

`/model` works like Claude Code's: the conversation survives the switch.

```
› /model                    # interactive picker
› /model gemma4:e4b         # by name
› /model 7                  # by number from /models
› /model qwen               # ambiguous -> lists the 9 candidates
```

On switch Icarus unloads the old model, re-probes capabilities (which can flip
tool-calling mode), re-reads the context window, and — if the new model's window
is *smaller* than the conversation needs — compacts history to fit before
continuing. `/cost` shows the full model history for the session.

## Controlling reasoning

```
› /think          # report current state
› /think off      # fastest, shortest output (default)
› /think on       # let the model reason
› /think auto     # follow what was detected for this model
```

Also available per-run: `icarus --think on "..."`.

If a model was detected as returning empty output with thinking disabled,
Icarus keeps it on regardless and says so — silently obeying would produce blank
replies.

## Flexible context windows

llama.cpp fixes the window at server launch, so changing it means relaunching
the model. On a two-GPU box that is not a free choice — the KV cache is
allocated up front, and its cost per token varies **19x** across the models on
this machine:

| Model | KV per token | at 32K |
|---|---|---|
| qwen3.6:35b-a3b | 80 KB | 2.5 GB |
| gemma4:e4b | 16 KB | 0.5 GB |
| gemma4:12b | 1,536 KB (naive) | 48 GB |

So every change is costed against the model's real geometry, read straight from
the GGUF header, and refused if it will not fit.

```
› /ctx                  # current window, KV cost, largest that fits
› /ctx 65536            # resize (also accepts /ctx 64)
› /ctx max              # largest that fits, capped at the trained length
› /ctx auto             # right-size to what this conversation is actually using
› /ctx reset            # back to the gen_config.py default
```

### Task profiles

```
› /profile              # list
› /profile quick        # 8K ctx, 1K output, 10 steps — shell questions
› /profile balanced     # 32K, 4K output, 40 steps — the default
› /profile code         # 64K, 8K output, 60 steps, 60K tool output — multi-file edits
› /profile deep         # largest that fits, 100 steps — long investigations
```

A profile sets the server-side window *and* the client-side budget (output
reserve, iteration cap, tool-output cap) in one move.

### How the sizing is done

**KV scaling is exact.** Predicted growth for an 8K→32K change on gemma4:e4b
was 384 MiB; measured was 384 MiB. 0% error.

**Sliding-window attention is accounted for.** Gemma keeps only a fixed window
of KV on 5 of every 6 layers, so the naive all-layers figure overstates it by
~5.8x — enough to refuse resizes that are actually fine. Icarus reads
`sliding_window`, `key_length_swa` and `shared_kv_layers` and costs the global
and windowed layers separately.

**Weights are measured, not guessed.** The GGUF file size overstates the GPU
footprint because llama.cpp keeps large embedding tables on the host —
gemma4:e4b is a 5.0 GB file that occupies 3.7 GB of VRAM. Icarus measures the
real figure whenever a model happens to be loaded and plans against it
thereafter, falling back to file size (a safe over-estimate) until then.

**The budget is stable.** Free VRAM right now is the wrong baseline, because
llama-swap evicts models on demand — another llama-server's allocation is
reclaimable. Icarus budgets against total VRAM minus what cannot be evicted
(ComfyUI, the Kokoro server, a Plex transcode), so the answer doesn't change
minute to minute.

Changes are written into llama-swap's config, which `--watch-config` reloads,
and recorded in `ctx_overrides.json` beside it. `gen_config.py` reads that
sidecar, so a resize survives a regeneration; `/ctx reset` restores the original
value immediately rather than waiting for one.

## Interrupting and steering a running turn

A turn is not a dead end. While the model streams or a tool runs:

| | |
|---|---|
| **type a line + Enter** | queued and injected at the next step as steering — the work so far is kept |
| **Esc** | interrupt now: the in-flight request is dropped and a running command is killed |

Steering arrives as a normal user message (`[The user sent this while you were
working]`), so the model treats it as the latest instruction while keeping
everything it has already done. Esc keeps a partial answer in history rather
than discarding it, and marks any un-run tool calls as skipped so the next turn
doesn't assume they happened.

Esc kills the whole process group, so `sleep 300` — or a runaway build — dies
immediately instead of leaving an orphan. Verified: abort honored in 2.00s with
no leftover processes.

The typed buffer lives on a line pinned to the bottom of the screen. Every
other writer goes through the console, which clears that line first and redraws
it after, so streamed tokens and your half-typed prompt never overwrite each
other. The spinner yields the line whenever you are typing.

All of this is automatically inert when stdin or stdout is not a terminal,
which is what keeps one-shot, piped, and cron usage safe. Disable with
`ui.interrupt: false`.

## Installing skills from public repositories

Icarus' runtime is fully local. `/skills install` is the one deliberate
exception, it only runs when you type it, and it only talks to github.com —
restricted to first-party repositories (the same set hermes trusts):

```
anthropics/skills   openai/skills   huggingface/skills   NVIDIA/skills
```

That is 412 skills as of this writing.

```
› /skills search pdf          # search the trusted repos
› /skills install pdf         # ambiguous -> shows which repos have it
› /skills install skill-creator
› /skills update              # which installed skills moved upstream
› /skills remove skill-creator
› /skills sources             # what is trusted, and why
```

A skill is executable instructions handed to an agent that can run shell
commands, so **nothing is written to disk before you see it**. Files are
downloaded to memory, scanned, and the findings shown with a verdict; then you
confirm. The scan flags remote-code-execution (`curl | sh`, `eval(base64…)`),
credential access (`~/.ssh`, `.env`, embedded keys), destructive commands, and
persistence (cron, systemd). A `dangerous` verdict changes the prompt to an
explicit "install anyway?".

Archive paths that try to escape the skill directory are dropped, provenance
(repo, path, blob sha, timestamp) is recorded in `~/.icarus/skills.lock.json`,
and a freshly installed skill is available to the model immediately — no
restart.

`GITHUB_TOKEN` is used if set, purely to raise GitHub's 60-request/hour
unauthenticated rate limit.

## Commands

| | |
|---|---|
| `/model [name\|number]` | switch model in flight |
| `/models` `/running` `/unload` | inventory, what's resident, free VRAM |
| `/think [on\|off\|auto]` | reasoning output |
| `/context` `/compact` | conversation's context usage; compact now |
| `/ctx [N\|max\|auto\|reset]` | resize the model's window (reloads it) |
| `/profile [name]` | task shape: quick / balanced / code / deep |
| `/caps [--reprobe]` | what was detected about this model |
| `/tools` | what's available |
| `/skills [search\|install\|remove\|update\|sources]` | skill library + public repos |
| `/new` `/sessions` `/resume <id>` | conversations |
| `/cost` | tokens used (always $0.00) |
| `/cwd [path]` | working directory |

While a turn runs: **type + Enter** to steer, **Esc** to interrupt.

## Tools

`terminal`, `read_file`, `write_file`, `edit_file`, `list_dir`, `search_files`,
`glob_files`, `todo`, `memory`, `skill`.

Deliberately ten. An 8B model handed forty tools picks the wrong one; handed ten
it picks well. Anything larger belongs in a skill.

Destructive shell commands are gated behind an approval prompt — `rm -rf`, `dd`,
`mkfs`, `systemctl stop`, `git push`, `curl | sh`, and friends. Answer `a` to
allow that class for the rest of the session. Patterns live under
`tools.require_approval` in the config.

## Skills

A skill is a directory with a `SKILL.md` carrying YAML frontmatter:

```markdown
---
name: my-skill
description: One line telling the model when this is relevant.
---
...body...
```

Only names and descriptions go in the system prompt; bodies load on demand via
the `skill` tool. On an 8K model that distinction decides whether it works at
all. Icarus reads `~/.icarus/skills` and also picks up the existing
`~/.hermes/skills` tree, so that library is available without copying it.

## Configuration

`~/.icarus/config.yaml` — only what you set overrides the defaults. Print the
effective config with `icarus --config`.

```yaml
model:
  default: ''                        # blank = follow whatever llama-swap has resident
  base_url: http://127.0.0.1:9292/v1
  temperature: 0.2
agent:
  max_iterations: 40
  context_threshold: 0.75            # compact at 75% of usable context
llama_swap:
  unload_before_switch: true
```

Env overrides: `ICARUS_BASE_URL`, `ICARUS_MODEL`, `ICARUS_HOME`.

## Token accounting

Every turn is written to Mission Control's `usage_stats.json` tagged
`source: "icarus"`, updating all four shapes MC reads (`models`, `recent_calls`,
`daily`, `hourly`), so Icarus turns appear in the Token Usage tab beside every
other local caller. Writes are atomic and best-effort — a missing or locked
stats file never fails a turn. Disable with `usage.enabled: false`.

## Layout

```
icarus/
  cli.py        REPL, slash commands, one-shot and piped modes
  loop.py       the agent loop; model switching; compaction triggers
  swap.py       llama-swap client — models, /running, /unload, real ctx sizes
  llm.py        chat completions over stdlib urllib; SSE streaming
  caps.py       per-model capability probe, cached to disk
  protocol.py   text tool-call protocol + multi-family leak parser
  context.py    token budgeting, calibration, compaction
  session.py    persistence and resume
  interrupt.py  raw-mode type-ahead + Esc, pinned input line
  ctxplan.py    VRAM-aware context sizing; edits llama-swap config
  gguf.py       GGUF header reader — trained ctx, KV geometry, SWA
  skills.py     SKILL.md discovery and the skill tool
  skillhub.py   trusted-repo search/install, safety scan, lockfile
  detect.py     autodetect the local inference stack
  usage.py      optional token-usage stats writer
  render.py     ANSI output, spinner with elapsed clock
  tools/        registry + the ten built-ins
```

## Further reading

[**What breaks when you build an agent for local models**](docs/building-for-local-models.md)
— the measured engineering writeup behind this tool: the five tool-call syntaxes
models leak as plain text, why sliding-window attention changes KV math by 5.8x,
how a 3-line file made a model answer "at least 3 lines", and why loopback isn't
local.

## License

MIT — see [LICENSE](LICENSE).
