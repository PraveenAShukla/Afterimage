# Harness

How the code makes sure a bad run cannot quietly become a result, and how paid APIs are kept safe.

## Two layers
1. **Code harness:** `scripts/check.sh`, run by the git pre-commit and pre-push hooks. It covers:
   - ruff bug-finding lint;
   - the full test suite;
   - a secret scan for key formats and the literal values of secret env vars;
   - a private-name denylist scan (the list itself is never committed);
   - checks that `.env`, `serve/site.env`, `internal/` and `runs/` are ignored.

   Install the hooks after cloning:
   `printf '#!/usr/bin/env bash\nexec scripts/check.sh --staged\n' > .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit`
2. **Experiment harness:** the layers below. API keys come only from the environment or a git-ignored, chmod-600
   `.env` (preflight fails otherwise). Every byte written to a run log goes through a redactor first, so a key
   echoed in a model output or error body is replaced by `[REDACTED:<name>]` and logged as an event.

**Sandbox.** Agents only return text. Model output is never executed, evaluated or used as a path. The only thing
taken from it is one number, via a regex. Agents have no tools, network or filesystem access.

## Money guards (paid APIs)
Every paid call must pass all of these, checked atomically under a file lock before the call:
- **Explicit approval:** a config with any paid model refuses to run without `--approve-usd N`. The dry run prints
  `usd_expected` and the worst case; the team approves an amount; that amount is a hard cap for the invocation.
- **Provider caps:** `.env` sets `BUDGET_USD_*`, 10% below each account's spending limit.
- **Spend-rate tripwire:** at most `MAX_USD_PER_10MIN` in any rolling 10 minutes.
- **Rate limiter:** `RATE_RPM_*` and `RATE_OUTPUT_TPM_*`, a sliding window shared by all games in the process.
- **Kill switch:** `python -m collusion.monitor --stop` creates `runs/STOP`. Paid calls are refused and every game
  aborts at its next round. `--resume` removes it.
- **Monitor:** `python -m collusion.monitor --watch 60 --auto-stop` shows spend per provider, open reservations,
  the last 10 minutes, and live runs with heartbeat and ETA. It alerts on 80% of a cap, the tripwire, hung runs
  and high fallback rates, and pulls the kill switch at 95% of a cap.

## Safety nets
| Layer | What it catches |
|---|---|
| `config.py` | unknown or mistyped keys, missing sampling params, wrong arm/model combos, bad seeds |
| `preflight.py` | server down, wrong served model, no vLLM version, probe call unparsable or truncated, seed ignored, uncommitted code, missing API key/price/cap, missing source runs |
| `health.py` (live) | fallback rate > 10%, truncation > 25%, 5 consecutive transport errors → **abort**; frozen prices → warning |
| `validate.py` (post-run) | re-derives shares, profits, BRs, summary, played-vs-parsed prices, prompt history vs played history, replay/branch lineage |
| `analyze.py` | only uses runs with status=complete AND validation ok; lists every excluded run |
| `budget.py` | hard per-provider USD caps, checked before every paid call with the worst-case cost |

Every run directory holds:
- `manifest.json`: config, seed, git sha, dirty files, packages, GPU model, vLLM version, prompts
- `calls.jsonl`: every attempt, with full prompt, raw output, reasoning, tokens, latency, seed, cost
- `rounds.jsonl`: prices, shares, profits, BRs, sources, statuses
- `events.jsonl`
- `debug.log`
- `heartbeat.json`
- `summary.json`
- `validation.json`

## What every run records (`runs/<run_id>/`)
| file | contents |
|---|---|
| `manifest.json` | config, seed, git sha + dirty files, packages, GPU model, vLLM version and served model, system prompts |
| `calls.jsonl` | one line per agent call, every retry included: full prompt, raw output, reasoning, parsed price, parse status, tokens, latency, seed, parameters actually sent, cost |
| `rounds.jsonl` | one line per round: both prices, shares, profits, best responses, where each price came from, decision status |
| `events.jsonl` | parse failures, truncations, clipping, fallbacks, aborts, resumes |
| `debug.log`, `heartbeat.json` | human-readable trace; live progress |
| `summary.json`, `validation.json` | headline numbers; the validator's verdict (re-derived from the logs) |

## Later additions
| Piece | What it does |
|---|---|
| `backends/http.py` `IFMBackend` | IFM's hosted K2: the vLLM request body plus Bearer auth from `IFM_K2_API_KEY` (the name is logged, never the value) |
| `ratelimit.py` min interval | `RATE_MIN_INTERVAL_S_<PROVIDER>` spaces call starts (burst limits); regression test in tests/test_spend_guards.py |
| `prompts.SUPERVISION` / `prompts.PRESSURE` | Optional system-prompt lines. With both off, the prompt is byte-identical to the spec (tests pin this) |
| `supervisor.py` | The real supervisor agent. Before each live round it writes a note (≤ 40 words) the firms read. Every call is logged to `supervisor.jsonl` |
| `validate.py` additions | The note in each prompt equals the round's note and the last successful note in supervisor.jsonl; the supervision and pressure lines are present for exactly the configured firms |
| `analysis/treatments.py` | Pre-registered contrasts for supervision and pressure; uses only complete, validated runs |

Run directories in supervisor arms also hold `supervisor.jsonl` (one line per supervisor call: prompt, raw output,
parsed note, tokens), and each round record carries `note1` / `note2`.
