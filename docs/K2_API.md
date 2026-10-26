# IFM K2 API — usage notes (no key in this file)

Key: `IFM_K2_API_KEY` in `.env` (owner-only, git-ignored). Added.

## Endpoint
- `POST https://api.ifm.ai/v1/chat/completions`, OpenAI-compatible, `Authorization: Bearer $IFM_K2_API_KEY`.
- `GET https://api.ifm.ai/v1/models` lists: `IFM/K2-Horizon-375B-A23B`, `IFM/Jais-3-MoVA-36B-A4B`, `IFM/Jais-ASR`.
- In the harness: provider `ifm` in `src/lee/backend.py` (e.g. `{provider: ifm, model: IFM/K2-Horizon-375B-A23B, extra: {max_tokens: 4000}}`).

```bash
curl https://api.ifm.ai/v1/chat/completions -H "Authorization: Bearer $IFM_K2_API_KEY" -H 'Content-Type: application/json' \
  -d '{"model":"IFM/K2-Horizon-375B-A23B","messages":[{"role":"user","content":"Hello"}],"max_tokens":2000}'
```

## Observed behaviour
| Item | Observed |
|---|---|
| Model type | Reasoning model (375B total / 23B active MoE). Reasoning comes back in `message.reasoning` / `reasoning_content`, the answer in `content`. |
| Truncation trap | With `max_tokens: 50` the reasoning ran out the budget and **reasoning text appeared in `content`** (finish `length`). Give it ≥ 2000–4000. |
| `reasoning_effort` | Accepted (`low`). Output length varied a lot between calls (49 vs 1,453 tokens for `low`), so this was not tested systematically. |
| Latency | 7–20 s per call. A 200-round game is ~200 sequential rounds → roughly 40–60 min per game even with firms in parallel. |
| Lee round-1 prompt | Parsed OK both ways (price 2.50), 212 input tokens, 49–1,154 output. |
| Price | **Unknown.** No price is published or returned. `configs/prices.yaml` has 0.0 marked UNVERIFIED: that means "unknown", not "free". |

## Rate limits (from response headers, not from docs)
| Window | Tokens | Requests |
|---|---|---|
| per minute | 24,000,000 | 10,000 |
| per hour | 1,440,000,000 | 600,000 |
| per day | 34,560,000,000 | 14,400,000 |
| `x-ratelimit-limit-requests` | — | 3 (meaning unclear; possibly concurrent; check before high parallelism) |

The limits are far beyond anything the sprint needs. In practice latency, not quota, is what binds.

## Before a real run
- **The budget guard does not limit K2.** With price 0.0, every reservation costs $0, so it passes even with no `BUDGET_USD_IFM` cap (checked). `lee.run` still demands `--approve-usd`, but only as a formality. If IFM turns out to charge, put the real price in `configs/prices.yaml` before large runs.
- Treat K2 as one more open-weight model in the S1 design, not as a replacement: the headline results (Qwen3 inherited, etc.) are frozen.
