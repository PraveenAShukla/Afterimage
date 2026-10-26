# How closest papers handle reproducibility (checked)
Method: arXiv HTML read through a summarising fetch tool, first ~100k characters per paper, so appendices may hold
details marked "not stated". Quotes are as returned by that tool; re-check before citing.

| Paper | Runs/condition | Model version + dates | Sampling | Seeds | Code/transcripts | Nondeterminism | Uncertainty |
|---|---|---|---|---|---|---|---|
| Fish et al. 2404.00806 | 21 × 300 rounds | "GPT-4 (0613 version)", Dec 2023-Jan 2024 | Appendix B (not read) | not stated | not stated | one line ("LLMs are stochastic") | Welch t, robust SE |
| Garra 2609.13037 | 21 repro / 7 treatments | DeepSeek-V3.1 via DeepInfra, no snapshot/dates | not stated | not stated | not stated | no; Fish reproduced "in direction, weaker" | Welch t, robust SE, "exploratory" |
| Peer-Preservation 2604.19784 | 100 samples/cell | names only | "temperature of 1.0 with default parameters for all models" | not stated | code + dataset on GitHub, example transcripts | no | bold if p<0.05 |
| Lee & Park 2609.18346 | 5 (GPT-5, Qwen 32B/14B) / **3 (Claude Sonnet & Haiku 4.5)** (App. L Table 6) | names only; Claude "accessed via the Anthropic API" (App. B) | T=0.7 + top_p=0.95 stated for Claude (App. B, F.1) - **not sendable, see below** | initial-price seed only | not stated | no | mean ± SD, Welch t, Bonferroni |

Takeaways for our report:
- The field standard is repeated runs + significance tests; snapshot pinning, seeds and nondeterminism are rarely
  reported.
- Peer-Preservation runs all models at T=1 default, which supports P8 (API arms at T=1).
- Gap we can fill: report per-model repeatability (identical prompt × N) and the noise floor, with full transcripts
  released.

## Check: Lee & Park's stated Claude settings cannot be sent
Their Appendix B says Claude Sonnet 4.5 / Haiku 4.5 were "accessed via the Anthropic API" with "temperature 0.7,
top-p 0.95". Anthropic's migration guide says passing both "will error on every Claude 4+ model". We sent exactly
that to both models (rejected requests are free): HTTP 400 "`temperature` and `top_p` cannot both be specified for
this model. Please use only one." (request ids in docs/api/claude45_temp_topp_check.txt).
=> The Claude sampling that actually ran in that paper is unknown: only one parameter, or a wrapper that drops one.
API behaviour at their run time cannot be checked by us; the guide says the rule covers all Claude 4+ models.
Also internally inconsistent: hardware is "four NVIDIA RTX 3090s" (App. F.1) but some models were "Served locally
via vLLM on V100 GPU Server" (App. B).
Lesson for our methods section: log the parameters actually SENT on every call (we do: sent_params), not just the
config.
