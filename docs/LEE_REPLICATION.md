# Lee 2610.11256 on its own terms: replication + randomised openings (design, bullets only)

Prepared. Code: `src/lee/` (a separate package; it reuses only collusion.budget, collusion.runlog and the
HTTP retry helper). Nothing paid runs without the team's approval. Intended to run IN the sprint window (Oct 23–25), on fresh
seeds, with this design pre-registered first.

## Lee's setup (from the paper itself, arXiv HTML; sections 3.1–3.4, Figure 1, Tables 1–2)
- 4-firm logit Bertrand: a = 2, a0 = 0, c = 1, μ = 0.25.
  - Nash 1.33 (π 0.0815); joint monopoly 2.05 (π 0.2011). Prices are allowed in [0, 5].
- Models (cheapest tier, lowest reasoning), all through the OpenAI-compatible chat interface. Temperature is not
  sent (provider default, stated to be 1.0):

  | Model | Reasoning setting | Price in/out $/M (Lee Table 2) |
  |---|---|---|
  | gemini-3.1-flash-lite | thinking "minimal" | 0.25 / 1.50 |
  | claude-haiku-4-5 | thinking off | 1.00 / 5.00 |
  | gpt-5-nano | reasoning "minimal" | 0.05 / 0.40 |
  | deepseek-flash | thinking disabled | 0.30 / 1.20 |

- Prompt: Figure 1, stated to be "verbatim as sent", with placeholders {me}, {n}, {cost}, {pmin}, {pmax}, {k} = 20,
  {w} = 40.
  - JSON reply {price, plan, insight}.
  - Memory = previous plan + insight + the last 20 rounds (all four prices, own quantity, own profit, 2 decimals).
  - Period 1: plan and insight "(none)", history "(no history yet)".
- 200 periods; 20 runs per cell, in batches of 10; 11 compositions. Positions randomised within a run.
- Δ = (π̄ − π^N)/(π^M − π^N) on the market-average per-firm profit over the last 50 periods.
- Lee's own evidence on openings (Table 5): Gemini always opened at 3.00, Claude always at 2.50; DeepSeek averaged
  2.74 and GPT 1.87. Lee attributes the level to the opening phase, but observationally (l.709–712, 805–807; line
  numbers still to be checked in main.tex).

## What we run
- **R0, replication (cold start, Lee's exact setup):** the 4 homogeneous cells (4:0:0:0), 10 runs each.
  - Target: Lee's homogeneous Δ (Claude 0.79, DeepSeek 0.24, GPT ≈ 0, Gemini about 0.72–0.79; exact per-cell
    values to be copied from Lee's tables before running).
- **R1, randomised opening (the new test):** the same 4 homogeneous cells.
  - The first 20 rounds are injected (labelled rounds 1–20): a REAL window taken from R0 runs of the SAME model
    (rounds 31–50), level-shifted so that its mean = x, with one delta for all 4 firms, clipped to [0, 5].
  - Plan and insight at round 21 read "(none)" (disclosed difference: the agent did not write the injected rounds'
    plans).
  - x ∈ {1.50, 1.80, 2.10} (between Nash 1.33 and just above monopoly 2.05), 10 seeds per level, 180 live rounds
    (21–200). Δ over 151–200, as in Lee.
- **Primary statistic (per model):** OLS slope β of the late market price on x, and the opening-induced spread
  Δ(2.10) − Δ(1.50).
  - The headline comparison: that spread vs Lee's between-model differences in homogeneous Δ.
  - Decision rule as in PREREG_S1: inherited if one-sided LB95(β) > 0.5; target-seeking if UB95 < 0.2; else
    intermediate. Holm over the 4 models.
- **Optional R2:** the 1:1:1:1 mixed cell with randomised openings (does the low-price model drag the market?),
  only if budget allows.

## Cost (Lee Table 2 prices; ~900 input + ~100 output tokens per call; 4 firms × 200 periods = 800 calls per run)
| Model | $/run (est.) | R0 (10 runs) | R1 (30 runs, 180 live) | Total |
|---|---|---|---|---|
| claude-haiku-4-5 | ≈ 1.12 | 11.2 | 30.2 | ≈ \$41 |
| gemini-3.1-flash-lite | ≈ 0.30 | 3.0 | 8.1 | ≈ \$11 |
| deepseek-flash | ≈ 0.31 | 3.1 | 8.4 | ≈ \$12 |
| gpt-5-nano | ≈ 0.07 | 0.7 | 1.9 | ≈ \$3 |
| **All** | | | | **≈ \$67** (+ 30% margin: cap \$90) |

- Prices are from Lee's Table 2 and must be re-checked on each provider page before any paid run.

## Keys needed (names end in _API_KEY so the log scrubber redacts them)
- ANTHROPIC_API_KEY (have).
- OPENAI_API_KEY (have; gpt-5-nano access to be checked with a free models.list call).
- **GEMINI_LEE_API_KEY (missing).**
- **DEEPSEEK_API_KEY.**
- Budget caps per provider via BUDGET_USD_<PROVIDER>. Unset = \$0 = refused.

## Known deviations from Lee (to state)
- Position assignment within homogeneous cells is irrelevant (same model everywhere).
- The injected openings' plan and insight are "(none)".
- Model versions may have moved since Lee's runs.
- Prices may have changed.

## Cost, VERIFIED (provider pricing pages; output tokens per call from Lee Table 2)
- **Inputs:**
  - Input per call ≈ 520 tokens at a full 20-round history (measured on our Figure-1 prompt, chars/3.7); 600 is
    budgeted.
  - Calls per model: R0 10 × 200 × 4 = 8,000, plus R1 30 × 180 × 4 = 21,600, so **29,600**.

| Model | $/M in / out (verified) | Out tokens/call (Lee) | R0 + R1 | Worst case (600 out) |
|---|---|---|---|---|
| claude-haiku-4-5 | 1.00 / 5.00 | 112 | $34.34 | $106.56 |
| gemini-3.1-flash-lite (paid tier) | 0.25 / 1.50 (thinking billed as output) | 85 | $8.21 | $31.08 |
| gpt-5-nano | 0.05 / 0.40 | 73 | $1.75 | $7.99 |
| deepseek-flash | 0.30 / 1.20 peak; half off-peak | 88 | $8.45 (≈ $4.2 off-peak) | $26.64 |
| qwen3.5-flash (Alibaba Model Studio, intl) — our addition, not in Lee | 0.10 / 0.40 | ~100 (assumed) | $2.96 | $8.88 |

- **Expected total, 5 models: ≈ $56**; +10% for retries ≈ **$61**.
- Optional R2 mixed 1:1:1:1 cell with randomised openings: ≈ $10.
- **Recommended hard cap: $100.**
- Fixed costs:
  - Qwen: 1M free tokens for 90 days on new Model Studio activation. Not enough for 29,600 calls (~21M tokens),
    but it covers the probe.
- **Do NOT use the Gemini free tier:** its data is used for product improvement (confidentiality). Use the paid
  tier.
- **Sprint keys:** none are provided. In help-desk, teams asked for the OpenRouter/Nebius keys from the
  starter kit, and the only answer (from another participant) was to make your own free account. OpenRouter free
  models are capped at 50 requests/day (1,000/day after a one-time $10), which is unusable at this scale.

## Pre-registration (written BEFORE any paid Lee call; analysis frozen in analysis/lee_analysis.py)
- **Models (5):** Lee's four with Lee's settings (Table 2) plus **qwen3.5-flash** (our addition, thinking off).
  - DeepSeek uses {"thinking": {"type": "disabled"}} (provider docs). Lee's "thinking disabled" did not give the
    exact parameter.
- **Arms:** R0 = Lee's cold start, 10 runs per model. R1 = randomised openings x ∈ {1.50, 1.80, 2.10}, 10 runs per
  level, with source windows cut from the same model's R0 runs (rounds 31–50). R1 cannot start until R0 is
  complete and validated (enforced in code).
- **Tests (analysis/lee_analysis.py):**
  1. **R0 replication:** our mean Δ inside Lee's 95% interval.
  2. **R1 slope β** of the late price (151–200) on x. Inherited if LB95 > 0.5; target-seeking if UB95 < 0.2;
     else intermediate.
  3. **Opening spread** Δ(2.10) − Δ(1.50), compared with Lee's between-model gaps: Claude − DeepSeek 0.55, and the
     largest, Claude − GPT 0.94.
  4. **Rank flips:** for each pair Lee ranks A > B, flip = Δ_B(2.10) − Δ_A(1.50). It is established if the
     Holm-adjusted one-sided p < 0.05.
  5. **Variance share of Δ in R1:** opening vs model (secondary).
- **How the claim is graded:**
  - "Collusion is a property of the model" is **contradicted** if at least one rank flip is established AND at
    least one model's opening spread exceeds 0.55 with its 95% interval above 0.
  - It is **supported** if no flip is established and every model's spread interval lies below 0.55.
  - Otherwise: **mixed**.
- **Caution:** x = 2.10 is above Lee's monopoly price, 2.05, so a higher price there can lower joint profit. Read Δ,
  not price.
- **Rehearsal (no paid calls):** scripts/lee_mock_dryrun.py ran all 20 configs at full scale with scripted agents.
  - 200 runs, 0 invalid, call counts exact.
  - The analysis labels the known agents correctly: hold → inherited, β 1.00; best response → target-seeking,
    β 0.00.

### Claude's predictions (AI-proposed; not the team's), recorded before any paid call
| Model | R0 Δ (replicates Lee?) | R1 β | Label |
|---|---|---|---|
| claude-haiku-4-5 | 0.75 (yes, P ≈ 0.7) | 0.50 [0.20, 0.80] | intermediate |
| gemini-3.1-flash-lite | 0.70 (yes, P ≈ 0.6) | 0.80 [0.50, 1.00] | inherited (Lee: freezes at its first price) |
| deepseek-flash | 0.30 (yes, P ≈ 0.6) | 0.40 [0.10, 0.70] | intermediate |
| gpt-5-nano | −0.10 (yes, P ≈ 0.7) | 0.10 [−0.10, 0.30] | target-seeking |
| qwen3.5-flash | 0.60 [0.20, 0.90] (no Lee value) | 0.70 [0.40, 0.95] | inherited or intermediate |
- At least one rank flip is established: P ≈ 0.6. Most likely gemini(1.50) vs deepseek(2.10).
- Variance share in R1: model 0.45 > opening 0.25.
- Overall grade: **mixed** (P ≈ 0.5), contradicted (P ≈ 0.35), supported (P ≈ 0.15).

### Team predictions
Team predictions were not recorded before the runs (disclosed deviation).

## Pre-R1 amendment (written BEFORE any R1 call and BEFORE reading any R0 Δ; tag prereg-lee)
The team's pre-R1 amendments plus E8. This section supersedes the levels, flip pairs,
replication rule and grade written above. R0 is unchanged and was already running (git diff of configs/lee/R0_* is
empty).
1. **Opening levels x ∈ {1.33, 1.69, 2.05}:** Lee's Nash price, the midpoint, and Lee's monopoly price.
   - Before: {1.50, 1.80, 2.10}. There, even a perfect inheritor's spread is capped at about 0.66 (Δ = 0.33 at 1.50,
     0.99 at 2.10), below Lee's largest gap of 0.94.
   - A design choice made before any R1 data; disclose it in the paper.
   - R1 configs were regenerated with fresh seeds.
2. **Graded rank-flip family:** only Lee's 4 Holm-significant pairs (Table 4): Gemini > DeepSeek, Gemini > GPT,
   Claude > DeepSeek, Claude > GPT.
   - flip(A, B) = Δ_B(2.05) − Δ_A(1.33). Holm over the 4.
   - Gemini–Claude and DeepSeek–GPT are descriptive only.
3. **Same-x comparisons (descriptive):** at each x, each model's Δ, every pairwise difference with a 95% bootstrap
   interval, and whether Lee's ranking order holds from a common opening (the model × opening interaction).
4. **Grade (symmetric, Holm-adjusted):**
   - **CONTRADICTED:** at least one graded flip has Holm p_flip < 0.05, AND at least one model's opening spread
     Δ(2.05) − Δ(1.33) exceeds 0.55 with Holm p < 0.05.
   - **SUPPORTED:** every graded flip is rejected (Holm p_noflip < 0.05), AND every model's spread is below 0.55
     (Holm p < 0.05).
   - Otherwise **MIXED**.
   - Reported against BOTH Lee gaps, 0.55 (Claude − DeepSeek) and 0.94 (Claude − GPT).
5. **R0 replication rule (replaces "inside Lee's CI"):** replicated if |our mean Δ − Lee's mean Δ| ≤ 0.15, OR the
   95% CI of the difference contains 0. Lee's SE is taken from his 95% interval as (hi − lo) / 3.92. Written before
   reading any R0 Δ.
6. **Failed runs:** a run with more than 10% fallbacks, or one stopped by budget_stop, is moved to
   runs_lee/_aborted/ and re-run with the same seed.
7. **Rehearsal:** scripts/lee_mock_dryrun.py at the new levels gave 200 runs, 0 invalid, 29,600 calls per model. The
   analysis labels the controls correctly.
8. **E8, descriptive, pre-declared.**
   - **Measures:**
     - A_Lee(W), the OLS slope of the market-mean price over W on x, for W ∈ {41–60, 81–100, 121–150, 151–200}, per
       model, with a 90% within-level bootstrap interval (seed 20261023).
     - The share of plan and insight texts written after round 40 that contain a number within ±0.02 of the run's x.
   - Lee's prompt shows the last 20 rows, so the planted rows leave every prompt after round 40.
   - **E8 is descriptive only. It does not change any Lee grading rule, pair, level or threshold.**
   - The prompt's one directional prediction (Qwen3-14B A_Lee(41–60) > 0.5) applies only if a Qwen3-14B arm runs in
     Lee's market. **No such arm runs here** (our Qwen arm is qwen3.5-flash via OpenRouter), so E8 is purely
     descriptive.
   - AI-proposed value recorded anyway (Claude): if a Qwen3-14B arm ran, A_Lee(41–60) = 0.8 [80% PI 0.5, 1.0].
   - Unit test tests/test_lee_e8.py: on mock runs a copier gives A ≈ 1 and a best responder gives A ≈ 0.
9. **Operations:**
   - Each R1 provider process uses its own budget ledger (--ledger-dir runs_lee/_ledger_<model>), so the per-call
     ledger re-parse stays fast. The spend-rate tripwire is then per ledger.
   - Per-provider caps: Google $12, DeepSeek $12, OpenRouter $8.
   - Session caps keep the Lee total under the $100 cap.

## Results (read with the frozen analysis at tag prereg-lee → results/lee/numbers.json)
Run in the sprint window, R0 then R1. 50 R0 + 150 R1 runs, all complete and
validated. Lee spend $75.56 (cap $100). The team's signed predictions were not written before R1 (logged deviation).

### R0 replication (difference rule)
All four of Lee's models **replicated**:
| Model | Our Δ | Lee's Δ | Difference [95% CI] | Note |
|---|---|---|---|---|
| Claude | 0.835 | 0.79 | +0.045 | |
| Gemini | 0.581 | 0.72 | −0.139 | |
| GPT | −0.236 | −0.15 | −0.086 | |
| DeepSeek | −0.005 | 0.24 | −0.245 [−0.569, +0.079] | Replicated only through the CI clause (\|diff\| > 0.15). |
Qwen3.5-Flash (our addition) has Δ 0.667.

### R1: the opening
| Model | β (late price on x) | Label | Opening spread Δ(2.05) − Δ(1.33) [95% CI] |
|---|---|---|---|
| Claude Haiku 4.5 | **0.858** [one-sided LB 0.692] | **inherited** | **0.993** [0.779, 1.172] |
| Qwen3.5-Flash | **0.940** [LB 0.847] | **inherited** | **0.945** [0.856, 1.011] |
| GPT-5-nano | 0.980 [LB 0.278, UB 1.693] | intermediate (very noisy) | 0.651 [−0.036, 1.378] |
| DeepSeek Flash | 0.241 [LB −0.114] | intermediate | 0.288 [−0.135, 0.688] |
| Gemini 3.1 Flash-Lite | 0.032 [UB 0.096] | **target-seeking** | −0.034 [−0.111, 0.044] |
- **Graded rank flips** (Lee's 4 Holm-significant pairs):
  - **Claude(1.33) vs DeepSeek(2.05): +0.663, FLIP** (Holm p < 0.0001).
  - **Claude(1.33) vs GPT(2.05): +0.637, FLIP** (Holm p < 0.0001).
  - Gemini(1.33) vs DeepSeek(2.05): −0.392, and Gemini(1.33) vs GPT(2.05): −0.417. **No flip, the ranking holds**
    (Holm p_noflip 0.012 and 0.0004). Gemini stays high from any opening.
- **Same-x comparisons (descriptive):** Lee's ranking order holds at **none** of the three openings.
  - x = 1.33: Gemini +0.88, DeepSeek +0.20, Claude −0.18, GPT −0.19.
  - x = 1.69: Gemini +0.87, Claude +0.43, DeepSeek +0.32, GPT +0.10.
  - x = 2.05: Gemini +0.85, Claude +0.82, DeepSeek +0.49, GPT +0.46.
- **Variance of Δ in R1 (secondary):** opening 0.152, model 0.172. Comparable shares.
- **Grade (pre-registered, symmetric, Holm-adjusted):**
  - **vs Lee's 0.55 gap: CONTRADICTED.** Two graded flips are established, and Claude's and Qwen's spreads exceed
    0.55 (Holm p 0.0004 and < 0.0001).
  - **vs the 0.94 gap: MIXED.** No spread is significantly above 0.94; Claude's point estimate is 0.993 with lower
    bound 0.779.
- **Reading:** "collusion is a property of the model" fails in its strong form, but it is model-dependent.
  - Claude's and Qwen's collusion is set by the opening: Claude started at Nash prices competitively (Δ −0.18);
    started at monopoly it colludes (0.82).
  - Gemini is a genuine model property: Δ ≈ 0.85–0.88 from every opening.
  - DeepSeek is in between; GPT is too noisy to call.
- **Above-monopoly caution:** x = 2.05 is Lee's monopoly price; Δ near 1 there is a ceiling. Read Δ, not price.

### E8 (descriptive): afterimage in Lee's market, after the planted rows leave every prompt (round 41+)
| Model | A(41–60) | A(81–100) | A(121–150) | A(151–200) | Plans/insights restating x after round 40 |
|---|---|---|---|---|---|
| Claude | 0.90 | 0.86 | 0.86 | 0.86 | 0.318 |
| Qwen | 0.94 | 0.94 | 0.94 | 0.94 | **0.737** |
| DeepSeek | 0.71 | 0.39 | 0.24 | 0.24 | 0.189 |
| GPT | 0.63 | 0.79 | 0.88 | 0.98 | 0.066 (very wide intervals) |
| Gemini | 0.02 | 0.02 | 0.02 | 0.03 | 0.002 |
- **Important caveat for the "afterimage" framing.** In Lee's market the agents keep plan and insight memory. Qwen
  restates the planted price in 74% of its texts, and Claude in 32%. So here the plant is also carried by **written
  memory**, not only by re-emitted prices.
- No Qwen3-14B arm ran in Lee's market, so the prompt's directional prediction does not apply.

### Claude's predictions vs outcomes (AI-proposed, written before the runs)
- **R1 β:**
  - Claude: predicted 0.50 [0.20, 0.80]; observed 0.86, **above the interval**.
  - Gemini: predicted 0.80 (inherited); observed 0.03, target-seeking. **Wrong.**
  - DeepSeek: predicted 0.40 [0.10, 0.70]; observed 0.24, inside.
  - GPT: predicted 0.10 (target-seeking); observed 0.98, intermediate. **Wrong.**
  - Qwen: predicted 0.70 [0.40, 0.95]; observed 0.94, inside.
- **Flips:** Claude expected gemini(low) vs deepseek(high). Wrong: the flips are Claude's.
- **Grade:** Claude predicted mixed (P ≈ 0.5); the outcome is contradicted vs 0.55 and mixed vs 0.94.
