# Price Afterimage A(W): definitions and the X0 rule

Track A. Status: **EXPLORATORY** unless a later tag pre-registers a test.
Code: `analysis/afterimage.py` (A(W), CIs, contrasts, gate), `analysis/x0_mc.py` (X0 data path + coverage).
Configs: `scripts/make_afterimage_configs.py` → `configs/afterimage/<exp>/`.

## Definition
- **A(W):** OLS slope, across the runs of one stage, of the window game-mean price on the nominal opening x.
  - x ∈ BAND {1.55, 1.75, 1.95}.
  - A copier keeps the whole plant (A = 1); a best responder keeps none of it (A = 0).
- **Windows:**
  - 10-round bins 21–30 … 161–170;
  - 20-round windows 41–60, 61–80, 81–100, 101–120, 121–140, 141–170;
  - only for runs that reach them: 171–220, 221–320, 321–470, 471–620.
- **Per firm too:** A1(W) and A2(W). In the relay arms, firm 1 is the model and firm 2 the mock.
- **CIs:**
  - within-level bootstrap: resample runs within each x level, the same draws as `v2.boot_slope_within`, seed 20261023, 10,000 draws;
  - window-cluster bootstrap: resample the opening window_ids.
- **Contrasts:**
  - window-paired: only the window_ids both stages ran, then a paired window-cluster bootstrap;
  - p = 2·min tail share;
  - each contrast is one Holm family over its 20-round windows;
  - the list is `CONTRASTS` in the code.
- **Exclusions:** runs not complete + valid, and duplicates across runs dirs. Counts go in each numbers.json.

## Gate (passed)
- `afterimage.py` recomputes, from its own data path:
  - `beta_band` (rounds 121–170);
  - `carry_slope_41_60`.
- It does this for all 7 arm-O stages in `results/v2/conf/numbers.json`, and matches to 9 significant digits.
- On a miss it exits 2 and writes nothing.
- It re-runs inside `analysis/verify.py`.

## K2 split (C5)
- **Why:** the frozen conf numbers predate the K2 runs. K2 in `runs_cluster` reshuffles the shared bootstrap draws, so verify had been failing on CIs only.
- **Fix:**
  - verify now runs the frozen conf set without the `v2O-k2` runs; it reproduces exactly;
  - K2 has its own file, `results/v2/k2/numbers.json` (`analysis/k2.py`): β_band 0.0149, 90% CI [−0.0056, +0.0363], reproduces the documented 0.015 [−0.006, +0.036];
  - VERIFY OK.

## X0: estimator positive control (rule fixed before any X0 run)
**What the mock does:**
- The mock partial matcher: p_i,t = fmt4(w·fmt3(p_−i,t−1) + (1−w)·Nash + ε), with ε ~ N(0, σ) and w = 0.5^(1/h).
- fmt3 is the 3-decimal history line the mock reads from its prompt.

**Finding, before the X0 runs (mechanical, not a model result):**
- A step (1−w)(p − Nash) below 0.0005 never registers through fmt3.
- So slow matchers freeze. The stall threshold is |p − Nash| = 0.0005/(1−w): ≈ 0.0075 at h = 10, ≈ 0.036 at h = 50, ≈ 0.14 at h = 200, ≈ 0.58 at h = 800.
- At h = 200 and h = 800 the mock barely moves from its opening. Checked on one h = 200, x = 1.55 run: flat at 1.5457/1.5467 from round 22 to round 170.
- So the analytic A = mean w^(t−20) is NOT what the mock runs. It is reported as `A_analytic`; it is not the target.

**Target:**
- A_q(W) = the slope of the process the mock actually runs, rebuilt independently in numpy from each run's logged rounds 1–20.
- At σ = 0: exact, with the mock's own string rounding.
- At σ > 0: the mean over R = 200 synthetic replicates with fresh noise and the same plants.

**X0 PASS requires all of:**
1. Data path: at σ = 0, the rebuild reproduces every logged price of every partial-matcher run (max |diff| ≤ 1e-9).
2. At σ = 0: |Â(W) − A_q(W)| ≤ 0.01 in every window, for every h, at 170 and 620 rounds.
3. At σ > 0: |Â(W) − A_q,σ(W)| ≤ 0.05 in every window, for every (h, σ).
4. Coverage: in R = 200 replicates per (h, σ), the within-level bootstrap 90% CI (1,000 draws) covers A_q,σ(W) in ≥ 80% of replicates, every window.
5. Averager Â within 0.05 of 1, and br Â within 0.05 of 0, in every 20-round window.

**Other X0 details:**
- The tolerances (0.01 / 0.05 / 80%) are the plan's. The only change is the target (A_q instead of the analytic curve), and the reason is the stall shown above.
- Half-life is added to the model stages only if X0 passes: λ̂ from the slope of log Â on bin centre (bins with Â > 0.05), h = ln 0.5 / ln λ̂, D_0.5 = the first bin with Â < 0.5.
- **Design:**
  - 24 seeds × 3 levels;
  - h ∈ {10, 50, 200, 800};
  - σ ∈ {0, 0.002, 0.004, 0.008};
  - plus 620-round runs (window 600) for h ∈ {200, 800} at σ 0;
  - plus averager and br;
  - all mock, run on a cluster CPU node.
- **Smoke test** (local, 9 runs, h = 10; numbers not used): data path exact. Coverage was ~0.75 at 3 runs per level, a known small-n percentile-bootstrap under-coverage, and 0.81–0.92 at 24 per level.

## Implication for the model runs (to check, not a claim)
- The models also read 3-decimal history. A model making sub-0.0005 adjustments would look like a copier.
- The frozen-mock pattern is: identical prices from round 22 on. Check the share of runs with zero round-to-round change before reading Â ≈ 1 as memory.

## Results
**X0: PASS** (`results/afterimage/x0/numbers.json`, `results/afterimage/x0/mc.json`):
- 1,440 mock runs on a cluster CPU node.
- Data path: exact for every partial-matcher run.
- σ = 0: max |Â − A_q| = 0.0 for every h, at 170 and 620 rounds.
- σ > 0: worst max |Â − A_q,σ| = 0.0375 (h 200, σ 0.008), within the 0.05 tolerance.
- Coverage ≥ 80% in every cell and window.
- Averager and br: exact (0.0).
- Noise un-freezes slow matchers. At h = 800, A_q = 0.999 flat at σ 0, but decays 0.975 → 0.89 by 141–170 at any σ > 0.
- mc.json is the cluster output. verify.py recomputes the empirical half (numbers.json), not the Monte Carlo.

**C1, per-window A(W) for the conf stages** (`results/afterimage/conf/numbers.json`, both firms' mean):

| Stage | 41–60 | 61–80 | 81–100 | 101–120 | 121–140 | 141–170 | half-life h (rounds) |
|---|---|---|---|---|---|---|---|
| Qwen3-14B (off) | 0.961 | 0.944 | 0.931 | 0.918 | 0.888 | 0.803 | ≈ 442 |
| Qwen2.5-14B | 0.949 | 0.819 | 0.707 | 0.610 | 0.570 | 0.548 | ≈ 134 |
| Sonnet 5.5 (n 12) | 0.639 | 0.567 | 0.507 | 0.385 | 0.378 | 0.357 | ≈ 114 |
| gpt-oss-20b | 0.782 | 0.561 | 0.461 | 0.429 | 0.412 | 0.297 | ≈ 79 |
| K2 (extra model) | 0.270 | 0.182 | 0.093 | 0.052 | 0.029 | 0.005 | ≈ 24 |
| OLMo-2-13B | −0.128 | −0.237 | −0.207 | −0.167 | −0.141 | −0.103 | — |

- Qwen3 95% CIs:
  - within-level: 0.927–0.987 at 41–60, 0.698–0.883 at 141–170;
  - window-cluster: 0.932–0.986 and 0.720–0.878.
- The half-life is an exponential fit to the 10-round bins, so it is descriptive. For Qwen3 it extrapolates far past round 170.
- **Quantization check:**
  - no model run is flat over rounds 41–170 (0 of 96 for Qwen3);
  - Qwen3 holds its price exactly in 32% of rounds, K2 (at Nash) in 50%;
  - so Qwen3's A ≈ 0.96 is not the mock-style rounding freeze.

## CPU pack (EXPLORATORY; `results/afterimage/cpu_pack.json`, `analysis/afterimage_cpu.py`)
- **C10, validation audit: PASS.** 110,129 prompts at t ≥ 41 (400 conf arm-O runs); 0 show a row numbered ≤ 20.
  So the planted rows really have left the prompt by round 41.
- **C4, sub-monopoly slope** (levels 1.55 and 1.75 only), A(41–60) → A(121–170):

| Model | A(41–60) | A(121–170) | 1.95 games ending above monopoly 1.925 |
|---|---|---|---|
| Qwen3 | 0.927 | 0.807 | 12 of 24 |
| Qwen2.5 | 0.936 | 0.435 | 19 of 24 |
| gpt-oss | 0.835 | 0.374 | 0 of 12 |

  - Qwen3's late afterimage is not a monopoly-ceiling artefact.
- **C6, within-cold-start persistence** (v2L, 12 runs per model): the run-level slope of the 121–170 level on the 41–60 level.

| Model | Slope | 90% CI |
|---|---|---|
| Qwen3 | 0.95 | [0.71, 1.15] |
| Qwen2.5 | 0.93 | [0.82, 1.02] |
| gpt-oss | 0.56 | [0.30, 0.88] |
| OLMo | 0.24 | [0.04, 0.43] |

  - A Qwen run's own early level carries into its late level even with no plant.
- **C7, natural afterimage** (v2T: 20 real openings from the model's own cold starts × 3 seeds; opening-cluster bootstrap, 90%).
  - The slope of the 121–170 level on the transplanted round-20 level:

| Model | Natural slope | 90% CI | Planted A(121–170) |
|---|---|---|---|
| Qwen3 | 0.45 | [0.29, 0.56] | 0.84 |
| Qwen2.5 | 0.48 | [0.38, 0.60] | 0.56 |

  - On the rounds 1–20 mean the slopes are 0.44 and 0.41, so measurement noise in the round-20 level does not explain the gap.
  - **Honest reading:** real openings carry over too, but for Qwen3 at about half the planted strength.
  - The ICC CIs in results/v2/conf/numbers.json H2 are 95% (bootstrap over openings).

## X0 under the amended test specification (POST HOC: run after the original X0 output was read; `analysis/x0_mc_spec4.py`, `results/afterimage/x0/mc_spec4.json`)
- **Grid:**
  - h ∈ {10, 50, 125, 200, 480, 800};
  - per-call noise tuned per h so the between-game SD of the 121–170 level is σ_eff ∈ {0.05, 0.11};
  - truth = continuous mean w^(t−20);
  - real logged plants; clip [1, 2.5];
  - R = 1000 per cell; designs quick 2×12, standard 3×12, full 3×18, conf 3×24;
  - both 90% CIs (within-level, window-cluster), 1000 draws.
- **Result:** in all 48 cells, every window (41–60, 81–100, 121–170) and both CIs, coverage is ≥ 0.80. Minimum 0.83; at 121–170 the minimum is 0.84 within and 0.83 cluster. Max |bias| 0.015.
- **Not done:** the per-h harness panels through the real runner for h = 125 and 480.

## Reset fix (analysis only, no runs; PREREG D2)
- A reset at round 41 (clearing the history) makes play independent of x by construction. Play then follows the cold start.
- **Cost:** Qwen3's cold-start late level is 1.802 (v2L, 12 runs; card qwen3-14b-off). That is above the 1.55 plant and well above Nash (1.473).
- **So:**
  - a reset raises the price that follows a low plant, and lowers the one after a 1.95 plant;
  - a reset removes history sensitivity, not the price level.

## D1 results as they land (frozen rules; `results/afterimage/<exp>/numbers.json`)
- **E4 Haiku 5.5** (T 1.0, effort low, adaptive thinking; 3 × 12 + 12 cold starts):
  - A(41–60) 0.193 [0.066, 0.330]; A(121–170) −0.006 [−0.042, 0.035] (within-level 90%);
  - label **unresolved**: not "none", only because UB90 A(41–60) = 0.33 ≥ 0.2;
  - cold start at Nash: late level 1.482, Δ 0.03;
  - all gates G0–G6 pass; $9.03;
  - Claude's prediction ("lets go") was wrong.
- **E5 (Qwen3-14B off, fresh seeds, Modal H100, same pinned model and vLLM 0.11.2; 96/96 runs complete, 0 excluded):**
  - **CONFIRMED** (pre-registered rule):
    - full view A(121–170) = 0.786, 90% CI [0.631, 0.906] within-level ([0.632, 0.892] cluster);
    - LB90 > 0.5;
    - the point is inside the frozen PI [0.75, 0.93];
    - A(41–60) = 0.968 [0.938, 0.995], inside [0.92, 1.00].
  - **Price-only view:**
    - A(121–170) 0.728 [0.345, 1.010];
    - D_PO = −0.058, 90% CI [−0.453, +0.254] → **unresolved** (too wide to say whether the share and profit columns carry part of it).
  - Claude's predictions held: A(121–170) 0.84 [0.72, 0.94] predicted; D_PO ≈ 0, but the CI is far wider than the ±0.10 predicted.
  - Server: a different process from the conf runs (Modal vs cluster), which is disclosed. The rule compares to frozen PIs, and D_PO is within E5.
  - Modal cost ≈ $3.2.
- **E1 relay cut + E1b** (cluster, 108/108 runs complete, 0 excluded; firm 1 = Qwen3-14B off; A1 = slope of firm 1's price):
  - **BR rival:** A1(41–60) 0.001 [−0.009, 0.011]; A1(121–170) 0.001 [−0.002, 0.003] → **loop-carried**.
    - D_BR vs self-play (window-paired, 12 windows) = −0.807 [−0.870, −0.743], Holm p < 0.001.
  - **COPY (tft) rival:** A1(121–170) 0.839 [0.765, 0.910]; D_COPY = +0.032 [−0.036, +0.091] → **preserved** (Holm p 0.44).
  - **E1b, Nash-fixed rival:** A1(121–170) 0.002 [−0.006, 0.010] → **none**.
  - **Reading:** Qwen3 tracks the rival's price.
    - The afterimage survives only when the rival also re-emits it.
    - It is carried by the two-agent loop, not by either agent alone.
    - A rival that best-responds, or sits at Nash, removes it by round 41.
  - All three of Claude's E1/E1b predictions held.
- **E4 Sonnet 5.5** (T 1.0, effort low, adaptive thinking; 3 × 12 + 12 cold starts; all complete, gates G0–G6 pass; $84.53):
  - A(41–60) 0.623 [0.440, 0.792]; A(121–170) 0.414 [0.218, 0.596];
  - label **unresolved** (LB90 A_early 0.44 < 0.5, so not "lets go"; LB90 A_late < 0.5, so not "relays");
  - cold start late level 1.712, Δ 0.67;
  - partial carry, between Haiku (≈ 0) and Qwen3. Claude's prediction ("lets go") was not met.
- **D2 lever 1, hint vs sham** (cluster, tag prereg-afterimage-D2 the tagged code; 48 games + 6 Q-hint, all complete, 0 excluded):
  - **WORKS** (pre-registered rule):
    - A_hint(121–170) = 0.000, UB90 0.000 < 0.2;
    - D_hint = A_hint − A_sham = −0.861 [−0.912, −0.810], UB90 < −0.15;
    - gates pass;
    - FN gap 0.00004 ≤ 0.05.
  - **Sham:** A(121–170) 0.861 [0.808, 0.915] (no effect of the sham line). Sham − base (E5 full, other server) +0.111 [−0.007, +0.275].
  - **Where prices moved:** under the hint, both plants end at exactly Nash (1.473), not the cold-start level (Qwen3 ≈ 1.80). Hint uptake is 99.9%: Qwen3 plays the stated best response every round.
  - **Reading:** the "fix" works by Qwen3 following an instruction-like number, not by any change in what it learns. It removes the afterimage AND the supra-competitive price, so it is an auditor's or regulator's lever, not one a profit-seeking deployer would adopt.
  - Claude's predictions: A_hint 0.30 [0.00, 0.75] (held; 0.00 is at the lower edge); P(WORKS) ≈ 0.4 (it worked).
- **E4 GPT-6.1-sol** (reasoning_effort low, T 1.0; 3 × 12 + 12 cold starts; 48/48 complete after the quota relaunch; gates pass; $33.20, of which ≈ $14.65 billed to the shell's key before the fix):
  - A(41–60) 0.076 [−0.045, 0.210]; A(121–170) 0.006 [−0.197, 0.204];
  - label **unresolved** (misses "none" only because UB90 A_early = 0.21 > 0.2);
  - cold start Δ 0.19 (late level 1.534).
  - **E4 family:** no API model relays (Holm). Haiku and sol hold ≈ nothing; Sonnet holds about half.
- **E2b GPT-6-luna, none vs high (final):**
  - **none:** 24/24 complete; A(121–170) 0.760 [0.624, 0.887].
  - **high:** 14/24 complete. 10 aborted by the truncation health check: reasoning at effort high overruns max_tokens 2048, a setting chosen in the D1 config.
  - The frozen rule text gives "reasoning shortens": D = −0.463 [−0.703, −0.264].
  - **But the high cell fails G1** (42% model-caused exclusions > 10%). The survivors are the runs that reasoned less, so the contrast is selection-biased.
  - **Reported status: NOT SCORED (G1).** The D value is shown as exploratory only.
  - A valid test needs max_tokens ≥ 8192 under a new tag (not run).
- **E2b amendment D3** (POST-DATA, exploratory; luna-high at max_tokens 8192; same seeds and windows as the D1 high cell; 24/24 complete, max truncation 0.3%; $4.36):
  - A_high8k(121–170) = 0.200 [0.006, 0.392] vs A_none = 0.760 [0.624, 0.887];
  - D = −0.560 [−0.776, −0.351] → "reasoning shortens" under the amendment rule (G1 passes);
  - A_high8k(41–60) = 0.848, so high reasoning takes the plant in early and lets it go faster.
  - It agrees in direction with the biased D1 survivors (−0.46).
  - It is not confirmatory: the rule was written after the D1 E2b data were read.
