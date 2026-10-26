# Pre-registration: Price Afterimage, wave D1 (tag `prereg-afterimage-D1`)

Written before any D1 run.
- **Code:**
  - `analysis/afterimage.py`: the `rules()` function encodes every label below;
  - configs from `scripts/make_afterimage_configs.py`.
- **Freeze:** no rule, window, seed count, CI rule or exclusion may change after any D1 data.
- **Times:** last launch; hard stop. Runs that finish later are excluded from confirmatory claims.

## Common
- **A(W):** OLS slope, across the runs of one stage, of the window-mean price on the nominal opening x (BAND {1.55, 1.75, 1.95}).
  - Game mean (p1+p2)/2, unless stated firm-1 (A1).
  - Rule window 121–170; early window 41–60; 20-round windows are reported (EXPL).
- **CIs:**
  - single-stage bounds (LB90/UB90): the 90% within-level bootstrap (10,000 draws, seed 20261023). This is the CI X0 validated;
  - contrasts: the window-paired cluster bootstrap 90% CI, over the opening window_ids both stages ran.
- **Exclusions:**
  - runs not complete + valid (validator), and duplicates across runs dirs;
  - counted per stage in numbers.json;
  - a budget_stop run is quarantined to `_aborted/` and relaunched with `--tag`.
- **Seeds:**
  - new master 97000 (seeds 970000–979999), ≥ 1000 away from 91000, 93000 and every other master;
  - drawn per (stage, level) and disjoint from every seed in configs/.
- **Stages:** all `ait-` (never v2* or cls-), each in its own runs dir with the 18 source symlinks (`scripts/make_runs_dir.sh`).
- **Opening and prompt:**
  - opening windows as conf: perm_seed 4102, cross_model, job i uses order[i % 18];
  - prompt E1; 170 rounds; 150-row window; T 0.7 for Qwen; the API models at their configured T 1.0.
- **Gates:** run `analysis/verify.py` before any number is cited. The afterimage gate must still reproduce the frozen conf numbers.

## E5 [MUST]: fresh-seed confirmation, plus the price-only view
- **Design:** Qwen3-14B-AWQ, thinking off; x ∈ BAND × 16 seeds × {full view `ait-e5-full-qwen3off`, price-only view `ait-e5-po-qwen3off`} = 96 runs.
- **CONFIRMED** if, for the full view, all of:
  - LB90 A(121–170) > 0.5;
  - A(121–170) ∈ [0.75, 0.93];
  - A(41–60) ∈ [0.92, 1.00].
  - PIs from the committed values: 0.837 ± 1.645·√(0.034² + 0.042²), and 0.961 ± 0.04.
- **D_PO** = A_PO(121–170) − A_full(121–170), window-paired:
  - CI inside ±0.15: "price columns suffice";
  - UB90 < −0.15: "share and profit carry part";
  - LB90 > +0.15: "hiding profit lengthens it";
  - otherwise "unresolved".
- The per-window curve is EXPL.

## E1 [MUST]: relay cut
- **Design:**
  - firm 1 = Qwen3-14B off;
  - firm 2 = mock from round 21: `ctl-br` (best response; `ait-e1-br-qwen3off`) or `ctl-tft` (the rival's last price; `ait-e1-copy-qwen3off`);
  - x ∈ BAND × 12 × 2 = 72 runs.
- **Outcome:** A1(W), the slope of firm 1's price.
- **Comparator:** A1_self, from the conf band runs `v2O-qwen3off`, window-paired on the common window_ids (order[0..11]).
- **BR cell:**
  - "loop-carried" if UB90 A1_BR(121–170) < 0.2;
  - "single-agent carry" if LB90 > 0.5;
  - otherwise "partial", reported as the share A1_BR / A1_self.
- **COPY cell:** "preserved" if the 90% CI of D_COPY = A1_COPY − A1_self (121–170) lies inside ±0.15.
- **Holm** over the 2 contrasts D_BR and D_COPY at 121–170 (bootstrap two-sided p).

## E1b [MUST, on the same server]: fixed-Nash rival
- **Design:** firm 2 = `ctl-nash` (fixed 1.4729); x ∈ BAND × 12 = 36 runs (`ait-e1b-nash-qwen3off`).
- **Labels:**
  - "own-column carry" if LB90 A1(121–170) > 0.5;
  - "none" if UB90 < 0.2;
  - otherwise "partial".

## E4 [MUST]: API breadth (the team approves each cap; at most 2 paid configs at once; one runs dir per model)
- **Models, in order:**
  1. Claude Haiku 5.5 (≈ $8);
  2. Claude Sonnet 5.5 at 12 per level (≈ $93);
  3. GPT-6.1-sol (≈ $37).
  - Costs are dry-run expectations from the measured output tokens per call.
- **Design:**
  - each model: O band 3 × 12 (`ait-e4-<m>`), plus 12 cold starts (arm L, `ait-e4L-<m>`);
  - MAX_USD_PER_10MIN 15 per ledger.
- **Labels:**
  - "relays" if LB90 A(121–170) > 0.5;
  - else "lets go" if LB90 A(41–60) > 0.5 and UB90 A(121–170) < 0.5;
  - else "none" if UB90 A(41–60) < 0.2;
  - otherwise "unresolved".
- **Holm** within the API family, on the one-sided bootstrap p of A(121–170) ≤ 0.5.
- **Disclosed in every row:**
  - Anthropic: T 1.0, effort low, adaptive thinking;
  - sol: reasoning_effort low.

## E2b [SHOULD]: GPT-6-luna, reasoning_effort none vs high
- **Design:** x ∈ {1.55, 1.95} × 12 × 2 cells = 48 runs (`ait-e2b-luna-none`, `ait-e2b-luna-high`).
- **Rule:** D = A_high(121–170) − A_none(121–170), window-paired. "Reasoning shortens" if UB90 D < −0.15.
- With 2 levels, the within-level CI uses only those 2 levels.

## Not run in D1
- **E2a** (Qwen3 thinking on): dropped before any D1 run for lack of GPU capacity (1 GPU per user on the cluster). Stated as a limitation.
- **E5b, E3:** dropped. There is no second GPU.
- **Hint/sham (lever 1):** separate tag D2.

## Predictions
- The team chose not to sign predictions for this wave. The values below are Claude's, recorded before any data, for reference only.

| Quantity | Claude's prediction |
|---|---|
| E5 full-view A(121–170) | 0.84 [0.72, 0.94] |
| E5 CONFIRMED? | yes, P ≈ 0.8 |
| E5 D_PO | ≈ 0 [−0.10, +0.10] |
| E1 A1_BR(121–170), label | 0.10 [−0.05, 0.35]; loop-carried or partial |
| E1 A1_COPY(121–170), preserved? | 0.85 [0.70, 0.98]; preserved |
| E1b A1(121–170) | ≤ 0.2 |
| E4 labels Haiku / Sonnet / sol | lets go / lets go / lets go or none |
| E2b D(121–170) | −0.15 [−0.45, +0.10] |

# Wave D2: lever 1, best-response hint vs matched sham (tag `prereg-afterimage-D2`)
Written before any D2 run. D1 rules and D1 code paths are unchanged; D2 only adds lines.
- **Lines:** one line each round after the history rows (never at round 1), with b = Market.best_response(r):
  - hint: "Note: if the competitor repeats its last price (r), your profit this round is highest at a price of b.";
  - sham, no number: "Note: prices in the history are rounded to three decimals.";
  - the validator recomputes the hint line from the shown rival price (within 6e-4).
- **Design:**
  - Qwen3-14B off; x ∈ {1.55, 1.95} × 12 seeds × {hint `ait-d2-hint-qwen3off`, sham `ait-d2-sham-qwen3off`} = 48 games;
  - plus Q-hint at q ∈ {1.473, 1.925} × 3 (`ait-d2-qhint-qwen3off`);
  - hint and sham in ONE job (one server). The hint never runs without the sham.
- **Primary:**
  - D_hint = A_hint(121–170) − A_sham(121–170), window-paired cluster 90% CI.
  - **WORKS** iff all of: UB90 A_hint(121–170) < 0.2 (within-level); UB90 D_hint < −0.15; the run gates pass (complete + valid); and the FN gap ≤ 0.05.
  - FN gap = mean |p1 − BR(1.473)| over the last 50 rounds of the q = 1.473 Q-hint runs.
- **Secondary:** A_sham − A_base, with base = E5 full view on the common windows.
  - E5 ran in another job, so on another server process; any server difference is disclosed.
- **Always printed:**
  - the late level per plant;
  - hint uptake (share of decisions within ±0.01 of b, rounds > 20);
  - whether prices moved toward Nash (1.473) or toward the cold-start level (Qwen3 ≈ 1.80).
- **Reset fix:** analysis only, no runs. A reset at round 41 removes history sensitivity, not the price level: play returns to the cold-start level, which is above the 1.55 plant.
- **Claude's predictions** (reference only; the team chose not to sign):
  - A_hint(121–170) 0.30 [0.00, 0.75];
  - A_sham − A_base 0.00 [−0.12, +0.12];
  - uptake 0.35 [0.05, 0.80];
  - P(WORKS) ≈ 0.4.

# Amendment D3: E2b luna-high rerun at max_tokens 8192 (tag `prereg-afterimage-D3`). POST-DATA.
Written AFTER the D1 E2b data were read (team decision). Labelled post hoc wherever it is used.
- **Why:** at D1, max_tokens 2048 cut off 9.5% of luna's high-effort calls (median output 922 tokens, p99 at the cap). 10 of 24 high runs failed the truncation check, so the high cell failed G1 and E2b was not scored.
- **Change, and only this:**
  - stage `ait-e2b-luna-high8k` (new run ids);
  - sampling.max_tokens 8192;
  - est_output_tokens 1400.
  - Same seeds, opening windows, prompt, T and reasoning_effort as the D1 high cell.
  - Configs from `scripts/make_e2b_amend_configs.py`.
- **Comparison:** D = A_high8k(121–170) − A_none(121–170), the D1 none cell, window-paired cluster 90% CI.
- **Rule:**
  - "reasoning shortens" if UB90 D < −0.15, else "no shortening shown";
  - "not scored (G1)" if more than 10% of the high8k runs are excluded.
- **Status:** exploratory and post-data. It cannot be cited as confirmatory.
- **Claude's prediction** (made after seeing the D1 survivors, so weak): D ≈ −0.3 [−0.6, 0.0].
