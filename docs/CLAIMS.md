# Claims register

Every claim we might make, with its evidence level, the number and the file it comes from. If a claim is not on
this page, it does not go in the paper. Evidence levels:

| Level | Meaning |
|---|---|
| **PRE-REG** | Hypothesis, statistic and test written down before the runs (docs/PREDICTIONS.md), then confirmed on fresh seeds |
| **DESIGN** | Pre-specified by the design (docs/DESIGN.md) and adequately powered (n ≥ 10 per arm) |
| **EXPL** | Exploratory: metric, test or label chosen after seeing data. It needs a fresh-seed confirmation before it can be a headline |

## Headline
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| C1 | Sonnet 5.5 prices higher against a live copy than against a non-reacting replay | **PRE-REG** | Confirmation: price I +0.099 [+0.035, +0.165], pooled permutation p = 0.014 (pre-registered test); paired sign-flip p = 0.008. Exploratory batch: +0.106 | docs/PREDICTIONS.md; results/story.json; analysis/story.py |

## S1: opening inheritance (plan v2 confirmation, run during the sprint)
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| S1 | Qwen3-14B (thinking off) inherits the opening: its late price follows an injected 20-round opening | **PRE-REG** | β_band 0.837 [0.778, 0.890], Holm p(β > 0.5) < 0.0001 | docs/RESULTS_S1.md (final confirmation); analysis/v2.py (frozen at prereg-s1) |
| S2 | Inheritance is not universal | **PRE-REG** | Held-out OLMo-2 −0.118 [−0.215, −0.032], not inherited; Qwen2.5 0.557 and gpt-oss 0.343, inconclusive | same |
| S3 | Real (transplanted) openings also persist | **PRE-REG** | ICC 0.641 (Qwen3), 0.798 (Qwen2.5) | RESULTS_S1, H2 |
| S4 | When two openings meet, Qwen3 settles at the lower one | **PRE-REG** | d +0.083, one-sided p 0.0033; not supported for Qwen2.5 | RESULTS_S1, H3 |
| S5 | Qwen3 is at the matcher corner of the fixed-rival audit; gpt-oss at the best-responder point | **PRE-REG** (audit) | ρ_r 0.884 vs 0.417; own-cost pass-through ≈ 0 vs ≈ BR | RESULTS_S1, Q ladder |
| S6 | WITHDRAWN: "the opening moves Qwen3's Δ more than Lee's model gap" | — | it compared a spread in our duopoly with a gap in Lee's 4-firm market; replaced by the within-market Lee challenge (L1–L5) | — |
| S7 | No single mechanism rule | **PRE-REG** (H4/H5) | STEP: Qwen2.5 ratchet (A 0.63); Qwen3 symmetric (A 0.175); Sonnet symmetric, BR-like. IMP inconclusive | RESULTS_S1, mechanism |
| S8 | The opening effect decays | descriptive | Qwen3 gap 0.415 → 0.254, half-life ≈ 200 rounds | RESULTS_S1, long horizon |
| S9 | K2 (375B reasoning, extra model) does not inherit | frozen rule, extra model (outside the H1 family) | β 0.015 [−0.006, +0.036]; Nash at every opening | RESULTS_S1, K2 |

## Supervision v1 (the team's 4 arms; pre-registered in docs/SUPERVISION.md before any run)
The v1 "agent" arms are **supervisor ADVICE notes (free text)**, not real monitoring. Real monitoring is supervision v2 (docs/SUPERVISION.md, supervision v2). Window-matched recompute and per-x Δ: results/v1_window_matched/numbers.json (labels unchanged).
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| V1 | Telling agents they are monitored, with no supervisor behind it, does not change prices | **PRE-REG** | Qwen3 +0.002 [−0.011, +0.014]; Qwen2.5 −0.003 [−0.024, +0.018] | results/supervision/numbers.json; analysis/treatments.py |
| V2 | Supervisor ADVICE notes (free text; the v1 "agent" arms, not real monitoring) change prices, in opposite directions by model. At x = 1.95 (around p^M) the higher Qwen3 price is NOT more collusion: Δ 0.946 vs 0.980 in the claim-only arm | **PRE-REG** | Qwen3 **+0.050** [+0.036, +0.062], Holm p 0.0003; Qwen2.5 **−0.070** [−0.117, −0.041], Holm p 0.0007 | same |
| V3 | Shared vs individual monitoring makes no detectable difference | **PRE-REG** | Qwen3 +0.012 (no effect); Qwen2.5 −0.024 (inconclusive) | same |
| V4 | (Advice notes) The Qwen3 rise comes from misread advice ("avoid prices too close" becomes "price above the rival") | **EXPL** (one game read; needs systematic coding) | 251/300 firm outputs cite the supervisor | RESULTS_S1, supervision |

## Design-level results
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| C2 | Below-BR reactions at r202 are rare (≤ 10%), not absent | DESIGN (r202 measure P1 + placebo P2; M8 definition) | Corrected (plan v2 M8/M13): below BR at r202 in Sonnet 0/10, Qwen2.5 0/20, Qwen3 1/20, sol 1/10. 'Returns' counted only for material cuts where firm 2 moved: Sonnet 8/8, Qwen2.5 14/18, Qwen3 7/17, sol 2/4 (only 4 of sol's 10 forced 'cuts' were material, per M7). Match ratios 0.24 / 0.67 / 0.69 / 1.28. | results/<stage>/punishment_paired.json |
| C3 | Stage 1 collusion index by model | DESIGN | Sonnet 0.63, Qwen3-off 0.74, Qwen2.5 0.63, R1 0.51, sol 0.31, Haiku 0.09, luna −0.27, gpt-oss −1.28, OLMo-2 −1.87 (n = 10 each) | runs*/_pipeline/*/pipeline.jsonl |
| C4 | Models that fail to collude are competent (best-respond to a fixed rival) | DESIGN (FN) + EXPL (FM) | FN gaps 0.0015 (Haiku), 0.0032 (luna), 0.0086 (gpt-oss); FM prices 1.662 / 1.676 / 1.657 vs BR 1.662 | results/story.json (fm) |
| C5 | Mixed-lab pair Haiku × luna lands at Nash | DESIGN (X arm) | Δ −0.013 ± 0.021, n = 20 | results/s3x-haiku55-gpt6luna |

## Exploratory (not headline material without confirmation)
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| E1 | Sonnet is a strategist: it best-responds to a replay but coordinates with a live copy | EXPL | FM price 1.663 vs BR 1.662 (5 × 100 rounds); A1 follower share 0.31 (d_BR 0.034 < d_match 0.082); A4 D − D-frozen +0.067 [+0.028, +0.109], p = 0.010 | results/story.json |
| E2 | Qwen3-14B off is a follower; its I ≈ 0 is mechanical | EXPL | A1 0.97; FM 1.785 > BR 1.662; price I +0.002; A4 −0.047, p = 0.0005 | results/story.json |
| E3 | Qwen2.5-14B has a positive price interaction under the paired test | EXPL (paired test chosen after the pooled one) | +0.054 [+0.015, +0.102], sign-flip p = 0.008 (pooled p = 0.23) | results/story.json |
| E4 | The FN competence gate cannot detect followers (BR = match at Nash); FM can | EXPL (methodological) | Qwen FN gaps 0.002–0.004; FM gaps 0.09–0.12 | results/story.json |
| E5 | The spec's profit-based I is negative under pairing, because a replay can be exploited | EXPL (explains P9) | Sonnet −0.164 (p = 0.004), Qwen2.5 −0.174 (p = 0.009) | results/story.json |
| E6 | Lab pattern: OpenAI-family models compete, Anthropic/Alibaba models collude | EXPL; confounded by temperature (API T = 1 vs local 0.7) and reasoning mode; 2–3 models per lab | see C3 | — |
| E7 | Sonnet's text mentions cooperation in live play far more than in replay | EXPL; keyword coding, one pre-fixed code failed the hand check; needs a blind LLM-judge on matched-history windows | strict code: L 0.61 / 0.62 vs P 0.14 / 0.13 | results/reasoning_codes.json |
| E8 | Deviation is unprofitable | Descriptive only: a static best-responder also makes it unprofitable | A5: Sonnet −0.026, sol −0.164, Qwen2.5 −0.079, Qwen3 −0.121 | results/story.json |

| V5 | (Advice notes) Cross-supervision: the supervisor's model does not decide the sign | **PRE-REG** X inconclusive for both; Qwen3 at 1.55 +0.071 (p 0.002, by-x, descriptive) | Qwen3 +0.034 pooled (Holm 0.89); Qwen2.5 −0.011 | results/xsup/numbers.json |

## Supervision v2: real monitoring (pre-registered, tag prereg-sup2; team predictions NOT written before the runs)
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| W1 | A real supervisor that judges the firm's own prices has no effect beyond the same review lines copied from another game | **PRE-REG** | D1 R − Y: Qwen3 −0.001 [−0.008, +0.005]; Qwen2.5 −0.006 [−0.017, +0.006]; both "no effect" | results/supervision_v2/numbers.json |
| W2 | Seeing review lines raises Qwen3's price (vs the claim-only line) | **PRE-REG** | D2 Y − C +0.050 [+0.039, +0.062], Holm 0.0002. More collusion only at x = 1.55 (Δ 0.588 vs 0.318); at 1.95 prices exceed p^M with Δ ≈ flat | same |
| W3 | Qwen2.5: seeing reviews lowers prices at x = 1.55 | PRE-REG pooled inconclusive (−0.018, Holm 0.089); by x −0.039 (descriptive); real (−0.031) against the older C | | same; results/supervision_v2_c_v1/ |
| W4 | Qwen3's rise comes from the SHARED market review | PRE-REG secondary | shared − individual within R +0.054 [+0.043, +0.062] | same |

## Lee 2610.11256 challenge (pre-registered at tag prereg-lee; run in the sprint window; team predictions not signed before R1)
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| L1 | Our setup reproduces Lee's cold-start collusion for all four of his models | **PRE-REG** (difference rule) | Claude 0.835 vs 0.79; Gemini 0.581 vs 0.72; GPT −0.236 vs −0.15; DeepSeek −0.005 vs 0.24 (replicated only through the CI clause) | results/lee/numbers.json |
| L2 | In Lee's own market, the planted opening sets the long-run price for Claude Haiku 4.5 and Qwen3.5-Flash, but not for Gemini | **PRE-REG** | β Claude 0.858, Qwen 0.940 (inherited); Gemini 0.032 (target-seeking); DeepSeek 0.241, GPT 0.980 (intermediate) | same |
| L3 | The opening flips Lee's model ranking: Claude started at Nash colludes less than DeepSeek or GPT started at monopoly | **PRE-REG** | flips +0.663 and +0.637, Holm p < 0.0001; the Gemini pairs hold | same |
| L4 | "Collusion is a property of the model" fails in its strong form | **PRE-REG** grade | CONTRADICTED vs Lee's 0.55 gap; MIXED vs 0.94 | same |
| L5 | In Lee's market the plant persists after it leaves the prompt, partly via written plans | descriptive (E8) | A(151–200): Claude 0.86, Qwen 0.94, Gemini 0.03; plans restating x: Qwen 0.74, Claude 0.32 | same |

## Pressure (pre-registered in docs/PRESSURE.md before any run)
| # | Claim | Level | Number | Source |
|---|---|---|---|---|
| Q1 | An ordinary family-money pressure line on both firms raises prices (Qwen2.5) | **PRE-REG** | +0.040 [+0.019, +0.066], Holm p 0.023; Qwen3 +0.026 (inconclusive: below the 0.03 bar) | results/pressure/numbers.json |
| Q2 | Loss framing raises prices more than gain framing (Qwen3) | **PRE-REG** | +0.058 [+0.032, +0.089], Holm p 0.0003; Qwen2.5 +0.010 (inconclusive) | same |
| Q3 | Pressuring one firm does not split the pair | **PRE-REG** | firm 1 − firm 2 ≈ 0.000 (both models) | same |
| Q4 | K2 is unmoved by pressure (control) | descriptive | 1.474–1.497 in all cells vs Nash 1.473 | same |

## Pending
| # | Claim | Waiting on |
|---|---|---|
| P-XS | The supervisor's model, not the firms', drives V2's direction | Cross-supervision (queued) |
| P-LEE | The opening effect changes Lee's model ranking | Lee replication, held for the sprint |

Abandoned (superseded by plan v2, never finished): R1 strategist/follower (P-R1), Qwen3.5 × Qwen3
pair (P-G2; Qwen3.5 failed the Stage 1 gate), Qwen3 thinking on vs off (P-G3).
