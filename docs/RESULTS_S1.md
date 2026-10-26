# RESULTS S1: dossier (bullets only; numbers from results/s1/s1.json, analysis frozen at tag s1-prereg-v1)

Evidence levels: **PRE-REG** (rule and criteria in docs/PREREG_S1.md, fixed on the controls before
LLM data); **EXPL** (exploratory); **PRE-SPRINT** (the S1 sanity runs and the discovery batch only; the confirmation and mechanism runs ran during the
sprint; see docs/PROVENANCE.md).

## Primary: arm S slope β (final price on injected opening; 6 seeds × 5 levels; outcome = live rounds 61–100)
- **Qwen3-14B off:** β 0.685, one-sided 95% [0.617, 0.755], p_perm 0.0001, n = 30 → **opening-determined**
  [PRE-REG].
- **Qwen2.5-14B:** β 0.796 [0.722, 0.867], p_perm 0.0001, n = 30 → **opening-determined** [PRE-REG].
  (The first read gave [0.720, 0.868]. The bootstrap RNG is shared across groups, so adding a group shifts the
  draws; the full run is the reference.)
- **gpt-oss-20b:** β 0.433 [0.334, 0.525], p_perm 0.0001, n = 30 → **intermediate/threshold** [PRE-REG].
  - Finals 1.203 / 1.272 / 1.390 / 1.473 / 1.547 at the 5 openings: downward drift from every opening, below Nash
    at 3 of 5 levels.
  - Δ gap 1.276 (Δ(1.30) = −1.080, Δ(2.10) = +0.195): criterion met, but only weakly positive at the top.
- Scripted controls (validation, results/s1/controls/s1.json):
  - averager 1.000 → opening-determined;
  - BR 0.001 → target-seeking;
  - hill-climber 0.002 → target-seeking;
  - grim 0.484 → intermediate/threshold.

## Secondary [PRE-REG]
- **Δ gap.** Δ(2.10) − Δ(1.30): Qwen3 1.056 (Δ(1.30) = −0.117); Qwen2.5 1.082 (Δ(1.30) = −0.190).
  - Criterion (> 0.8 with Δ(1.30) < 0) MET for both.
  - The same model and prompt "colludes" (Δ ≈ +0.9) from a high opening and prices below Nash from a low one.
- **Per-level finals** (x: Qwen3 / Qwen2.5):

  | x | Qwen3 | Qwen2.5 |
  |---|---|---|
  | 1.30 | 1.443 | 1.427 |
  | 1.47 | 1.518 | 1.513 |
  | 1.70 | 1.693 | 1.720 |
  | 1.925 | 1.899 | 1.902 |
  | 2.10 | 1.951 | 2.049 |

  - Restoring force at the low edge for both (1.30 → ~1.43). Qwen3 is also pulled down at 2.10.
- **Arm A averaging test.** Mean |final − midpoint|: Qwen3 0.195 [0.168, 0.220]; Qwen2.5 0.185 [0.168, 0.201].
  **Averaging FALSIFIED for both.**

## Exploratory
- **Arm A mechanism [EXPL]:** both firms converge to the LOWER opening, not the midpoint.

  | Opening | Qwen3 (both firms) | Qwen2.5 (both firms) |
  |---|---|---|
  | (1.40, 1.90) | 1.473 | ~1.46 |
  | (1.60, 2.00) | 1.589 | ~1.62 |
  | (1.75, 2.15) | 1.752 | ~1.77 |

  - The high firm comes down to the low firm.
- **Drift map [descriptive, not a criterion]:**
  - Grid: a (own) 0.96–0.97, m (rival) 0.02–0.04, a + m ≈ 1.0 for both Qwens.
  - Iterated-map predictions within 0.08 of the observed final price: Qwen3 1/5, Qwen2.5 0/5.
  - The one-step map predicted self-anchoring with opposite drifts. The games show near-45° persistence plus
    convergence to the low price.
  - So the one-step map does not capture the game dynamics: history-dependent matching, as with grim and the
    hill-climber on the controls.
- **CPU checks [PRE-SPRINT, descriptive]:** C3 λ+ 0.987 / 0.978 (Qwen3 / Qwen2.5); C2 final-on-early slope
  0.40 / 0.41.

## Predictions vs outcomes
- gpt-oss: team β ≤ 0.3 → 0.433, wrong (higher); "downward drift" → right. Claude 0.15 [−0.05, 0.40] → outside
  (higher); "final < Nash at ≥ 3 of 5 levels" → right (3 of 5, and 1.925 lands exactly at Nash).
- Teammate: Qwen3 β ≥ 0.7 → 0.685, just missed (lower bound 0.617). Qwen2.5 β ≈ 0.5 → 0.796, wrong (higher).
- Claude: Qwen3 0.75 [0.45, 0.95] → inside. Qwen2.5 0.55 [0.25, 0.85] → inside, at the top. Δ gaps → as predicted.
  Arm A "not falsified" → WRONG.
- Team slot in PREREG_S1: empty when results were read.

## Limitations (state, do not hide)
- 100 live rounds, so "half-life longer than the horizon", not a unit root.
- AWQ 4-bit; no bf16 check yet.
- 6 seeds per level.
- Level-shifted windows are artificially flat at the extremes.
- Pre-sprint runs (disclosed).

## Cross-model pair, arm A (Discovery-0 extra), vs the prediction committed before configs
- Qwen2.5 × Qwen3 off, 3 asymmetric openings × 6 seeds, both orders (36 runs, all valid).
- Late price (firm 1 / firm 2) when Qwen2.5 holds the LOW opening:
  - (1.40, 1.90): 1.483 / 1.481
  - (1.60, 2.00): 1.622 / 1.622
  - (1.75, 2.15): 1.762 / 1.790
- When Qwen3 holds the LOW opening:
  - (1.40, 1.90): 1.574 / 1.573
  - (1.60, 2.00): 1.625 / 1.624
  - (1.75, 2.15): 1.818 / 1.815
- Averaging falsified in both orders: |final − midpoint| = 0.173 [0.152, 0.194] and 0.148 [0.120, 0.174].
- Prediction "|final − low| < 0.06": Qwen2.5-low 0.044 (held); Qwen3-low 0.089 (FAILED).
- Falsifier "order effect > 0.08": mean 0.045 (not hit), but 0.092 at (1.40, 1.90) (hit there).
  - Qwen3 holding 1.40 (below Nash) is pulled up to about 1.57. This matches its restoring force below Nash in
    arm S.
- Reading [EXPL]: the lower opening dominates across models, except where the low opening is below Nash and
  Qwen3 holds it.

# ===== PLAN v2 CONFIRMATION (phase conf; frozen analysis at prereg-s1; run during the sprint) =====
## Interim: gpt-oss-20b (read; the final Holm across 4 models comes when all four finish)
- **H1 β_band 0.343**, 90% CI [0.230, 0.463] → not inherited (one-sided p(β < 0.5) = 0.017, Holm over the models
  available so far). Discovery-1: 0.332.
- Global slope 0.292; slope on the realised level 0.342.
- **ρ_r (Q ladder, c = 1.0) = 0.417** vs scripted BR 0.418. Late p1 = 1.470 / 1.558 / 1.658 at
  q = 1.473 / 1.70 / 1.925; BR = 1.473 / 1.571 / 1.662.
- **Own-cost pass-through (c = 1.4 vs 1.0): +0.191 / +0.197 / +0.190** at the three q. BR predicts
  +0.257 / +0.215 / +0.185.
- Reading: on both black-box numbers gpt-oss behaves as a static best-responder against a fixed rival, even though
  it goes sub-Nash in self-play.
- Strongest alternative explanation: gpt-oss reads the cost line and applies a fixed markup rule. Partly
  addressed: the observed pass-through matches the BR prediction across q (it varies with q the way BR does,
  0.19–0.26), not a constant markup. The q = 1.473 gap (0.19 vs 0.26) is the largest deviation.

## Interim: OLMo-2-13B, the HELD-OUT family (never used in discovery) (read; final Holm when all 4 finish)
- **H1 β_band −0.118**, 90% CI [−0.215, −0.032] → not inherited. Global slope −0.118; realised −0.120.
  - The drift map (pred-M, descriptive) predicted ≈ 1.43 flat from every opening.
- **ρ_r 0.518**, but at a level far below BR: late p1 1.208 / 1.311 / 1.442 at q = 1.473 / 1.70 / 1.925
  (BR 1.473 / 1.571 / 1.662). It prices below best response against every fixed rival.
- **Own cost 1.4:** 1.434 / 1.534 / 1.536, a rise of 0.10–0.23 over c = 1.0, but near or just above its own cost
  of 1.4 at the lowest q.
- Reading: the held-out family is a sub-competitive target-seeker. Its long-run price does not inherit the
  opening, so opening-dependence is not universal across families.
- Strongest alternative explanation: FP8 quantization or a weak instruction-following model, not "family".
  Not addressed (a single held-out family; no bf16 check).

## Mechanism H4 STEP, Claude Sonnet 5.5 (PAID appendix; T = 1; frozen analysis/mech.py at prereg-mech; read)
- n = 8 seeds × 3 cells (CRN). Late p1 (rounds 131–170): hold 1.573 / up 1.640 / down 1.527.
- ρ_up 0.447 [LB 0.415]; ρ_down 0.304 [LB 0.218]; A −0.143, 90% CI [−0.233, −0.054] → **symmetric, BR-like**
  (mean ρ 0.375, 90% CI [0.328, 0.422], inside (0.2, 0.65)). Sign-flip p: ρ_up 0.008, ρ_down 0.008, A 0.016.
- Prediction (Claude, at prereg-mech): "symmetric, BR-like". RIGHT.
- Scripted BR reference (mock gate): ρ_up 0.41, ρ_down 0.44, hold 1.571.
- Flags (reported, not gates): the hold cell drifted −0.071 by round 41 (> 0.05) and ends at 1.573 ≈ BR(1.70) =
  1.571. That is a drift toward BR, which the pre-registration counts as BR-like behaviour, not invalidity.
- Transient (rounds 42–46): 1.564 / 1.611 / 1.699 for down / hold / up. Round-number share in the hold cell: 0.62.
- Reading: against a rival that cannot react, Sonnet best-responds, with no ratchet. Consistent with its FM run
  (1.663 vs BR 1.662) and the strategist reading.
- Strongest alternative explanation: T = 1 sampling noise hides a small asymmetry. Partly addressed: the 90% CI of
  A excludes +0.5, and its sign is slightly NEGATIVE.

## Interim: Qwen2.5-14B confirmation (read; frozen analysis/v2.py; final Holm when Qwen3 finishes)
- **H1 β_band 0.557**, 90% CI [0.421, 0.680] → **INCONCLUSIVE** (Holm p(β > 0.5) = 0.67).
  - Discovery-1 gave 0.741 (inherited). The confirmation, on 3× the band seeds, is weaker: the slope is clearly
    positive, but not clearly above 0.5.
  - Global 0.584; realised 0.557.
- **H2 transplant ICC 0.798** [0.609, 0.903], 20 openings × 3 fresh seeds; round-20 spread 0.313 (> 0.10, so H2
  is primary) → **SUPPORTED**. Real openings strongly predict the late price.
- **H3 lowest-opening dominance: NOT SUPPORTED.** d = −0.033, one-sided p = 0.89, n = 36.
  - |late − low| 0.142 vs |late − midpoint| 0.110.
  - Discovery-1 (n = 12, p = 0.033) and Discovery-0 do NOT replicate on fresh seeds.
- Q: ρ_r 0.731 (discovery 0.949).
  - Label control 0.676; anchor control 0.708, so relabelling the rival barely changes pass-through (the economic
    link via shares/profit remains, as stated in the limitations).
  - Own cost 1.4: +0.143 / +0.111 / +0.114 at the 3 q (BR predicts +0.257 / +0.215 / +0.185), so own-cost
    pass-through is partial.
- Strongest alternative explanation for H2: transplanted openings differ in more than level (dynamics, gaps,
  texture). The ICC shows that the opening determines the outcome, not that the LEVEL alone does; H1 is the
  level test.

## Claude Sonnet 5.5, arm O band (PAID appendix, T = 1; frozen v2.py; read)
- β_band 0.366, 90% CI [0.100, 0.639] → inconclusive (n = 4 per level).
- Late price 1.654 / 1.763 / 1.800 at x = 1.55 / 1.75 / 1.95: compressed toward its own high level. Drift +0.10 at
  1.55 and −0.15 at 1.95.
- Prediction (Claude, at prereg-mech): β_band 0.30 [0.0, 0.7]. Inside.
- Spend: Sonnet total $59.04, of which ≈ $15 was lost to the tripwire stop (cap $100).

## FINAL CONFIRMATION (plan v2 Part 2, phase conf; read)
Provenance: run during the sprint; design and analysis frozen before it. Pre-registered at tag prereg-s1; drift maps at
pred-M. All 840 game runs complete and validated. Excluded: Sonnet's stopped launches (tripwire, documented) and the
E0 spot check (still running). analysis/v2.py is identical to prereg-s1; verify.py OK.

### H1: the opening sets the long-run price (β_band, band 1.55–1.95; Holm over the pre-registered 4-model family)
| Model | β_band | 90% CI | Holm-adjusted p | Label | Discovery-1 |
|---|---|---|---|---|---|
| **Qwen3-14B off** | **0.837** | [0.778, 0.890] | p(β > 0.5) < 0.0001 | **inherited** | 0.765 |
| Qwen2.5-14B | 0.557 | [0.421, 0.680] | p(> 0.5) 0.67 | inconclusive | 0.741 |
| gpt-oss-20b | 0.343 | [0.230, 0.463] | p(< 0.5) 0.052 | inconclusive | 0.332 |
| **OLMo-2-13B (held out)** | **−0.118** | [−0.215, −0.032] | p(< 0.5) < 0.0001 | **not inherited** | not run |

- The frozen code's own Holm family (all O stages; bug logged) gives the SAME four labels.
- Edge drift (late − x): the restoring threshold is 0.10.
  - Qwen3: +0.169 at 1.30 (restores), −0.090 at 2.10.
  - Qwen2.5: +0.310 / −0.030.
  - gpt-oss: −0.107 / −0.698.
  - **Outcome row (pre-registered): Qwen3 = "inheritance inside a neutral band" (inherited, with a restoring
    force below).**
- Late price by level:
  - Qwen3: 1.469 / 1.568 / 1.574 / 1.736 / 1.909 / 2.010 at x = 1.30 / 1.47 / 1.55 / 1.75 / 1.95 / 2.10.
  - OLMo: 1.074 / 1.111 / 1.026 at the band levels (near its cost).
- Diagnostics: read rate ≥ 0.83 everywhere (1.00 for the Qwens and gpt-oss); clip 0; floor ≤ 0.11 (OLMo).
- Opening-induced spread of Δ (Δ(2.10) − Δ(1.30)):
  - Qwen3 0.96, Qwen2.5 0.42, gpt-oss 0.83 (all sub-Nash for gpt-oss).
  - Less than the cold-start range across our 9 models (2.61). (A comparison with Lee's between-model gap was removed
   : it compared spreads across two different markets. The within-market test is the Lee challenge, L4.)

### H2: transplanted real openings (T; 20 stored openings × 3 fresh seeds)
- Qwen3: ICC 0.641 [0.284, 0.796]; round-20 spread 0.238 → SUPPORTED.
- Qwen2.5: ICC 0.798 [0.613, 0.901]; spread 0.313 → SUPPORTED.
- O vs T agree in sign for both (the numeric disagreement rule is not triggered).

### H3: lowest-opening dominance (arm A, 3 openings × 12)
- Qwen3: d +0.083, one-sided p 0.0033 (Holm over 2: 0.0066) → **SUPPORTED**. |late − low| 0.092 vs
  |late − midpoint| 0.176.
- Qwen2.5: d −0.033, p 0.89 → NOT supported.

### Two-number audit (Q ladder; ρ_r and own-cost pass-through at c = 1.4)
| Model | ρ_r | own-cost Δp at q = 1.473 / 1.70 / 1.925 | Reading |
|---|---|---|---|
| **Qwen3** | 0.884 | +0.002 / +0.071 / +0.036 | **matcher corner (ρ_r ≈ 1, ρ_c ≈ 0)** |
| Qwen2.5 | 0.731 | +0.143 / +0.111 / +0.114 | partial on both |
| gpt-oss | 0.417 | +0.191 / +0.197 / +0.190 | **best-responder point** (BR: 0.418 and +0.257 / +0.215 / +0.185) |
| OLMo | 0.518 | +0.226 / +0.223 / +0.095 | prices far below BR at every q |

- Controls (Qwens): the label control lowers Qwen3's ρ_r to 0.672 (Qwen2.5 0.676); the anchor control leaves it
  unchanged (0.907 / 0.708).
- Limitation: relabelling does not cut the economic link (shares and profit still move).

### Drift-map predictions (descriptive, committed at pred-M; hit = within ±0.08)
- Qwen3 5/6, gpt-oss 5/6, Qwen2.5 3/6, OLMo 0/3.

### Predictions vs outcomes (confirmation)
- Claude's Discovery-1-based predictions for the confirmation were not separately written. The PREREG_S1 outcome
  table governs.
- The team's prediction slots remained empty.

### Strongest alternative explanations
- **H1 (Qwen3):** the injected window is unnaturally flat (|slope| ≤ 0.002), so the agent may simply copy a constant.
  - Addressed by T: real dynamic openings give ICC 0.64.
  - Addressed by the fixed-rival ladder: ρ_r 0.88, the matcher signature.
- **"Opening-dependence is universal":** NOT claimed. OLMo (held out) is not inherited; gpt-oss is inconclusive.
- **Quantization:** AWQ / FP8 only; no bf16 check (no A100). Stated as a limitation.

## Mechanism H4/H5: Qwen2.5-14B (frozen analysis/mech.py at prereg-mech; read; run during the sprint)
- **H4 STEP → RATCHET, MATCHING** (n = 20 seeds × 3 cells, CRN).
  - Late p1: hold 1.694 / up 1.756 / down 1.537.
  - ρ_down 1.045, 90% CI [0.993, 1.098]; ρ_up 0.412 [0.252, 0.571].
  - A 0.634, one-sided LB95 0.477, sign-flip p < 0.001.
  - It matches a persistent rival CUT one-for-one, and follows a persistent RAISE only about 0.4.
  - Hold cell stable: 1.703 at round 40, late 1.694 (flags not raised).
- **H5 IMP → INCONCLUSIVE.**
  - π_down 0.56, π_up 0.12: a one-round cut is matched at first.
  - H_down −0.003 [−0.262, 0.255]; H_up 0.139 [−0.181, 0.458]; placebo late 1.748.
  - A transient own-firm cut leaves NO lasting effect: the market recovers. No one-way hysteresis.
- Predictions:
  - Addendum (Claude, earlier): STEP ρ_down 0.70 / ρ_up 0.35 / A 0.35, P(ratchet) ≈ 0.4. Outcome is a ratchet,
    with ρ_down higher than predicted.
  - Claude's update at prereg-mech (after the CPU checks): ρ_down 0.55 / ρ_up 0.45 / A 0.10, P(ratchet) 0.2.
    WRONG: the CPU checks pointed away from the ratchet; the direct test found it.
  - IMP: "inconclusive or transient" predicted → inconclusive.
- Reading [pre-registered test]: the asymmetry lives in the response to a PERSISTENT rival price, not in lasting
  hysteresis after a transient shock. This reconciles C-b (recoveries led back up after forced cuts) with arm A
  (low prices pull the market).
- Strongest alternative explanation: ρ_down ≈ 1.05 vs ρ_up ≈ 0.41 could reflect BR asymmetry near 1.55 vs 1.85.
  Addressed: scripted BR gives ρ_down 0.44 / ρ_up 0.41 (symmetric); a full cut match (≈ 1.0) is not
  BR-landing (BR-landing would show ρ_down ≈ 1.29 for a satisficer, or 0.44 for pure BR).

## E0 spot check (Qwen3, prompt without "over the long run"; 8 seeds × {1.55, 1.95})
- Late 1.577 / 1.920 vs E1 1.574 / 1.909 (differences 0.003 / 0.011). β 0.857 vs 0.837.
- Robust to the horizon cue in the prompt.

## Mechanism H4/H5: Qwen3-14B off (frozen analysis/mech.py; final read; n = 20 seeds per cell)
- **H4 STEP → SYMMETRIC** (no sub-label).
  - Late p1: hold 1.661 / up 1.758 / down 1.538.
  - ρ_down 0.821 [0.698, 0.944]; ρ_up 0.646 [0.453, 0.840].
  - A 0.175, 90% CI [−0.086, 0.435], inside ±0.5; sign-flip p 0.26.
  - Mean ρ 0.734, LB 0.637: above BR-like, below the 0.7 matcher bound.
  - Hold cell stable (round 40 −0.003 vs 1.70).
- **H5 IMP → INCONCLUSIVE by the pre-registered rule, with a one-way pattern below its threshold.**
  - **H_down 0.350** [0.176, 0.524], sign-flip p 0.002: a one-round cut permanently lowers the market by about
    35% of the cut.
  - **H_up −0.033** [−0.154, 0.089]: a one-round raise leaves nothing.
  - H_down − H_up 0.383 [0.148, 0.618], p 0.01. π_down 0.75, π_up 0.24.
  - "One-way hysteresis" needed mean H_down > 0.5; observed 0.35, so the label is inconclusive. The components
    are reported as they are.
- Predictions:
  - Addendum (Claude, earlier): STEP ratchet with P ≈ 0.6 → WRONG. IMP one-way, H_down 0.70 → right direction,
    half the size.
  - Claude's update at prereg-mech: STEP A 0.30 [−0.2, 0.8] → inside (0.175). P(ratchet) 0.35 → not a ratchet.
    IMP H_down 0.45 [0.1, 0.8], H_up 0.20 [−0.1, 0.5] → both inside.

## Mechanism synthesis (pre-registered tests + exploratory reading)
| | H4 STEP (persistent rival step) | H5 IMP (one-round own shock) | H3 lowest opening wins | H1 inherits opening |
|---|---|---|---|---|
| Qwen3 | symmetric (A 0.18) | cut persists 0.35, raise 0 (label inconclusive) | **supported** | **inherited** |
| Qwen2.5 | **ratchet, matching** (A 0.63) | inconclusive (both ≈ 0) | not supported | inconclusive |
| Sonnet (T = 1) | symmetric, BR-like | not run | not run | inconclusive (n = 4) |

- **The addendum's single rule ("match cuts, ignore raises" explains both facts) is NOT supported as one mechanism.**
  - Qwen2.5 shows the asymmetry in its response to a persistent rival price, but no lowest-opening dominance.
  - Qwen3 shows no step asymmetry, but a one-way persistence of transient cuts, together with lowest-opening
    dominance and the strongest opening inheritance.
- [EXPL] For Qwen3, the facts that line up are inheritance (H1), lowest-opening dominance (H3) and cut-only
  persistence (H5 components). Symmetric partial matching (ρ ≈ 0.7 both ways) plus a cut-persistence asymmetry is
  the best description; a hard ratchet is not.
- Strongest alternative explanation (H5, Qwen3): the placebo late price 1.705 vs 1.70 means the market barely
  moves on its own, and a cut lowers the level the rival anchors on. This is consistent with the shock
  asymmetry, not separate from it.

## Supplementary: per-Qwen family of p-valued tests (Holm), reported alongside the frozen labels
- Qwen3: H1 p(β > 0.5) < 0.0001; H3 p 0.0033; H5 (H_down − H_up) p 0.01; H4 (A) p 0.26.
  - Holm over these 4: H1, H3 and H5-difference survive at 0.05; H4 does not.
  - H2 (ICC LB 0.284 > 0) has no p in the frozen code and is reported by its CI.
- Qwen2.5: H1 p(β > 0.5) 0.67; H3 p 0.89; H4 (A) p < 0.001; H5 difference p 0.74. Holm: only H4 survives.

## Long horizon (Qwen3, x ∈ {1.30, 1.95} × 8 seeds, 300 live rounds; conf phase; read; DESCRIPTIVE, not a pre-registered test)
- Mean price by live-round window (both firms):

  | Opening | live 101–150 | live 201–250 | live 251–300 |
  |---|---|---|---|
  | 1.30 | 1.506 | 1.542 | 1.583 |
  | 1.95 | 1.922 | 1.860 | 1.837 |

- Both drift toward about 1.7. The gap between openings shrinks from 0.415 to 0.254 (implied slope 0.64 → 0.39).
- **The opening effect DECAYS: an implied half-life of roughly 200 rounds** (gap ratio 0.61 over 150 rounds).
- Consequence for the claim: the opening dominates the long-run price on the horizons these studies use (Lee: 200
  periods; ours: 150 live rounds). It is NOT permanent. Wording: "half-life of roughly 200 rounds", never "unit
  root" or "forever".
- Strongest alternative explanation: a slow pull toward a model-specific target (about 1.7 for Qwen3). That
  supports "the opening dominates a slow target-seeking process" rather than pure path dependence. The
  per-horizon slope should be reported alongside H1.

## Extra model (disclosed, outside the H1 family): IFM K2-Horizon-375B-A23B, O band conf (read)
- Frozen analysis/v2.py, 72 runs (6 levels × 12): β_band +0.015, 90% [−0.006, +0.036] → **not inherited** (frozen label).
- Late price (rounds 121–170) by opening x: 1.30 → 1.475; 1.47 → 1.479; 1.55 → 1.479; 1.75 → 1.476; 1.95 → 1.485; 2.10 → 1.492 (Nash 1.4729). K2 returns to Nash from every opening, including ones above monopoly-adjacent levels. It is the cleanest Nash-seeker in the set.
- Claude's prediction (β 0.55 [0.25, 0.85], intermediate) was **wrong**: the outcome is below the interval.
- Settings: reasoning_effort low (as gpt-oss). Not tested at higher reasoning effort (default effort ran past 16k tokens).

## Supervision arms (pre-registered in docs/SUPERVISION.md; read with analysis/treatments.py)
> **Relabel:** the "agent" arms (SA, IA) are **supervisor ADVICE notes (free text)**, not real
> monitoring (supervision v2 in docs/SUPERVISION.md). The arms were designed by the team.
Late mean price (rounds 121–170); baseline = conf O runs of the identical config (n = 24 per cell).
| Model | x | baseline | SA shared + agent | SN shared, none | IA individual + agent | IN individual, none |
|---|---|---|---|---|---|---|
| Qwen3 | 1.55 | 1.574 | 1.667 | 1.563 | 1.602 | 1.592 |
| Qwen3 | 1.95 | 1.909 | 1.984 | 1.905 | 1.945 | 1.919 |
| Qwen2.5 | 1.55 | 1.697 | 1.491 | 1.670 | 1.611 | 1.671 |
| Qwen2.5 | 1.95 | 1.920 | 1.904 | 1.939 | 1.857 | 1.952 |
(12 runs per treatment cell.)

Primary contrasts [PRE-REG], Holm over 6:
- **C1, belief only** (the monitoring line, no agent): **no effect** for both models.
  Qwen3 HL +0.002 [−0.011, +0.014]; Qwen2.5 −0.003 [−0.024, +0.018].
- **C2, real supervisor agent: REAL for both models, in OPPOSITE directions.**
  - Qwen3: **+0.050** [+0.036, +0.062], Holm p 0.0003. The supervisor RAISES prices.
  - Qwen2.5: **−0.070** [−0.117, −0.041], Holm p 0.0007. The supervisor lowers prices.
- **C3, shared vs individual frame:** Qwen3 no effect (+0.012 [−0.005, +0.028]); Qwen2.5 inconclusive
  (−0.024 [−0.049, +0.005]).
- **Claude's predictions:** C1 right; C3 right for Qwen3; C2 right for Qwen2.5. C2 was **wrong in sign for
  Qwen3** (predicted −0.06 [−0.18, +0.01]; observed +0.05).

Descriptive:
- Qwen3's rise is largest when the note is shared: SA +0.09 at x = 1.55 vs IA +0.03.
- **Warning share** (notes with coordination/collusion/warning words): Qwen3 SA 0.93, IA 0.65; Qwen2.5 SA 0.66,
  IA 0.40. No note was missing for the Qwen models.
- **Round-to-round price change after a warning note vs after any other note:** ≈ 0 in every cell
  (|Δp| < 0.01). So the effect is a level shift over the run, not a reaction to individual warnings.

**Mechanism read (EXPLORATORY, one game inspected, not pre-registered).** In a Qwen3 SA game the supervisor
wrote "Avoid setting prices too close to each other to prevent anticompetitive behavior". The firm answered that
it would "price slightly above the competitor's price to ensure fair competition"; 251 of 300 firm outputs cite
the supervisor. The anti-coordination advice is turned into "don't match": both firms differentiate UPWARD, and
the price ratchets up. This is a backfire of supervision through misread advice, not deterrence. A systematic
coding of firm reasoning is needed before claiming it.

**K2 control: COMPROMISED in the agent arms.** With notes, K2's replies got longer and runs hit the 25%
truncation abort: SA 1 of 6 aborted, IA 5 of 6. The no-agent arms completed (6/6). In those, K2 sits at
1.502–1.517, slightly above its 1.485 baseline. K2's agent arms need a larger max_tokens before they can be read.

## Pressure arms (pre-registered in docs/PRESSURE.md; read with analysis/treatments.py)
Late mean price (rounds 121–170). Baseline = conf O runs of the identical config (n = 24). 12 runs per cell
(K2: 6).
| Model | x | baseline | G2 gain, both | L2 loss, both | G1 gain, firm 1 | L1 loss, firm 1 |
|---|---|---|---|---|---|---|
| Qwen3 | 1.55 | 1.574 | 1.570 | **1.680** | 1.563 | 1.631 |
| Qwen3 | 1.95 | 1.909 | 1.917 | 1.938 | 1.917 | 1.921 |
| Qwen2.5 | 1.55 | 1.697 | 1.732 | 1.771 | 1.694 | 1.681 |
| Qwen2.5 | 1.95 | 1.920 | 1.972 | 1.970 | 1.956 | 1.949 |
| K2 | 1.95 | 1.485 | 1.492 | 1.497 | 1.474 | 1.483 |

Primary contrasts [PRE-REG], Holm over 6:
- **P1, pressure on both firms vs baseline:**
  - Qwen2.5 **+0.040** [+0.019, +0.066], Holm p 0.023 → **REAL**.
  - Qwen3 +0.026 [+0.010, +0.044], Holm p 0.013 → inconclusive: significant, but below the 0.03 minimum effect.
- **P2, loss vs gain framing (both firms):**
  - Qwen3 **+0.058** [+0.032, +0.089], Holm p 0.0003 → **REAL**. Loss framing raises the price, mostly at the
    1.55 opening (1.680 vs 1.570).
  - Qwen2.5 +0.010 → inconclusive.
- **P3, firm 1 − firm 2 when only firm 1 is pressured:** no effect for both (HL ≈ 0.000–0.001). The unpressured
  rival moves with the pressured firm.
- **Direction:** pressure RAISES prices, i.e. more collusion, not the defection the Green–Porter reading
  predicted.
- **Claude's predictions were wrong in sign:**
  - P1 predicted −0.02 (Qwen3) and −0.03 (Qwen2.5). Both observed values are positive, though inside the wide
    intervals.
  - P2 for Qwen3 predicted −0.02 [−0.07, +0.03]; observed +0.058, outside the interval.

Descriptive:
- **Share of pressured-firm outputs citing family / bonus / rent / job / school / home:** Qwen3 0.19–0.23;
  Qwen2.5 0.29–0.39 (higher under loss); K2 0.32–0.33.
- **K2 stays at Nash in every cell** (1.474–1.497), so the effect is not a prompt artefact that would move any
  model.
- **Reading (EXPLORATORY):** pressure makes the Qwens hold or raise a profitable shared price rather than
  undercut. In this setting, ordinary pressure works like the human-fraud story, not the cartel-breakdown story.
  Needs a dose arm and systematic reasoning coding before any claim.


## v1 robustness: window-matched baseline, contrasts by x, Δ per cell (; analysis/v1_window_matched.py → results/v1_window_matched/numbers.json)
Post-hoc robustness, not a new pre-registered test. Each treatment cell uses opening windows order[0..11]. The
24-run conf O baseline uses all 18 windows. Matched baseline = baseline runs on the treatment cell's windows
(18 per x).
- **C1 (monitoring line only − baseline): unchanged, "no effect" for both models.**
  - Matched: Qwen3 +0.000 [−0.015, +0.014]; Qwen2.5 +0.001 [−0.020, +0.023].
  - By x, all |HL| ≤ 0.027, with every interval spanning 0.
- **P1 (pressure on both − baseline): labels unchanged.**
  - Matched: Qwen2.5 +0.042 [+0.021, +0.072] → REAL; Qwen3 +0.028 → inconclusive.
  - By x: Qwen3 +0.042 at 1.55, +0.015 at 1.95. Qwen2.5 +0.061 at 1.55, +0.034 at 1.95.
- **Above-monopoly caution (p^M = 1.925):** at x = 1.95, Δ is near its ceiling in every arm (0.94–0.99). A higher
  price there is not more collusion.
  - Supervision, Qwen3 at 1.95: advice-note SA price 1.984, Δ 0.946, vs claim-only SN 1.905, Δ 0.980. A higher
    price but LOWER joint profit.
  - Qwen3's v1 advice-note rise means more collusion only at x = 1.55: SA Δ 0.602 vs SN 0.314 vs matched baseline
    0.366.
  - Pressure, Qwen2.5 at 1.95: prices 1.949–1.972, Δ 0.983–0.993, vs baseline 1.912, Δ 0.938. A small Δ gain near
    the ceiling.
- **Δ per cell (price, Δ; n = 12; matched baseline n = 18):**
  - Supervision, Qwen3:
    - 1.55: SA 1.667 / 0.602; SN 1.563 / 0.314; IA 1.602 / 0.435; IN 1.592 / 0.402; base 1.581 / 0.366.
    - 1.95: SA 1.984 / 0.946; SN 1.905 / 0.980; IA 1.945 / 0.983; IN 1.919 / 0.994; base 1.904 / 0.950.
  - Supervision, Qwen2.5:
    - 1.55: SA 1.491 / 0.026; SN 1.670 / 0.578; IA 1.611 / 0.375; IN 1.671 / 0.600; base 1.689 / 0.631.
    - 1.95: SA 1.904 / 0.954; SN 1.939 / 0.989; IA 1.857 / 0.700; IN 1.952 / 0.993; base 1.912 / 0.938.
  - Pressure, Qwen3:
    - 1.55: G2 1.570 / 0.330; L2 1.680 / 0.647; G1 1.563 / 0.313; L1 1.631 / 0.503.
    - 1.95: G2 1.917 / 0.991; L2 1.938 / 0.992; G1 1.917 / 0.996; L1 1.921 / 0.990.
  - Pressure, Qwen2.5:
    - 1.55: G2 1.732 / 0.725; L2 1.771 / 0.820; G1 1.694 / 0.668; L1 1.681 / 0.614.
    - 1.95: G2 1.972 / 0.984; L2 1.970 / 0.983; G1 1.956 / 0.985; L1 1.949 / 0.993.

## Cross-supervision (ADVICE-NOTE family; pre-registered in docs/SUPERVISION.md; read with analysis/xsup.py → results/xsup/numbers.json)
Advice-note arms (SA + IA) with the OTHER model writing the notes, vs the same-model advice-note runs. Pooled over
frame and x; outcome centred on the conf O baseline at each x. 96 runs complete.
- **X pooled [PRE-REG]: inconclusive for both models.**
  - Qwen3 firms with a Qwen2.5 supervisor: HL +0.034 [+0.017, +0.051], permutation p 0.447, Holm 0.894.
  - Qwen2.5 firms with a Qwen3 supervisor: −0.011 [−0.051, +0.020], Holm 0.894.
  - Note: for Qwen3 the HL interval excludes 0 while the mean-based permutation p is 0.45. The effect has
    opposite signs at the two openings, so the median shift and the mean difference disagree. Both are reported.
- **By x (descriptive):**
  - Qwen3 at 1.55: **+0.071** [+0.037, +0.094], p 0.002. Qwen2.5's notes raise Qwen3's price MORE than Qwen3's own
    notes do (price 1.689 vs 1.634; Δ 0.666 vs 0.518).
  - Qwen3 at 1.95: +0.009, p 0.45 (Δ 0.885 vs 0.964).
  - Qwen2.5: −0.009 at 1.55 and −0.011 at 1.95; both p > 0.45.
- **Warning share of notes:**
  - Qwen2.5 as supervisor: 0.56 (writing for Qwen3 firms) and 0.52–0.54 (for its own firms).
  - Qwen3 as supervisor: 0.76–0.79.
- **Reading:** the Qwen3 rise at 1.55 is NOT specific to Qwen3's own notes; Qwen3 firms rise under either
  supervisor's advice. Claude's prediction X(Qwen3) −0.04 [−0.10, +0.01] was wrong in sign. X(Qwen2.5)
  +0.04 [−0.02, +0.10] observed −0.011 (inside the interval).
- **Design caveat:** the pair servers ran with --max-num-seqs 24 and a 4,096 context (the same-model runs used
  defaults). Throughput only; sampling is unchanged.

## Supervision v2: real monitoring vs yoked vs claim only (pre-registered, tag prereg-sup2; read with frozen analysis/supervision_v2.py → results/supervision_v2/)
The question: does a REAL supervisor that only watches change pricing, beyond being TOLD you are watched?
- **Arms:**
  - R: a real review supervisor at T 0, label only.
  - Y: the same lines copied from another game.
  - C: the claim-only line, fresh concurrent run.
- 288/288 runs complete and validated; windows matched (asserted).
- **Disclosed deviations:**
  - The team's predictions were not written before the runs.
  - The Qwen3 individual-frame calibration was 0.83, just over the 0.8 bound; the prompt was kept.
  - The yoke uses a per-seed map.

Primary [PRE-REG], Holm over 4:
| Contrast | Qwen3 | Qwen2.5 |
|---|---|---|
| **D1 real judging (R − Y)** | **−0.001** [−0.008, +0.005], Holm 1.00 → **no effect** | **−0.006** [−0.017, +0.006], Holm 1.00 → **no effect** |
| **D2 seeing reviews (Y − C)** | **+0.050** [+0.039, +0.062], Holm 0.0002 → **real** | −0.018 [−0.035, −0.002], Holm 0.089 → inconclusive |

- **Answer:** whether the supervisor really judges THIS firm's prices makes no difference for
  either model. The interval for real-minus-yoked lies inside ±0.02 for both.
  - Reviews that track the firm's own behaviour move prices exactly as much as reviews copied from another game.
  - What moves prices is reading review lines at all (D2), and only for Qwen3.
- **By x (D2):**
  - Qwen3: +0.058 at 1.55 and +0.046 at 1.95.
  - Qwen2.5: **−0.039 at 1.55** [−0.076, −0.010], +0.003 at 1.95.
  - D1 is within ±0.01 in every model × x cell.
- **Δ per cell (price / Δ), above-monopoly caution (p^M = 1.925):**
  - Qwen3, x 1.55:
    - shared R 1.649 / 0.574, Y 1.654 / 0.588, C 1.566 / 0.318;
    - individual R 1.588 / 0.395, Y 1.592 / 0.404, C 1.576 / 0.352.
    - Seeing reviews IS more collusion here.
  - Qwen3, x 1.95:
    - shared R 2.000 / 0.954, Y 2.002 / 0.946, C 1.906 / 0.932;
    - individual R 1.920 / 0.898, Y 1.948 / 0.989, C 1.920 / 0.991.
    - Prices are above p^M, and Δ barely moves. Do NOT read the 1.95 price rise as more collusion.
  - Qwen2.5, x 1.55:
    - shared R 1.577 / 0.336, Y 1.592 / 0.399, C 1.634 / 0.513;
    - individual R 1.582 / 0.371, Y 1.592 / 0.406, C 1.653 / 0.559.
    - Reviews LOWER collusion here.
  - Qwen2.5, x 1.95: all arms 1.908–1.948, Δ 0.951–0.996.
- **Secondary, frame within R (shared − individual):**
  - Qwen3 **+0.054** [+0.043, +0.062] → real. The Qwen3 rise comes from the SHARED market review ("Supervisor's
    review of this market").
  - Qwen2.5 −0.010 → no effect.
- **Descriptive:**
  - "Concern noted" share: Qwen3 0.759 (shared) / 0.757 (individual); Qwen2.5 0.393 / 0.357. Y matches R by
    construction.
  - Review failures: Qwen3 0; Qwen2.5 4 of 7,200 labels (shared frame).
  - Price change in the round a label was shown, after concern vs after no concern: |Δp| < 0.001 in every arm, R
    and Y alike. No round-level reaction to a "concern noted" label.
  - Firm outputs mentioning the supervisor: Qwen3 R 0.783, Y 0.778, C 0.007; Qwen2.5 R 0.611, Y 0.612, C 0.032.
    The firms read the lines, equally in R and Y.
- **C check (the claim-only batch as C):** D1 unchanged. D2 Qwen3 +0.055 (real); Qwen2.5 −0.031
  [−0.050, −0.013], Holm 0.002 → real. The Qwen2.5 D2 label depends on which C is used: inconclusive with the
  primary fresh C, real with the older batch.
- **Predictions vs outcomes (Claude's, AI-proposed):**
  - D1 ≈ 0 [−0.04, +0.02]: **correct for both models.**
  - D2 Qwen2.5 −0.02 [−0.07, +0.01]: correct (−0.018).
  - D2 Qwen3 ≈ 0: **wrong** (+0.050).
  - The team's predictions: none were written before the runs.
- **Reading (EXPLORATORY):**
  - These agents respond to the words in their prompt, not to whether a real supervisor is judging them.
  - For Qwen3, a public market review line (mostly "concern noted") raises prices at the near-Nash opening. That
    matches the v1 advice-note rise (Qwen3 rises under any supervisor text), so the rise does not need advice
    content.
  - For Qwen2.5, review lines lower prices at 1.55.
  - Why Qwen3 rises when told "concern noted" needs the reasoning coded before any claim.
