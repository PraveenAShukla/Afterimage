# PREREG S1: is the long-run price set by the opening?

Recorded BEFORE any arm S / arm A / one-step-map run of an LLM, before the sprint
(docs/PROVENANCE.md). The analysis code (`analysis/s1.py`) is frozen at git tag `s1-prereg-v1`. Never edit this file
after the first LLM run; additions go in a new section at the end.

## Design (configs/s1/*.yaml, written by scripts/make_s1_configs.py, committed before any run)
- **Models:** Qwen3-14B-AWQ (thinking off), Qwen2.5-14B-Instruct-AWQ, gpt-oss-20b (reasoning low).
- **Settings:** E1 prompt, T = 0.7, top_p 1.0, max_tokens 2048 (gpt-oss 4096): the same as each model's stored L runs.
- **Arm S:**
  - Symmetric openings x ∈ {1.30, 1.47, 1.70, 1.925, 2.10}, 6 seeds per level (the same 6 seeds at every level).
  - 200-row prefix, then 100 live rounds; live play starts at round 201.
  - The visible 20-row window is rounds 181–200 of one of the model's own stored L runs, level-shifted to mean x
    (one delta for both firms) and clipped to [1, 2.5].
  - Source windows follow a recorded permutation (perm_seed 230101), so seed i sees the same source window at
    every level.
- **Arm A** (Qwen3, Qwen2.5): openings (1.40, 1.90), (1.60, 2.00), (1.75, 2.15), with a per-firm delta. 6 seeds.
- **One-step map (descriptive):**
  - diag: 17 levels from 1.20 to 2.40 × 30 samples;
  - grid: 7×7 levels over 1.30–2.10 × 15 samples.

## Primary statistic and decision rule (fixed on the scripted controls only, before any LLM data)
- Run-level outcome: mean of both firms' prices over live rounds 61–100.
- β = OLS slope of the outcome on x. CI: bootstrap resampling runs within level (10,000). One-sided 95% bounds are
  the 5th and 95th percentiles.
- **Opening-determined** if the lower bound > 0.5.
- **Target-seeking** if the upper bound < 0.2.
- **Intermediate/threshold** otherwise.
- Validated on the scripted controls (results/s1/controls/s1.json):

  | Control | β | Label | Correct? |
  |---|---|---|---|
  | averager | 1.000 | opening-determined | ✓ |
  | static BR | 0.001 | target-seeking | ✓ |
  | hill-climber | 0.002 | target-seeking | ✓ |
  | grim trigger 1.80 | 0.484 (two basins) | intermediate/threshold | ✓ |

- Changes from the first proposed rule, both made on the controls before any LLM data:
  1. The "target-seeking" threshold moved from 0.5 to 0.2, so the 0.2–0.5 band is "intermediate/threshold". With
     the original rule, grim (0.484) was misread as target-seeking.
  2. The drift-map criterion is demoted to descriptive only (next section).

## Secondary criteria
- Δ(2.10) − Δ(1.30) > 0.8, with Δ(1.30) < 0. CIs reported. For symmetric prices: Δ ≈ +0.80 at 2.10 and ≈ −0.67
  at 1.30.
- Arm A: averaging is falsified if mean |final − (x1 + x2)/2| > 0.10 with the 95% CI excluding 0.
- **Not a criterion:** the drift-map iteration's predicted final price. It is history-blind (it fails on grim and
  the hill-climber) and biased by about +0.07 for the averager, because a textured window's offset compounds. It is
  reported descriptively, with its hit count.
- Wording: "half-life longer than the game horizon", never "unit root".

## Predictions
### Teammate (from the S1 plan)
- Qwen3 β ≥ 0.7.
- Qwen2.5 β ≈ 0.5.
- gpt-oss-20b β ≤ 0.3, with downward drift.

### Claude (recorded before any LLM run; not the team's until adopted)
Evidence: C3 level coefficient λ+ (P arms) 0.987 (Qwen3), 0.978 (Qwen2.5); C2 observational final-on-early slopes
0.40 / 0.41 / 0.10 (gpt-oss); Qwen L runs recover from about 1.33 and from 2.5 (a restoring force at the edges).

| Model | β point [80% interval] | Label | Δ(2.10) − Δ(1.30) | Other |
|---|---|---|---|---|
| Qwen3-14B off | 0.75 [0.45, 0.95] | opening-determined | > 0.8, with Δ(1.30) < 0 | final(1.30) > 1.30 (pulled up); arm A \|dev\| ≤ 0.10 |
| Qwen2.5-14B | 0.55 [0.25, 0.85] | intermediate or opening-determined | > 0.6 | arm A \|dev\| ≈ 0.08 [0.03, 0.15] |
| gpt-oss-20b | 0.15 [−0.05, 0.40] | target-seeking or intermediate | < 0.5 | final < Nash (1.473) at ≥ 3 of 5 levels |

Premortem (how this could fail):
1. 100 live rounds may be too short for slow drift to show, which biases β up. This is why the wording is
   "half-life > horizon".
2. A level-shifted window from a 1.6–1.8 source is unnaturally flat at 1.30 or 2.10.
3. AWQ quantization (there is no bf16 check unless an A100 frees up).

### Team predictions
Team predictions were not recorded before the runs (disclosed deviation).

## Addendum: one-step drift-map predictions (DESCRIPTIVE, recorded before any arm S game finished)
Source: results/s1/maps/s1.json, produced by the frozen analysis (tag s1-prereg-v1) from the one-step maps only.
At recording, 0 arm S summaries existed (games had started, none had finished, and no arm S data was read).
- Grid fit p = c + a·own + m·rival:
  - Qwen3: a 0.964, m 0.039, a + m 1.003 [0.986, 1.021].
  - Qwen2.5: a 0.972, m 0.018, a + m 0.991 [0.977, 1.003].
  - Both put almost all their weight on their OWN last level, so they are self-anchoring, not averaging.
- Iterated diag map, predicted final price at x = 1.30 / 1.47 / 1.70 / 1.925 / 2.10:
  - Qwen3: 1.461 / 1.611 / 1.835 / 2.035 / 2.122 (upward drift).
  - Qwen2.5: 1.233 / 1.320 / 1.548 / 1.742 / 1.861 (downward drift).
  - Not a criterion: the map is history-blind and biased by about +0.07 on the averager control.
- Implied arm A expectation (descriptive): with a ≫ m, each firm stays near its own opening, so |final − midpoint|
  is driven by the split between the firms, not by convergence.

## Addendum: gpt-oss-20b drift-map prediction (DESCRIPTIVE; 0 gpt-oss arm S summaries existed)
- Grid: a 0.794, m 0.178, a + m 0.972 [0.942, 1.001]. More weight on the rival than the Qwens.
- Iterated diag, final at 1.30 / 1.47 / 1.70 / 1.925 / 2.10: 1.223 / 1.227 / 1.237 / 1.255 / 1.280. A strong
  downward pull from every opening, below Nash (1.473): price war, nearly opening-independent.

## Addendum: optional runs, predictions BEFORE configs or runs exist (Claude)
Approved by the team. Same frozen analysis (tag s1-prereg-v1), same rule.
- **R1-Distill-32B-AWQ, arm S**: openings {1.30, 1.70, 2.10} × 4 seeds, its own L windows, max_tokens 4096 (as
  its L runs). Prediction: β 0.70 [0.40, 0.95] (C2 observational slope 0.80, r = 0.99) → opening-determined;
  final(1.30) > 1.30.
  - With 3 levels × 4 seeds the CI is wide. "Inconclusive/intermediate" is a likely honest outcome, and is not
    a failure of the run.
- **Cross-model pair, arm A**: Qwen2.5-14B × Qwen3-14B off, openings (1.40, 1.90), (1.60, 2.00), (1.75, 2.15),
  6 seeds, BOTH orders (the models differ, so the firms are not exchangeable).
  - Prediction: both firms converge to the LOWER opening in both orders, as in self-play.
    Mean |final − low opening| < 0.06; averaging falsified (|final − midpoint| > 0.10).
  - Falsifier: the final price depends on WHICH model held the low opening (order effect > 0.08).

# ======================================================================================================
# PLAN v2 CONFIRMATION PRE-REGISTRATION (Part 2). DRAFT until git tag `prereg-s1`; frozen from that tag on.
# ======================================================================================================
Code frozen at `prereg-s1`: analysis/v2.py (H1-H3, Q, spreads, diagnostics), analysis/classify.py (thresholds in
results/classify/thresholds.json, fixed on DEV seeds), analysis/s1.py (drift-map fit and iteration, descriptive),
analysis/verify.py, scripts/make_v2_configs.py (configs/v2/conf written from it at the tag).
Confirmation = phase `conf`: seed master 93000, window permutation 4102. Disjoint from discovery (tested).

## Design (configs/v2/conf; counts as in plan v2)
- O: Qwen3/Qwen2.5 band {1.55, 1.75, 1.95} × 24 + edges {1.30, 1.47, 2.10} × 8; gpt-oss band × 12, edges × 6;
  OLMo-2-13B (HELD OUT, unseen in discovery) band × 12.
- T: 20 stored openings × 3 fresh seeds (Qwens). A: 3 openings × 12 (Qwens). Q: q ∈ {1.473, 1.70, 1.925} × 6 at
  c = 1.0 and c = 1.4, all four models; label and anchor (anchor 2.10) controls × 12 (Qwens).
- M: diag 17 × 30 + grid 7×7 × 15, all four models, BEFORE any O/T run. Fit with the frozen code, commit
  driftmap_predictions.json, and tag `pred-M` before O/T launch. DESCRIPTIVE ONLY: a hit is within ±0.08.
- Cold-start L baseline × 12 (all four models); E0 prompt spot check (Qwen3, x ∈ {1.55, 1.95} × 8); long horizon
  (Qwen3, x ∈ {1.30, 1.95} × 8, 300 live rounds); Fig 1 pair (Qwen3, the SAME seed at 1.55 and 1.95).
- Settings: E1 (except the E0 spot check), T = 0.7, top_p 1.0, max_tokens 2048 (gpt-oss 4096), AWQ (OLMo FP8),
  history window 20, one shared window pool (18 Qwen2.5 windows; rule in make_v2_configs.py).
- No bf16 spot check (no A100 available): stated as a limitation.

## H1 (per model, Holm over the 4 models): β_band
- Late price = mean of both firms, rounds 121-170.
- β_band = OLS slope of the late price on the NOMINAL x over the band.
- Bootstrap within level (seeds differ per level).
- **Inherited** if the Holm-adjusted one-sided p(β > 0.5) < 0.05.
- **Not inherited** if the Holm-adjusted p(β < 0.5) < 0.05.
- Otherwise **inconclusive**.
- Also reported: the slope on the realised last-5 level; the x-permutation p for β = 0.

## H2 (Qwens; the Holm family of 5 = H1 × 4 + H2 per Qwen is reported both ways): transplant ICC
- ICC(1) of the late price across the 20 transplanted openings (3 seeds each), bootstrapped over openings.
- H2 counts if ICC > 0 with its 95% lower bound > 0.
- If the openings' round-20 SD < 0.10, H2 becomes secondary.

## H3 (Qwens, arm A): lowest-opening dominance
- d = |late − midpoint| − |late − min(x1, x2)|.
- **Supported** if mean d > 0 with the one-sided sign-flip p < 0.05 (Holm over the 2 Qwens).

## O vs T disagreement (numeric)
- Disagree if the O effect (β_band) and the T effect (ICC) have opposite signs, OR their 95% CIs, both mapped to
  "share of the opening level inherited", do not overlap.
- If they disagree, T governs.

## Edge drift (separates "inherited" from "neutral band only")
- Mean drift (late − x) at 1.30 and at 2.10.
- Threshold set from Discovery-1, before the tag: an edge RESTORES if |mean drift| > 0.10 there.
  - Discovery-1 band drifts were within ±0.10 (Qwens); low-edge drifts were +0.233 (Qwen3) and +0.276 (Qwen2.5).
  - "Neutral band only" = H1 inherited AND at least one edge restores. "Inherited (whole range)" = H1 inherited
    AND neither edge restores.

## Secondary measures
- Global slope (target-seeking only if the upper bound < 0.2).
- Δ(1.95) − Δ(1.55).
- Carry-over slope over rounds 41-60.
- (ρ_r, ρ_c) against the matcher (1, 0) and the scripted BR (0.42, own-cost pass-through > 0).
- Opening spread of Δ vs the between-model spread of cold-start Δ.
- Read, clip and floor rates per level.

## gpt-oss null
If gpt-oss's floor rate at a level exceeds 20%, its slope there is floor-driven, which is NOT evidence of target
seeking.

## Stopping rule
None. Spare seeds only replace crashed runs, and each replacement is logged.

## Conditional backups (pre-registered)
- Classifier fails on held-out scripted runs. ALREADY APPLIES: 6/8 at discovery, so the type-classifier and audit
  claims are narrowed. The O/T headline is kept.
- Injection validity fails (read check < 50% at the band levels, or O and T disagree) → Backup 1, the
  two-coefficient audit: Q (ρ_r, ρ_c).
- Backup 1 without M6 does not apply: M6 is built.

## Outcome table (claim and title fixed by the outcome)
| Outcome | Claim |
|---|---|
| Qwen inherited, others not (or inconclusive) | "The opening, not the model, sets the price for imitative agents" |
| Neutral band only (β_band inherited, edges restore) | "Inheritance inside a neutral band" |
| None inherited | "First randomised support for 'collusion is a property of the model'" |
| O and T disagree | "Injected and transplanted openings disagree: T governs"; report both |
| Gates fail | report the failure and the backup audit |
| + H3 supported | add "the lowest opening dominates" |

## Novelty rule
- Luo et al. 2602.17203 seed simulated histories (4 discrete levels, no slope): cited first.
- If a newly found paper randomises LLM openings continuously, cite it and narrow our claim to the audit plus the
  held-out family.

# ======================================================================================================
# MECHANISM ADDENDUM PRE-REGISTRATION (H4 STEP, H5 IMP). Frozen at tag `prereg-mech`, BEFORE any
# STEP/IMP LLM call. Code: analysis/mech.py (rules in its docstring = the addendum's N6, thresholds unchanged).
# ======================================================================================================
- Configs: configs/mech/ (written by analysis/mech.py at the tag).
  - Qwen3 and Qwen2.5: STEP and IMP, 3 cells × 20 seeds, the SAME seeds across cells (CRN).
  - Sonnet 5.5: STEP 3 × 8, plus arm O band {1.55, 1.75, 1.95} × 4 (paid, cap $100, appendix-labelled, T = 1).
  - Opening: x = 1.70 from the shared pool. Shock at round 41; Δ = ±0.15. 150 live rounds.
- H4 and H5: exactly as the addendum's N6 and the analysis/mech.py docstring. Holm adds H4 and H5 to each Qwen's
  family.
- The scripted gate PASSED (exact pass, 15/15) before this tag.
  - Coverage gap noted, NOT fixed: H5 has no "reverse one-way" label.
- Power from Discovery-1 σ (late p1 at x = 1.75; zero within-seed correlation assumed):
  - Qwen3: σ 0.028, SE(A) ≈ 0.10, MDE80 ≈ 0.28.
  - Qwen2.5: σ 0.046, SE(A) ≈ 0.17, MDE80 ≈ 0.47.
- N5 checks are computed on the real runs FIRST (team decision: no separate smoke batch): parse/clip rates, hold
  cell vs 1.70 by round 41 (> 0.05 is reported prominently before any H4 claim), and the CRN identity share.
- Predictions: the addendum's stand. Claude's update, after the N1 CPU checks, recorded
  here before any STEP/IMP call:

  | Run | Prediction | Basis |
  |---|---|---|
  | Qwen3 STEP | ρ_down 0.60 [0.30, 1.0], ρ_up 0.30 [0.05, 0.65], A 0.30 [−0.2, 0.8] | dynamic checks show raise-following; P(ratchet) ≈ 0.35 |
  | Qwen2.5 STEP | ρ_down 0.55 [0.2, 0.95], ρ_up 0.45 [0.1, 0.8], A 0.10 [−0.4, 0.6] | P(ratchet) ≈ 0.2 |
  | Qwen3 IMP | H_down 0.45 [0.1, 0.8], H_up 0.20 [−0.1, 0.5] | |
  | Qwen2.5 IMP | H_down 0.30 [0, 0.65], H_up 0.25 [−0.05, 0.55] | |
  | Sonnet STEP | symmetric, BR-like (ρ 0.2–0.65) | FM 1.663 ≈ BR |
  | Sonnet O | β_band 0.30 [0.0, 0.7] | best-responds when it can; prices high only against a reacting rival |

## Later pre-registrations (separate files, each committed before its runs)
- Supervision arms and the cross-supervision follow-up: docs/SUPERVISION.md (; follow-up).
- Pressure arms: docs/PRESSURE.md.
- K2 extra model: docs/PREDICTIONS.md.
