# DISCOVERY_S1: exploratory discovery batch (plan v2 E; phase = disc; PRE-SPRINT, disclosed)

All numbers come from results/v2/disc/numbers.json (analysis/v2.py). Bullets only. Discovery numbers are EXPLORATORY:
they set the confirmation's thresholds, and they never stand in for the confirmation.

## Design
- 20-row injected opening (rounds 1–20), then 150 live rounds (21–170). Late price = mean of both firms, rounds
  121–170. E1, T = 0.7, AWQ.
- One shared window pool: 18 Qwen2.5 L-run windows (rounds 181–200), |slope| ≤ 0.002 per round, firm gap ≤ 0.10.
- Arm O:
  - Qwen3 and Qwen2.5: band {1.55, 1.75, 1.95} × 8, edges {1.30, 2.10} × 4.
  - gpt-oss: band × 4, edges × 2.
- Arm Q: q ∈ {1.473, 1.70, 1.925} × 4, both Qwens.
- Arm A: 3 asymmetric openings × 4, both Qwens.
- One-step map M (v2 framing: rounds 1–20 shown, decide at 21).

## β_band (H1 statistic), with predictions recorded before the batch (docs/PREDICTIONS.md)

| Model | β_band | 90% CI | Holm p(β > 0.5) | Label | Claude's prediction [80%] |
|---|---|---|---|---|---|
| Qwen3-14B off | 0.765 | [0.626, 0.891] | 0.003 | inherited | 0.85 [0.60, 1.05], inside |
| Qwen2.5-14B | 0.741 | [0.629, 0.847] | 0.0006 | inherited | 0.85 [0.60, 1.05], inside |
| gpt-oss-20b | 0.332 | [0.168, 0.497] | 0.95 | inconclusive (p(β < 0.5) = 0.14) | 0.40 [0.10, 0.70], inside |

- Slope on the realised last-5 level: 0.767 / 0.743 / 0.331, the same as nominal (the windows are flat).
- Global slope (all levels): 0.641 / 0.651 / 0.384.

## Edge drift (late − x)

| Model | at 1.30 | at 2.10 | Reading |
|---|---|---|---|
| Qwen3 | +0.233 | −0.094 | restoring force below the band; mild above |
| Qwen2.5 | +0.276 | −0.029 | restoring force below; none above |
| gpt-oss | −0.135 | −0.603 | downward drift everywhere (price war) |

- Δ by level, Qwen3: 0.19 / 0.40 / 0.81 / 0.97 / 0.93; Qwen2.5: 0.33 / 0.56 / 0.92 / 0.99 / 0.86.
- **Δ(1.30) is POSITIVE** in the v2 design, unlike Discovery-0 (−0.12 / −0.19 with 100 live rounds after a
  200-row prefix). So the v1 secondary criterion "below Nash from a low opening" does not carry over to the v2
  design.
- The edge threshold (|drift| > 0.10 restores) was fixed from these numbers in PREREG_S1 before the tag.

## Diagnostics
- Read check: 100% of round-21 outputs mention a price within ±0.03 of x, at every level, for every model.
- Clip rate 0, floor rate 0 at every level (no gpt-oss floor collapse at these levels).

## Fixed-rival ladder ρ_r (Q)
- Qwen3 0.928; Qwen2.5 0.949. Scripted static BR 0.418, matcher 1.0.
  - Late p1: Qwen3 1.471 / 1.654 / 1.891 and Qwen2.5 1.478 / 1.643 / 1.907, at q = 1.473 / 1.70 / 1.925.
- Claude's predictions (0.69, 0.61) were too LOW: both models nearly match a fixed rival over 100 rounds.
  - Earlier 300-round FM runs gave 1.785 / 1.750 at q = 1.925: the longer horizon pulls further toward BR.

## Opening spread vs model spread (the Lee comparison)
- Within-model Δ(2.10) − Δ(1.30): Qwen3 0.734; Qwen2.5 0.524; gpt-oss 1.316.
- Between-model cold-start Δ range across our 9 models: 2.61 (OLMo −1.87 … Qwen3 0.74; includes the price-war
  models).
- Lee's Claude − DeepSeek: 0.55.
- Reading [EXPL]: the opening alone moves a Qwen's Δ about as much as Lee's difference between two model families.
  It moves Δ less than the full range across our 9 models.

## Scripted ground truth (analysis/classify.py)
- Dev 8/8 (fitted). Held-out (noise sd 0.02, dev thresholds) 6/8: grim and the hill-climber are missed.
- β_band's inherited / not call is correct 8/8.

## Arm A: lowest-opening dominance (H3 statistic, exploratory)
- d = |late − midpoint| − |late − min(x1, x2)|; > 0 means the late price is nearer the LOW opening.
- Qwen3: d = +0.153, sign-flip p = 0.0016. |late − low| = 0.042 vs |late − midpoint| = 0.195.
- Qwen2.5: d = +0.077, p = 0.033. |late − low| = 0.070 vs |late − midpoint| = 0.147.
- Late price by opening:

  | Opening | Qwen3 | Qwen2.5 |
  |---|---|---|
  | (1.40, 1.90) | 1.463 | 1.485 |
  | (1.60, 2.00) | 1.618 | 1.651 |
  | (1.75, 2.15) | 1.734 | 1.823 |

- This replicates Discovery-0 (design v1) and the cross-model pair. Averaging is rejected in all three.
- n = 12 runs per model; all 128 Discovery-1 runs are valid, 0 excluded.

## Mechanism CPU checks N1 (read; predictions committed before computing; results/s1/mech_cpu.json)
| Check | Qwen3 | Qwen2.5 | gpt-oss | Claude's prediction |
|---|---|---|---|---|
| C-a FM (rival fixed at 1.925) | early 1.907 → late 1.785; corr 0.52; 0/10 up → neither | 1.979 → 1.750; corr 0.28; 1/10 up → neither | — | "upward pull": WRONG |
| C-b recovery after a forced cut: firm 1 raises first, firm 2 follows within 2 | 5 of 7 → follows raises | 14 of 16 → follows raises | — | both follow: right |
| C-c P-arm split β+ / β− | 0.171 / 0.222; LB(β− − β+) = −0.04 → no ratchet | 0.321 / 0.121; LB −0.46 → no ratchet | — | not a ratchet: right |
| C-d one-step grid: rival weight, rival above / below | 0.007 / 0.133 | 0.008 / 0.107 | 0.053 / 0.571 | "weak < 0.10": WRONG (below ≈ 0.11–0.13) |
| C-f K3 seed-cluster β (Discovery-0) | 0.685 [0.636, 0.732], opening-determined | 0.796 [0.723, 0.870], opening-determined | 0.433 [0.339, 0.535], intermediate | labels unchanged: right |
| C-g repeat rate at x = 1.70 | 0.37 | 0.16 | 0.11 | 0.3–0.7: Qwen3 right, others below |
| C-i A_self (mock check PASSED: 0.02 / 1.02 / −0.98) | −0.11 [−0.18, 0.44] | −0.23 [−0.36, 0.14] | +0.50 [0.42, 0.57] | Qwen3 < 0.5: right |
| C-e text codes | NOT RUN: dictionary not committed | | | |

- Reading [EXPL]:
  - Against a STANDING gap (one-step probe), the Qwens respond to a rival below them and ignore one above.
  - In live dynamics they follow both directions (C-b, C-c, C-i ≈ 0).
  - A simple ratchet does not explain arm A's convergence to the low opening. STEP/IMP (placebo-controlled,
    pre-registered) is the decisive test.
  - gpt-oss shows a ratchet-like self-play drift (A_self 0.50), consistent with its downward drift.
- The long-horizon arm is allowed by its gates: C-g repeat rate < 0.9, and C-i is not void.
