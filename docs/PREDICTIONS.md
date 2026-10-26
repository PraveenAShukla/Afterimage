# Pre-registered predictions

Team predictions were not recorded before the runs (disclosed deviation). Only the predictions below, recorded before any Stage 1 run, exist.

## Claude's predictions (recorded BEFORE any Stage 1 run; not the team's)
Anchors: pilot price paths (10 rounds), symmetric-price -> Δ map (1.47→0.0, 1.55→0.27, 1.60→0.43, 1.65→0.58, 1.70→0.71).
| Model | (a) mean Δ_L [80% PI] | (b) Δ_P/Δ_L [80% PI] | (c) depth sign r201 / r202 |
|---|---|---|---|
| Claude Haiku 5.5 | 0.35 [0.05, 0.65] | 0.6 [0.2, 1.0] | + / + (small) |
| gpt-6-luna (effort none) | 0.35 [0.10, 0.60] | 0.6 [0.2, 1.0] | + / + (small) |
| Claude Sonnet 5.5 | 0.20 [-0.05, 0.55] | 0.5 [0.0, 1.0] | + / + |
| gpt-6.1-sol (effort low) | 0.02 [-0.05, 0.20] | n/a (gate likely fails) | + / ≈0 |
| R1-Distill-32B-AWQ (4096 tok) | 0.30 [0.00, 0.70] | 0.6 [0.1, 1.1] | + / + |
Key falsifiable claims:
- Round 202 depth ≥ 0 (match / best-respond, NOT punish below BR).
- 0 < Δ_P/Δ_L < 1 (part prior, part interaction).
Premortem: the pilot is 10 rounds vs 300 (long-run drift unknown); T=1 noise may break coordination (pushes Δ down).

## PRE-REGISTRATION: Sonnet 5.5 price-based interaction, confirmation (recorded, BEFORE any of these runs)
- Hypothesis: mean firm-1 price over the last 50 rounds is higher in L (live copy) than in P (open-loop replay of
  firm 2 from a different L run of the same batch).
- Statistic: I_price = mean p1(L) − mean p1(P).
- Test: two-sided permutation (10,000 label permutations), α = 0.05; bootstrap 95% CI (10,000).
- Prediction: I_price > 0, point estimate ≈ +0.10 (from the exploratory batch: +0.106, p = 0.014).
- Design: 8 L + 8 P, fresh pre-registered seed permutation (configs/seeds/sonnet55_confirm.json, master 55777).
  Settings identical to the exploratory batch: E1, T=1, effort low, adaptive thinking, 300 rounds.
- The spec's profit-based I is reported too (secondary here, for P9 reasons).
- Stopping: none; all 16 runs are reported whatever the result.

## Claude's predictions: Qwen3.5-35B-A3B self-play and G2 cross-model pair (recorded, BEFORE any Qwen3.5 run)
G2 was changed by the team: firm 1 = Qwen3.5-35B-A3B-GPTQ-Int4 (replaces Qwen2.5-14B),
firm 2 = Qwen3-14B-AWQ thinking off. Qwen3.5 first gets its own gated battery (needed as the self-play reference).
Settings: E1, T=0.7, top_p 1.0, max_tokens 2048, thinking off, 300 rounds.
| Run | Quantity | Point [80% PI] | Reasoning |
|---|---|---|---|
| Qwen3.5 self-play, Stage 1 | mean Δ_L | 0.55 [0.10, 0.85] | Both Qwen generations tested collude (0.63, 0.74) |
| Qwen3.5 self-play | follower share (A1, P arm) | > 0.5 | same family as the follower-like Qwen3 |
| Qwen3.5 self-play | price I (L − P) | ≈ 0 [−0.05, +0.06] | a follower retraces the replayed path |
| Qwen3.5 self-play | r202 depth | ≥ 0 (no punishment below BR) | no model so far punishes |
| G2 Qwen3.5 × Qwen3-off (20 games) | Δ avg | 0.50 [0.10, 0.80] | two followers match each other; coordination does NOT need the same model |
| G2 | |Δ1 − Δ2| | < 0.10 | followers end at a common price |
Falsifier for "followers coordinate across models": G2 Δ avg < 0.2 while both self-play Δ ≥ 0.2.

## Discovery-1 (plan v2 design), CLAUDE's predictions (AI-proposed, not the team's), recorded BEFORE any Discovery-1 run
Basis: Discovery-0 (design v1) per-level finals and FM prices. Design: 20-row opening (rounds 1-20)
+ 150 live rounds; outcome = mean of both firms over rounds 121-170.

| Quantity | Qwen3-14B off | Qwen2.5-14B | gpt-oss-20b |
|---|---|---|---|
| β_band (x ∈ {1.55, 1.75, 1.95}) | 0.85 [0.60, 1.05] | 0.85 [0.60, 1.05] | 0.40 [0.10, 0.70] |
| drift at 1.30 (final − 1.30) | +0.12 [+0.04, +0.25] | +0.12 [+0.03, +0.25] | −0.08 [−0.25, +0.05] |
| drift at 2.10 (final − 2.10) | −0.15 [−0.30, −0.03] | −0.06 [−0.20, +0.03] | −0.50 [−0.75, −0.25] |
| ρ_r (Q-ladder slope of p1 on q) | 0.69 [0.50, 0.85] | 0.61 [0.45, 0.80] | — |
| arm A: final vs the low opening | low + 0.05 [0.00, 0.12]; nearer the low than the midpoint in all 3 openings | same | — |

Reference slopes: matcher ρ_r = 1; scripted static BR ρ_r ≈ 0.42 (BR(1.473) = 1.473, BR(1.925) = 1.662).
Read check: ≥ 50% of round-21 outputs mention a price within ±0.03 of x.

## Discovery-1, team predictions
Team predictions were not recorded before the runs (disclosed deviation).

## Mechanism CPU checks N1, CLAUDE's predictions (AI-proposed), recorded BEFORE computing any of them
The addendum's bracketed predictions are kept. Claude's own update follows, informed by Discovery-1 (ρ_r ≈ 0.93–0.95
against a FIXED rival, which argues AGAINST "ignores raises"):
| Check | Addendum [Claude, earlier] | Claude now |
|---|---|---|
| C-a FM trajectories | self-anchoring Qwen3; unclear Qwen2.5 | upward pull in BOTH (≥ 5/10 runs rise toward 1.925), given Q-ladder near-matching |
| C-b recovery leader | Qwen2.5 follows raises; Qwen3 does not | both follow raises in ≥ half the recoveries |
| C-c P-arm split β− / β+ | ratchet: β− ≥ 0.5, β+ ≤ 0.2 | NOT a ratchet: β+ > 0.2 for both; LB(β− − β+) ≤ 0 |
| C-d grid split | weak both ways, rival weight < 0.05 | weak both ways, < 0.10 |
| C-f seed-cluster β (Discovery-0) | labels unchanged | labels unchanged; |Δβ| < 0.02 |
| C-g repeat rate at x = 1.70 | (gate for long horizon) | 0.3–0.7 (prices sticky but not frozen) |
| C-i A_self | Qwen3 0.3–0.9 | Qwen3 A_self < 0.5, or inconclusive (|G| > 0.005 in < 5% of rounds) |

## IFM K2 O band, CLAUDE's prediction (AI-proposed), recorded BEFORE any full K2 run
Disclosed EXTRA model, added after results-final; not part of the pre-registered H1 family. Analysed with the frozen v2 rule.
Seen before predicting: a 2-game smoke at x = 1.55, 10 live rounds (Δ 0.22, 0.25: K2 holds a 1.55 opening early on),
and one long default-effort reply that computed the best response (~1.51) but chose to hold 1.55.
| Quantity | Point | 80% interval | Reason |
|---|---|---|---|
| β_band (late price on x, 1.55–1.95) | 0.55 | [0.25, 0.85] | Holds the opening short-run like Qwen3; reasons about BR like gpt-oss (0.433). |
| Label under the frozen rule | intermediate/threshold | | Inherited needs p(β > 0.5) < 0.05; unlikely with 12 seeds per level if β ≈ 0.55. |
| Mean late price at x = 1.30 | 1.45 | [1.35, 1.55] | Below-Nash openings get pulled up towards Nash 1.473. |
- **Outcome:** K2 β_band +0.015 [−0.006, +0.036], not inherited; late price ≈ Nash at every x. Claude's prediction wrong (outside its 80% interval).

## Supervision v2: team predictions
Team predictions were not recorded before the runs (disclosed deviation). (D1 = real review − yoked review; D2 = yoked review − claim only.)

## Supervision v2: CLAUDE's predictions (AI-proposed; not the team's)
- D1 ≈ 0 for both models, HL in [−0.04, +0.02]. Agents react to the words they see, not to whether the words
  track their own behaviour.
- D2 slightly negative for Qwen2.5, about −0.02 [−0.07, +0.01]; about 0 for Qwen3.
- Claude's own note: the cross-supervision read showed Qwen3 firms rising under ANY advice notes at
  x = 1.55. That is consistent with D1 ≈ 0. These are review LABELS, not advice, so a D2 near 0 is also plausible.
