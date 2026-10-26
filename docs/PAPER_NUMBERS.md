# Numbers for the draft's pending boxes (frozen rules; run `analysis/verify.py` before citing)

## FROZEN VALUES: copy these exactly (tag `results-afterimage-final`; within-level 90% CIs; contrasts window-paired cluster)
- Bootstrap CI bounds carry Monte Carlo error of about ±0.003 (10,000 draws). Earlier lines below may differ in the third decimal, because they were written before the last experiments were added. Use this table.

| Quantity | Value |
|---|---|
| E5 full A_late / A_early | 0.786 [0.622, 0.905] / 0.968 [0.936, 0.995] → CONFIRMED |
| E5 price-only A_late; D_PO | 0.728 [0.344, 1.016]; −0.058 [−0.451, +0.254] → unresolved |
| E1 BR A1_late; D_BR | 0.001 [−0.002, 0.003]; −0.807 [−0.869, −0.743] → loop-carried |
| E1 COPY A1_late; D_COPY | 0.839 [0.766, 0.910]; +0.032 [−0.035, +0.091] → preserved |
| E1b Nash A1_late | 0.002 [−0.006, 0.011] → none |
| E4 Haiku 5.5 early / late | 0.193 [0.063, 0.330] / −0.006 [−0.041, 0.035] → unresolved |
| E4 Sonnet 5.5 early / late | 0.623 [0.441, 0.798] / 0.414 [0.230, 0.597] → unresolved |
| E4 GPT-6.1-sol early / late | 0.076 [−0.045, 0.208] / 0.006 [−0.200, 0.201] → unresolved (UB90 early 0.208, at the 0.2 line) |
| D2 sham A_late; D_hint; sham − base | 0.861 [0.808, 0.915]; −0.861 [−0.912, −0.810]; +0.111 [−0.007, +0.275] → WORKS |
| E2b none; high8k (D3, post-data); D | 0.760 [0.624, 0.887]; 0.200 [0.006, 0.392]; −0.560 [−0.776, −0.351] → exploratory "reasoning shortens" |

Sources: results/afterimage/{e1,e1b,e5,e4,d2}/numbers.json and cards/. Within-level 90% CIs for single stages;
window-paired cluster 90% CIs for contrasts (PREREG_AFTERIMAGE.md).

## Abstract, "[E1: relay-cut sentence]"
- A relay cut shows the market carries the plant, not the agent: when firm 2 is replaced from round 21 by a scripted best responder, Qwen3-14B's own afterimage is 0.001 (90% CI [−0.002, 0.003]); with a rival-price tracker it is 0.839 [0.765, 0.910].

## Section 3, E5 box (CONFIRMED)
- "On 48 fresh games the afterimage replicates: A_late = 0.786 (90% CI [0.631, 0.906]), A_early = 0.968 [0.938, 0.995]."
- Server note: Modal H100, same pinned model revision and vLLM 0.11.2.

## Section 4, E1 / E1b / E5 price-only box
- **E1 BR:** loop-carried.
  - A1_BR(121–170) = 0.001 [−0.002, 0.003] (already 0.001 in rounds 41–60).
  - D_BR vs self-play = −0.807 [−0.870, −0.743].
- **E1 COPY:** preserved.
  - A1_COPY(121–170) = 0.839 [0.765, 0.910]; D_COPY = +0.032 [−0.036, +0.091].
  - Holm over the two contrasts: D_BR p < 0.001, D_COPY p = 0.44.
- **Sentence (pre-agreed):** "A best-responding rival removes the afterimage and a tracking rival preserves it, so the two agents carry it between them."
- **E1b, Nash-fixed rival:** none. A1(121–170) = 0.002 [−0.006, 0.010].
- **E5 price-only:** unresolved.
  - D_PO = −0.058 [−0.453, +0.254].
  - A_PO(121–170) = 0.728 [0.345, 1.010].
  - Sentence: "Hiding shares and profits leaves the afterimage at 0.73, but the contrast with the full view is too uncertain to call (D_PO −0.06 [−0.45, +0.25])."

## Section 7, E4 box (API breadth; Anthropic T 1.0, effort low, adaptive thinking)
- **Haiku 5.5:** "In our duopoly, Claude Haiku 5.5 is unresolved: A_late = −0.006 [−0.041, 0.034]; A_early = 0.193 [0.067, 0.329] (it misses 'none' only because UB90 A_early ≥ 0.2)." Cold start at Nash (Δ 0.03).
- **Sonnet 5.5:** "In our duopoly, Claude Sonnet 5.5 is unresolved: A_late = 0.414 [0.218, 0.596]; A_early = 0.623 [0.440, 0.792]." Cold start Δ 0.67. n 12 per level; this replaces the old n = 4 row.
- **GPT-6.1-sol:** "In our duopoly, GPT-6.1-sol is unresolved: A_late = 0.006 [−0.197, 0.204]; A_early = 0.076 [−0.045, 0.210]." Cold start Δ 0.19; 48/48 runs.
- **Holm within the API family** on p(A_late ≤ 0.5): no model relays.

## Section 7, E2b box
- Not scored (G1): at reasoning_effort high, 10 of 24 luna runs truncated at max_tokens 2048 and were excluded (model-caused).
- Exploratory only: D = A_high − A_none = −0.46 [−0.70, −0.26]; A_none 0.76.
- Sentence: "For GPT-6-luna the pre-registered high-reasoning cell failed the completeness gate (output cap 2,048 tokens). In a post-data rerun with an 8,192-token cap (same seeds and windows), high reasoning shortens the afterimage: A_late 0.20 [0.01, 0.39] against 0.76 [0.62, 0.89] with reasoning off (D = −0.56 [−0.78, −0.35]; exploratory)."

## Section 7, D2 box (WORKS)
- **Sentence:** "With the hint, Qwen3's afterimage falls to 0.000, 0.861 [0.810, 0.912] below the sham, and it still best-responds to a rival at Nash (FN gap 0.00004)."
- **Always print:**
  - uptake 99.9% (decisions within ±0.01 of the hinted price);
  - prices end exactly at Nash (1.473) at both plants, not at the cold-start level (1.80);
  - the sham alone: A_late 0.861 [0.808, 0.915].
- **Caveat to state:** the fix works because Qwen3 follows a stated best response. It removes the supra-competitive price too, so it is an auditor's or regulator's lever, not a deployer's.

## Prediction scorecard (Claude's predictions recorded before data; the team did not sign)
| Prediction | Outcome | Held? |
|---|---|---|
| E5 A_late 0.84 [0.72, 0.94] | 0.786 | yes |
| E5 CONFIRMED (P ≈ 0.8) | CONFIRMED | yes |
| E5 D_PO ≈ 0 [−0.10, +0.10] | −0.058, but CI [−0.45, +0.25] | point yes; precision no |
| E1 A1_BR 0.10 [−0.05, 0.35] | 0.001 | yes |
| E1 A1_COPY 0.85 [0.70, 0.98], preserved | 0.839, preserved | yes |
| E1b ≤ 0.2 | 0.002 | yes |
| E4 Haiku "lets go" | unresolved | no |
| E4 Sonnet "lets go" | unresolved | no |
| D2 A_hint 0.30 [0.00, 0.75] | 0.000 | yes (at the edge) |
| D2 P(WORKS) ≈ 0.4 | WORKS | (probabilistic) |
| E4 sol "lets go or none" | unresolved (≈ 0) | partly (the 'none' side) |
| E2b D −0.15 [−0.45, +0.10] | D1 not scored (G1); D3 rerun (post-data) −0.56 [−0.78, −0.35] | direction yes; size larger (post-data) |
