# Afterimage card: gpt-6.1-sol

- GPT-6.1-sol, reasoning effort low, T 1.0; D1 (tag prereg-afterimage-D1)
- levels × n: {'1.550': 12, '1.750': 12, '1.950': 12}; window ids 12

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.076 | [-0.045, 0.208] | [-0.060, 0.214] |
| A_late (121–170) | 0.006 | [-0.200, 0.201] | [-0.216, 0.221] |

**Label:** unresolved (PREREG_AFTERIMAGE label table; both CIs agree)

**Depth:** D_0.5 0 rounds after the planted rows left (bin 41-50); D_0.2 0 rounds after the planted rows left (bin 41-50). Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.076, 61-80 -0.052, 81-100 0.015, 101-120 0.072, 121-140 0.042, 141-170 -0.019

**Cold start:** n 12, mean 1.534, SD 0.090, price index 0.14, Δ 0.19

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 48 runs, 14887 calls, $33.20

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
