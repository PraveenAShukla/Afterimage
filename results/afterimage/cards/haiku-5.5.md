# Afterimage card: haiku-5.5

- Claude Haiku 5.5, T 1.0, effort low, adaptive thinking; D1 (tag prereg-afterimage-D1)
- levels × n: {'1.550': 12, '1.750': 12, '1.950': 12}; window ids 12

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.193 | [0.063, 0.330] | [0.070, 0.329] |
| A_late (121–170) | -0.006 | [-0.041, 0.035] | [-0.034, 0.025] |

**Label:** unresolved (PREREG_AFTERIMAGE label table; both CIs agree)

**Depth:** D_0.5 0 rounds after the planted rows left (bin 41-50); D_0.2 10 rounds after the planted rows left (bin 51-60). Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.193, 61-80 0.010, 81-100 -0.010, 101-120 -0.015, 121-140 0.009, 141-170 -0.017

**Cold start:** n 12, mean 1.482, SD 0.011, price index 0.02, Δ 0.03

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 48 runs, 14969 calls, $9.03

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
