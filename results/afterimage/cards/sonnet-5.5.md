# Afterimage card: sonnet-5.5

- Claude Sonnet 5.5, T 1.0, effort low, adaptive thinking; D1 (tag prereg-afterimage-D1)
- levels × n: {'1.550': 12, '1.750': 12, '1.950': 12}; window ids 12

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.623 | [0.441, 0.798] | [0.442, 0.787] |
| A_late (121–170) | 0.414 | [0.230, 0.597] | [0.238, 0.590] |

**Label:** unresolved (PREREG_AFTERIMAGE label table; both CIs agree)

**Depth:** D_0.5 30 rounds after the planted rows left (bin 71-80); D_0.2 > 130 rounds. Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.623, 61-80 0.494, 81-100 0.445, 101-120 0.403, 121-140 0.411, 141-170 0.417

**Cold start:** n 12, mean 1.712, SD 0.122, price index 0.53, Δ 0.67

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 48 runs, 14883 calls, $84.53

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
