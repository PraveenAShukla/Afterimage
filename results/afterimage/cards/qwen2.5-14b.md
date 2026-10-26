# Afterimage card: qwen2.5-14b

- Qwen2.5-14B-Instruct-AWQ, T 0.7; committed conf runs (tag results-final)
- levels × n: {'1.550': 24, '1.750': 24, '1.950': 24}; window ids 18

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.949 | [0.924, 0.972] | [0.923, 0.974] |
| A_late (121–170) | 0.557 | [0.422, 0.681] | [0.417, 0.699] |

**Label:** unresolved (PREREG_AFTERIMAGE label table; both CIs agree)

**Depth:** D_0.5 > 130 rounds; D_0.2 > 130 rounds. Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.949, 61-80 0.819, 81-100 0.707, 101-120 0.610, 121-140 0.570, 141-170 0.548

**Cold start:** n 12, mean 1.869, SD 0.203, price index 0.88, Δ 0.76

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 140 runs, 42480 calls, $0.00

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
