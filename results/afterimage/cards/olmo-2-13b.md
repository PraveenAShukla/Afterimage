# Afterimage card: olmo-2-13b

- OLMo-2-1124-13B-Instruct, T 0.7; committed conf runs (tag results-final)
- levels × n: {'1.550': 12, '1.750': 12, '1.950': 12}; window ids 12

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | -0.128 | [-0.396, 0.157] | [-0.332, 0.063] |
| A_late (121–170) | -0.118 | [-0.214, -0.032] | [-0.212, -0.035] |

**Label:** degenerate (NOT SCORED) (a gate in G0-G5 failed)

**Depth:** D_0.5 0 rounds after the planted rows left (bin 41-50); D_0.2 0 rounds after the planted rows left (bin 41-50). Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 -0.128, 61-80 -0.237, 81-100 -0.207, 101-120 -0.167, 121-140 -0.141, 141-170 -0.103

**Cold start:** n 12, mean 1.022, SD 0.009, price index -1.00, Δ -1.85

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost FAIL; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 48 runs, 14885 calls, $0.00

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
