# Afterimage card: sonnet-5.5-old

- Claude Sonnet 5.5, T 1.0, effort low, adaptive thinking (n 4 per level); committed conf runs (tag results-final)
- levels × n: {'1.550': 4, '1.750': 4, '1.950': 4}; window ids 4

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.639 | [0.330, 0.979] | [0.313, 0.988] |
| A_late (121–170) | 0.366 | [0.092, 0.623] | [0.186, 0.580] |

**Label:** degenerate (NOT SCORED) (a gate in G0-G5 failed)

**Depth:** D_0.5 50 rounds after the planted rows left (bin 91-100); D_0.2 > 130 rounds. Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.639, 61-80 0.567, 81-100 0.507, 101-120 0.385, 121-140 0.378, 141-170 0.357

**Cold start:** none run

**Validity:** G0 instrument pass; G1 completeness FAIL; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size FAIL; G7 co-metric n/a

**Cost:** 12 runs, 3600 calls, $21.19

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
