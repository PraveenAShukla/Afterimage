# Afterimage card: gpt-oss-20b

- gpt-oss-20b, reasoning effort low, T 0.7; committed conf runs (tag results-final)
- levels × n: {'1.550': 12, '1.750': 12, '1.950': 12}; window ids 12

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.782 | [0.681, 0.883] | [0.675, 0.887] |
| A_late (121–170) | 0.343 | [0.230, 0.463] | [0.208, 0.483] |

**Label:** lets go (partly) (PREREG_AFTERIMAGE label table; both CIs agree)

**Depth:** D_0.5 30 rounds after the planted rows left (bin 71-80); D_0.2 > 130 rounds. Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.782, 61-80 0.561, 81-100 0.461, 101-120 0.429, 121-140 0.412, 141-170 0.297

**Cold start:** n 12, mean 1.153, SD 0.072, price index -0.71, Δ -1.29

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 82 runs, 25144 calls, $0.00

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
