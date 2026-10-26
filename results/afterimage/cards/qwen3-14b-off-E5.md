# Afterimage card: qwen3-14b-off-E5

- Qwen3-14B-AWQ, thinking off, T 0.7 (E5 fresh seeds); D1 (tag prereg-afterimage-D1)
- levels × n: {'1.550': 16, '1.750': 16, '1.950': 16}; window ids 16

| | A | 90% within | 90% cluster |
|---|---|---|---|
| A_early (41–60) | 0.968 | [0.936, 0.995] | [0.939, 0.995] |
| A_late (121–170) | 0.786 | [0.622, 0.905] | [0.633, 0.893] |

**Label:** relays (PREREG_AFTERIMAGE label table; both CIs agree)

**Depth:** D_0.5 > 130 rounds; D_0.2 > 130 rounds. Half-life: blank: X0 passed under the original rule, but the amended test specification (R 1000, both CIs, sigma tuned per h) was not run.

**Curve (EXPL):** 41-60 0.968, 61-80 0.872, 81-100 0.824, 101-120 0.823, 121-140 0.813, 141-170 0.769

**Cold start:** n 12, mean 1.802, SD 0.150, price index 0.73, Δ 0.81

**Validity:** G0 instrument pass; G1 completeness pass; G2 decisions pass; G3 uptake pass; G4 near cost pass; G5 clipping pass; G6 size pass; G7 co-metric n/a

**Cost:** 60 runs, 18490 calls, $0.00

**What A is not:**
- Not a conduct test: no forced deviation or punishment read.
- Not shown to be stable across seed and window draws (Qwen2.5's A_late moved 0.741 -> 0.557 between phases).
- Defined only inside the 1.55-1.95 band.
- It fades: on the long arm the gap between plants shrinks over 150 rounds.
- Surviving round 41 is not the finding: the informative quantity is the per-round loss after the planted rows leave the 20-row prompt window.
