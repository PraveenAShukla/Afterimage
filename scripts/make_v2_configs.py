"""Plan v2 configs for one phase: configs/v2/<phase>/*.yaml. Fixed once written (refuses to overwrite).

  .venv/bin/python scripts/make_v2_configs.py disc     # Discovery-1
  .venv/bin/python scripts/make_v2_configs.py conf     # Confirmation (fresh seeds, fresh window permutation)

One shared window pool for all models: rounds 181-200 of the stored Qwen2.5-14B L runs, keeping windows with
|within-window slope| <= 0.002 per round and max firm gap <= 0.10 (rule fixed before Discovery-1;
18 of 20 qualify). Arm O: 20-row opening (labelled rounds 1-20) + 150 live rounds (21-170).
Discovery and confirmation use different seed masters AND different window permutations (plan v2 M1b).
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

POOL_TEMPLATE = "s3-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s{seed}"
EXCLUDED_WINDOWS = {61860: "slope -0.0030 > 0.002", 210629: "firm gap 0.27 > 0.10"}
POOL = [s for s in json.loads((ROOT / "configs/seeds/qwen25_14b.json").read_text())["L"] if s not in EXCLUDED_WINDOWS]
TRANSPLANT = {"qwen3off": ("s3-qwen3_14b_off_L_E1_Qwen3-14B-AWQ_s{seed}", "qwen3_14b_off"),
              "qwen25": ("s3-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s{seed}", "qwen25_14b")}

FIRMS = {
    "qwen3off": ({"provider": "vllm", "model": "Qwen/Qwen3-14B-AWQ",
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}, 2048),
    "qwen25": ({"provider": "vllm", "model": "Qwen/Qwen2.5-14B-Instruct-AWQ"}, 2048),
    "gptoss": ({"provider": "vllm", "model": "openai/gpt-oss-20b", "extra_body": {"reasoning_effort": "low"}}, 4096),
    "olmo2": ({"provider": "vllm", "model": "allenai/OLMo-2-1124-13B-Instruct"}, 2048),
}
PHASE = {
    "disc": {"master": 91000, "perm": 4101,
             "O": {"qwen3off": ({1.55: 8, 1.75: 8, 1.95: 8, 1.30: 4, 2.10: 4}),
                   "qwen25": ({1.55: 8, 1.75: 8, 1.95: 8, 1.30: 4, 2.10: 4}),
                   "gptoss": ({1.55: 4, 1.75: 4, 1.95: 4, 1.30: 2, 2.10: 2})},
             "Q": {"qwen3off": 4, "qwen25": 4}, "A": {"qwen3off": 4, "qwen25": 4}, "T": {}},
    "conf": {"master": 93000, "perm": 4102,
             "O": {"qwen3off": ({1.55: 24, 1.75: 24, 1.95: 24, 1.30: 8, 1.47: 8, 2.10: 8}),
                   "qwen25": ({1.55: 24, 1.75: 24, 1.95: 24, 1.30: 8, 1.47: 8, 2.10: 8}),
                   "gptoss": ({1.55: 12, 1.75: 12, 1.95: 12, 1.30: 6, 1.47: 6, 2.10: 6}),
                   "olmo2": ({1.55: 12, 1.75: 12, 1.95: 12})},
             "Q": {"qwen3off": 6, "qwen25": 6, "gptoss": 6, "olmo2": 6},
             "A": {"qwen3off": 12, "qwen25": 12}, "T": {"qwen3off": 3, "qwen25": 3}},
}
Q_LADDER = [1.473, 1.70, 1.925]
A_OPEN = [(1.40, 1.90), (1.60, 2.00), (1.75, 2.15)]
SAMP = lambda mt: {"temperature": 0.7, "top_p": 1.0, "max_tokens": mt}  # noqa: E731


def seeds_for(phase: str, key: str, n: int) -> list[int]:
    """Deterministic, phase-specific seeds: random.Random(f'{master}|{key}').sample, disjoint across phases (tested)."""
    rng = random.Random(f"{PHASE[phase]['master']}|{key}")
    return sorted(rng.sample(range(PHASE[phase]["master"] * 10, PHASE[phase]["master"] * 10 + 9999), n))


def src(perm: int, cross: bool = True) -> dict:
    return {"template": POOL_TEMPLATE, "pool": POOL, "perm_seed": perm, "cross_model": cross}


def configs(phase: str) -> dict[str, dict]:
    P, out = PHASE[phase], {}
    for m, levels in P["O"].items():
        firm, mt = FIRMS[m]
        for x, n in levels.items():
            out[f"O_{m}_x{x:.3f}"] = {"stage": f"v2O-{m}", "arm": "O", "phase": phase, "prompt": "E1", "rounds": 170,
                                      "window": 150, "opening": [x, x], "opening_source": src(P["perm"]),
                                      "seeds": seeds_for(phase, f"O|{m}|{x}", n), "sampling": SAMP(mt),
                                      "parallel_games": min(n, 12), "firm1": firm, "firm2": firm}
    for m, n in P["Q"].items():
        firm, mt = FIRMS[m]
        for q in Q_LADDER:
            out[f"Q_{m}_q{q:.3f}"] = {"stage": f"v2Q-{m}", "arm": "Q", "phase": phase, "prompt": "E1", "rounds": 100,
                                      "fixed_price": q, "seeds": seeds_for(phase, f"Q|{m}|{q}", n), "sampling": SAMP(mt),
                                      "parallel_games": n, "firm1": firm}
    for m, n in P["A"].items():
        firm, mt = FIRMS[m]
        for x1, x2 in A_OPEN:
            out[f"A_{m}_{x1:.2f}-{x2:.2f}"] = {"stage": f"v2A-{m}", "arm": "A", "phase": phase, "prompt": "E1",
                                               "rounds": 170, "window": 150, "opening": [x1, x2],
                                               "opening_source": src(P["perm"]),
                                               "seeds": seeds_for(phase, f"A|{m}|{x1}", n), "sampling": SAMP(mt),
                                               "parallel_games": n, "firm1": firm, "firm2": firm}
    for m, k in P["T"].items():
        firm, mt = FIRMS[m]
        tmpl, seedfile = TRANSPLANT[m]
        pool = json.loads((ROOT / f"configs/seeds/{seedfile}.json").read_text())["L"]
        # 20 stored openings x k fresh seeds: k configs, each covering all 20 openings once (job i -> opening order[i])
        for j in range(k):
            out[f"T_{m}_rep{j + 1}"] = {"stage": f"v2T-{m}", "arm": "T", "phase": phase, "prompt": "E1", "rounds": 170,
                                        "window": 150, "opening_source": {"template": tmpl, "pool": pool, "perm_seed": P["perm"]},
                                        "seeds": seeds_for(phase, f"T|{m}|{j}", len(pool)), "sampling": SAMP(mt),
                                        "parallel_games": 10, "firm1": firm, "firm2": firm}
    if phase == "conf":
        out.update(conf_extras())
    return out


def conf_extras() -> dict[str, dict]:
    """Part 2 arms beyond O/Q/A/T (plan v2): own-cost audit, label/anchor controls, cold-start baseline, E0
    prompt, long horizon, and the Fig-1 same-seed pair. All phase=conf, seeds disjoint from discovery."""
    P, out = PHASE["conf"], {}
    for m in ("qwen3off", "qwen25", "gptoss", "olmo2"):                       # Q own-cost audit: c in {1.0, 1.4}
        firm, mt = FIRMS[m]
        for q in Q_LADDER:
            out[f"Q_{m}_c1.4_q{q:.3f}"] = {"stage": f"v2Qc-{m}", "arm": "Q", "phase": "conf", "prompt": "E1",
                                           "rounds": 100, "fixed_price": q, "market": {"c": 1.4},
                                           "seeds": seeds_for("conf", f"Qc|{m}|{q}", 6), "sampling": SAMP(mt),
                                           "parallel_games": 6, "firm1": firm}
    for m in ("qwen3off", "qwen25"):                                          # label / anchor controls
        firm, mt = FIRMS[m]
        for q in Q_LADDER:
            for tag, view in (("label", {"label": True}), ("anchor", {"anchor": 2.10})):
                out[f"Q_{m}_{tag}_q{q:.3f}"] = {"stage": f"v2Q{tag}-{m}", "arm": "Q", "phase": "conf", "prompt": "E1",
                                                "rounds": 100, "fixed_price": q, "view": view,
                                                "seeds": seeds_for("conf", f"Q{tag}|{m}|{q}", 12), "sampling": SAMP(mt),
                                                "parallel_games": 12, "firm1": firm}
    for m in ("qwen3off", "qwen25", "gptoss", "olmo2"):                       # cold-start baseline (no opening)
        firm, mt = FIRMS[m]
        out[f"L_{m}_coldstart"] = {"stage": f"v2L-{m}", "arm": "L", "phase": "conf", "prompt": "E1", "rounds": 170,
                                   "seeds": seeds_for("conf", f"L|{m}", 12), "sampling": SAMP(mt),
                                   "parallel_games": 12, "firm1": firm, "firm2": firm}
    firm, mt = FIRMS["qwen3off"]
    for x in (1.55, 1.95):                                                    # E0 prompt spot check
        out[f"O_qwen3off_E0_x{x:.3f}"] = {"stage": "v2OE0-qwen3off", "arm": "O", "phase": "conf", "prompt": "E0",
                                          "rounds": 170, "window": 150, "opening": [x, x], "opening_source": src(P["perm"]),
                                          "seeds": seeds_for("conf", f"OE0|{x}", 8), "sampling": SAMP(mt),
                                          "parallel_games": 8, "firm1": firm, "firm2": firm}
    for x in (1.30, 1.95):                                                    # long horizon: 300 live rounds
        out[f"O_qwen3off_long_x{x:.3f}"] = {"stage": "v2Olong-qwen3off", "arm": "O", "phase": "conf", "prompt": "E1",
                                            "rounds": 320, "window": 300, "opening": [x, x], "opening_source": src(P["perm"]),
                                            "seeds": seeds_for("conf", f"Olong|{x}", 8), "sampling": SAMP(mt),
                                            "parallel_games": 8, "firm1": firm, "firm2": firm}
    fig1 = seeds_for("conf", "fig1", 1)                                       # Fig 1: the SAME seed at 1.55 and 1.95
    for x in (1.55, 1.95):
        out[f"O_qwen3off_fig1_x{x:.3f}"] = {"stage": "v2Ofig1-qwen3off", "arm": "O", "phase": "conf", "prompt": "E1",
                                            "rounds": 170, "window": 150, "opening": [x, x], "opening_source": src(P["perm"]),
                                            "seeds": fig1, "sampling": SAMP(mt), "parallel_games": 1,
                                            "firm1": firm, "firm2": firm}
    return out


def maps(phase: str, models: list[str]) -> dict[str, dict]:
    out = {}
    for m in models:
        firm, mt = FIRMS[m]
        base = {"prompt": "E1", "sampling": SAMP(mt), "firm1": firm, "opening_source": src(PHASE[phase]["perm"]),
                "prefix_rows": 20, "phase": phase}
        out[f"map_diag_{m}"] = {"stage": f"v2M-{m}", "mode": "diag", "levels": [round(1.20 + 0.075 * i, 3) for i in range(17)],
                                "samples": 30, "seed_base": PHASE[phase]["master"] * 100 + 1, **base}
        out[f"map_grid_{m}"] = {"stage": f"v2M-{m}", "mode": "grid", "levels": [1.30, 1.433, 1.567, 1.70, 1.833, 1.967, 2.10],
                                "samples": 15, "seed_base": PHASE[phase]["master"] * 100 + 50001, **base}
    return out


def main() -> int:
    phase = sys.argv[1]
    d = ROOT / "configs" / "v2" / phase
    d.mkdir(parents=True, exist_ok=True)
    allc = {**configs(phase), **maps(phase, list(PHASE[phase]["O"]))}  # conf maps include the held-out OLMo
    for name, c in allc.items():
        p = d / f"{name}.yaml"
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
        p.write_text(yaml.safe_dump(c, sort_keys=False))
    print(f"wrote {len(allc)} configs to {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
