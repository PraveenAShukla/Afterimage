"""Write every S1 config (configs/s1/*.yaml) from one place, so levels, seeds and sources are fixed before any run.

  .venv/bin/python scripts/make_s1_configs.py          # refuses to overwrite an existing file

Models (open, E1, T=0.7, AWQ, $0): Qwen3-14B thinking off, Qwen2.5-14B, gpt-oss-20b. Controls: averager, static BR,
grim trigger at 1.80, hill-climber (mock policies on Qwen3's real windows, opening_source.cross_model: true).
Source windows: each model's own stored L runs (rounds 181-200), chosen by a recorded permutation (perm_seed).
Seeds: 6 per level, the SAME 6 at every level (with the same permutation index, each seed sees the same source
window at every level, so level effects are not confounded by window texture).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "s1"

MODELS = {
    "qwen3off": {"firm": {"provider": "vllm", "model": "Qwen/Qwen3-14B-AWQ",
                          "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
                 "max_tokens": 2048, "template": "s3-qwen3_14b_off_L_E1_Qwen3-14B-AWQ_s{seed}",
                 "pool": json.loads((ROOT / "configs/seeds/qwen3_14b_off.json").read_text())["L"], "seed_base": 81000},
    "qwen25": {"firm": {"provider": "vllm", "model": "Qwen/Qwen2.5-14B-Instruct-AWQ"},
               "max_tokens": 2048, "template": "s3-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s{seed}",
               "pool": json.loads((ROOT / "configs/seeds/qwen25_14b.json").read_text())["L"], "seed_base": 82000},
    "gptoss": {"firm": {"provider": "vllm", "model": "openai/gpt-oss-20b", "extra_body": {"reasoning_effort": "low"}},
               "max_tokens": 4096, "template": "s1-gptoss20b_L_E1_gpt-oss-20b_s{seed}",
               "pool": list(range(20002, 20012)), "seed_base": 83000},
}
CONTROLS = ["averager", "br", "grim:1.80", "hillclimb:0.02"]
LEVELS_S = [1.30, 1.47, 1.70, 1.925, 2.10]
OPEN_A = [(1.40, 1.90), (1.60, 2.00), (1.75, 2.15)]
DIAG = [round(1.20 + 0.075 * i, 3) for i in range(17)]          # 1.200 ... 2.400
GRID = [1.30, 1.433, 1.567, 1.70, 1.833, 1.967, 2.10]           # 7 x 7
N_SEEDS = 6
PERM_SEED = 230101


def sampling(mt: int) -> dict:
    return {"temperature": 0.7, "top_p": 1.0, "max_tokens": mt}


def write(name: str, cfg: dict) -> None:
    p = OUT / f"{name}.yaml"
    if p.exists():
        sys.exit(f"refusing to overwrite {p} (configs are fixed once written)")
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))


def src(m: dict, cross: bool = False) -> dict:
    s = {"template": m["template"], "pool": list(m["pool"]), "perm_seed": PERM_SEED}
    if cross:
        s["cross_model"] = True
    return s


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, m in MODELS.items():
        seeds = [m["seed_base"] + i for i in range(1, N_SEEDS + 1)]
        for x in LEVELS_S:
            write(f"S_{key}_x{x:.3f}", {"stage": f"s1S-{key}", "arm": "S", "prompt": "E1", "rounds": 300, "window": 100,
                                        "opening": [x, x], "opening_source": src(m), "seeds": seeds,
                                        "sampling": sampling(m["max_tokens"]), "parallel_games": 6,
                                        "firm1": m["firm"], "firm2": m["firm"]})
        if key != "gptoss":
            for x1, x2 in OPEN_A:
                write(f"A_{key}_{x1:.2f}-{x2:.2f}", {"stage": f"s1A-{key}", "arm": "A", "prompt": "E1", "rounds": 300,
                                                     "window": 100, "opening": [x1, x2], "opening_source": src(m),
                                                     "seeds": [s + 50 for s in seeds], "sampling": sampling(m["max_tokens"]),
                                                     "parallel_games": 6, "firm1": m["firm"], "firm2": m["firm"]})
        base = {"prompt": "E1", "sampling": sampling(m["max_tokens"]), "firm1": m["firm"], "opening_source": src(m)}
        write(f"map_diag_{key}", {"stage": f"s1map-{key}", "mode": "diag", "levels": DIAG, "samples": 30,
                                  "seed_base": m["seed_base"] * 10 + 1, **base})
        write(f"map_grid_{key}", {"stage": f"s1map-{key}", "mode": "grid", "levels": GRID, "samples": 15,
                                  "seed_base": m["seed_base"] * 10 + 5, **base})
    q3 = MODELS["qwen3off"]
    for ctl in CONTROLS:
        tag = ctl.split(":")[0]
        f = {"provider": "mock", "model": f"control-{tag}", "strategy": ctl}
        for x in LEVELS_S:
            write(f"S_ctl-{tag}_x{x:.3f}", {"stage": f"s1S-ctl-{tag}", "arm": "S", "prompt": "E1", "rounds": 300,
                                            "window": 100, "opening": [x, x], "opening_source": src(q3, cross=True),
                                            "seeds": [84000 + i for i in range(1, N_SEEDS + 1)],
                                            "sampling": sampling(2048), "parallel_games": 6, "firm1": f, "firm2": f})
        for x1, x2 in OPEN_A:
            write(f"A_ctl-{tag}_{x1:.2f}-{x2:.2f}", {"stage": f"s1A-ctl-{tag}", "arm": "A", "prompt": "E1", "rounds": 300,
                                                     "window": 100, "opening": [x1, x2], "opening_source": src(q3, cross=True),
                                                     "seeds": [84050 + i for i in range(1, N_SEEDS + 1)],
                                                     "sampling": sampling(2048), "parallel_games": 6, "firm1": f, "firm2": f})
        write(f"map_diag_ctl-{tag}", {"stage": f"s1map-ctl-{tag}", "mode": "diag", "levels": DIAG, "samples": 5,
                                      "seed_base": 840001, "prompt": "E1", "sampling": sampling(2048), "firm1": f,
                                      "opening_source": src(q3, cross=True)})
    print(f"wrote {len(list(OUT.glob('*.yaml')))} configs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
