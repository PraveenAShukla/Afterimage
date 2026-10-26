"""Write configs/lee/*.yaml (fixed once written). R0 = Lee's cold-start homogeneous cells (phase rep, 10 seeds).
R1 = randomised openings x in {1.50, 1.80, 2.10} (phase conf, 10 fresh seeds per level), with source windows = rounds
31-50 of the SAME model's R0 runs. Per-model API settings (`extra`) follow Lee's Table 2 (lowest reasoning, no
temperature); scripts/lee_probe.py must confirm each provider accepts them before any real run.

  .venv/bin/python scripts/make_lee_configs.py
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "configs" / "lee"
MODELS = {
    "claude": {"provider": "anthropic", "model": "claude-haiku-4-5", "extra": {"max_tokens": 600}},  # thinking off = default
    "gemini": {"provider": "google", "model": "gemini-3.1-flash-lite", "extra": {"reasoning_effort": "minimal", "max_tokens": 600}},
    "gpt": {"provider": "openai", "model": "gpt-5-nano", "extra": {"reasoning_effort": "minimal", "max_completion_tokens": 600}},
    # thinking off: {"thinking": {"type": "disabled"}} (DeepSeek thinking-mode guide, verified)
    "deepseek": {"provider": "deepseek", "model": "deepseek-flash", "extra": {"thinking": {"type": "disabled"}, "max_tokens": 600}},
    # our addition (not in Lee): cheapest current Qwen tier; thinking is on by default, so enable_thinking: false
    # via OpenRouter (no Alibaba key); same snapshot as Model Studio qwen3.5-flash-2026-02-23. Thinking
    # must be OFF: the probe gave 64 output tokens with reasoning.enabled=false, but hit the 600 cap with no price by default.
    "qwen": {"provider": "openrouter", "model": "qwen/qwen3.5-flash-02-23", "extra": {"reasoning": {"enabled": False}, "max_tokens": 600, "provider": {"order": ["alibaba"], "allow_fallbacks": False}}},
}
LEVELS = [1.33, 1.69, 2.05]  # pre-R1 amendment: Lee Nash, midpoint, Lee monopoly


def seeds(key: str, n: int) -> list[int]:
    return sorted(random.Random(f"lee|{key}").sample(range(700000, 799999), n))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, f in MODELS.items():
        r0 = seeds(f"R0|{name}", 10)
        base = {"firm": f, "n": 4, "rounds": 200, "max_output_tokens": 600, "est_output_tokens": 120, "parallel_games": 5}
        out[f"R0_{name}"] = {"stage": "lee", "phase": "rep", "cell": f"homog-{name}", "seeds": r0, **base}
        for x in LEVELS:
            out[f"R1_{name}_x{x:.2f}"] = {"stage": "lee", "phase": "conf", "cell": f"homog-{name}", "seeds": seeds(f"R1|{name}|{x}", 10),
                                          "opening": {"x": x, "source_template": f"lee_rep_homog-{name}_s{{seed}}", "pool": r0,
                                                      "perm_seed": 4301, "window": [31, 50]}, **base}
    for k, c in out.items():
        p = OUT / f"{k}.yaml"
        new = yaml.safe_dump(c, sort_keys=False)
        if p.exists() and p.read_text() != new:
            sys.exit(f"refusing to overwrite {p} (content differs; delete it deliberately first)")
        if not p.exists():
            p.write_text(new)
    print(f"wrote {len(out)} configs to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
