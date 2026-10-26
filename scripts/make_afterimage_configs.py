"""Write the configs for the "Price Afterimage" experiments.

Every config is a copy of the pre-registered conf O config configs/v2/conf/O_qwen3off_x<x>.yaml (same window pool,
perm_seed 4102, cross_model, prompt E1, sampling), with ONLY these changed:
- the stage, using a non-"v2"/non-"cls-" prefix (relay-, rsn-, ai-, x0-), so the frozen v2.py never globs it;
- the seeds, from a NEW seed master (random.Random("afterimage|<exp>|<cell>|<x>") over 2,000,000-2,999,999),
  checked to be disjoint from every seed in configs/;
- the experiment's own change (rival, view, model or rounds), listed per experiment below.
Job i uses opening window order[i % 18], so 12 seeds use order[0..11] and 16 seeds order[0..15] (window-matched to
the conf runs on those windows).

  .venv/bin/python scripts/make_afterimage_configs.py            (writes configs/afterimage/<exp>/*.yaml)

Experiments (never-cut set):
  e1   relay-br / relay-copy: firm 2 = mock 'br' / 'tft' (the rival from round 21); x in {1.55, 1.75, 1.95} x 12
  e1b  relay-nash: firm 2 = mock 'fixed:1.4729'; x in {1.55, 1.75, 1.95} x 12
  e2a  rsn-on: both firms Qwen3-14B thinking ON (configs/models/qwen3_14b_on8k.yaml: max_tokens 8192);
       rounds 100 / window 80; x in {1.55, 1.95} x 12
  e5   ai-full / ai-po: x in {1.55, 1.75, 1.95} x 16; ai-po uses view {price_only: true}
  x0   estimator positive control, both firms mock: partial:w (w = 0.5**(1/h), h in {10, 50, 200, 800}), averager, br;
       x in {1.55, 1.75, 1.95} x 24; noise sigma in {0, 0.002, 0.004, 0.008} (mock '~sd' suffix); plus a 620-round
       arm (window 600) for h in {200, 800} at sigma 0. Stage carries h and sigma, because strategy is not in run_id.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "configs" / "v2" / "conf"
OUT = ROOT / "configs" / "afterimage"
MOCK = "mock"


def used_seeds() -> set[int]:
    s = set()
    for p in (ROOT / "configs").rglob("*.yaml"):
        c = yaml.safe_load(p.read_text()) or {}
        if isinstance(c, dict) and isinstance(c.get("seeds"), list):
            s |= {int(x) for x in c["seeds"] if isinstance(x, int)}
    return s


def base(x: float) -> dict:
    c = yaml.safe_load((CONF / f"O_qwen3off_x{x:.3f}.yaml").read_text())
    assert c["arm"] == "O" and c["phase"] == "conf" and c["opening"] == [x, x] and c["opening_source"]["perm_seed"] == 4102
    return c


def mock(strategy: str, label: str = MOCK) -> dict:
    return {"provider": "mock", "model": label, "strategy": strategy}


MASTER = 97000   # master*10 + 9999 < 1,100,000; >= 1000 from 91000 / 93000 and every other master
MODELS = ROOT / "configs" / "models"


def model(name: str) -> dict:
    return yaml.safe_load((MODELS / f"{name}.yaml").read_text())


def cells() -> list[tuple[str, str, float | None, int, dict]]:
    """(experiment, stage, x or None for a cold start, n_seeds, changes). x0 keeps its original seed scheme (already
    run); every other cell is an ait- stage seeded from MASTER."""
    out = []
    for x in (1.55, 1.75, 1.95):
        out.append(("e1", "ait-e1-br-qwen3off", x, 12, {"firm2": mock("br", "ctl-br"), "parallel_games": 24}))
        out.append(("e1", "ait-e1-copy-qwen3off", x, 12, {"firm2": mock("tft", "ctl-tft"), "parallel_games": 24}))
        out.append(("e1b", "ait-e1b-nash-qwen3off", x, 12, {"firm2": mock("fixed:1.4729", "ctl-nash"), "parallel_games": 24}))
        out.append(("e5", "ait-e5-full-qwen3off", x, 16, {"parallel_games": 24}))
        out.append(("e5", "ait-e5-po-qwen3off", x, 16, {"view": {"price_only": True}, "parallel_games": 24}))
    # E4 API breadth. est_output_tokens = measured mean output per call in our ledgers:
    # Sonnet 5.5 463-505, Haiku 5.5 908-1018, GPT-6.1-sol 114-123.
    for m, est in (("haiku55", 1000), ("sonnet55", 500), ("gpt61sol", 125)):
        y = model(m)
        api = {"firm1": y["firm"], "firm2": y["firm"], "sampling": y["sampling"], "parallel_games": y["parallel_games"],
               "est_output_tokens": est}
        for x in (1.55, 1.75, 1.95):
            out.append(("e4", f"ait-e4-{m}", x, 12, api))
        out.append(("e4", f"ait-e4L-{m}", None, 12, api))
    y = model("gpt6luna")                                                           # E2b reasoning none vs high
    for eff, est in (("none", y["est_output_tokens"]), ("high", 1500)):
        firm = {**y["firm"], "reasoning_effort": eff}
        for x in (1.55, 1.95):
            out.append(("e2b", f"ait-e2b-luna-{eff}", x, 12, {"firm1": firm, "firm2": firm, "sampling": y["sampling"],
                                                              "parallel_games": 12, "est_output_tokens": est}))
    for x in (1.55, 1.95):                                                          # D2 lever 1: hint vs sham
        for kind, line in (("hint", "br_hint"), ("sham", "sham")):
            out.append(("d2", f"ait-d2-{kind}-qwen3off", x, 12, {"view": {"line": line}, "parallel_games": 24}))
    for q in (1.473, 1.925):                                                        # Q-hint: the FN gap gate
        out.append(("d2", "ait-d2-qhint-qwen3off", -q, 3, {"view": {"line": "br_hint"}}))
    for h in (10, 50, 200, 800):
        w = 0.5 ** (1 / h)
        for sd in (0, 0.002, 0.004, 0.008):
            strat = f"partial:{w:.6f}" + (f"~{sd}" if sd else "")
            for x in (1.55, 1.75, 1.95):
                out.append(("x0", f"x0-h{h}-sd{sd}", x, 24, {"firm1": mock(strat), "firm2": mock(strat), "parallel_games": 24}))
        if h in (200, 800):
            for x in (1.55, 1.75, 1.95):
                out.append(("x0", f"x0long-h{h}-sd0", x, 24, {"firm1": mock(f"partial:{w:.6f}"), "firm2": mock(f"partial:{w:.6f}"),
                                                             "rounds": 620, "window": 600, "parallel_games": 24}))
    for strat in ("averager", "br"):
        for x in (1.55, 1.75, 1.95):
            out.append(("x0", f"x0-{strat}-sd0", x, 24, {"firm1": mock(strat), "firm2": mock(strat), "parallel_games": 24}))
    return out


def seeds(exp: str, stage: str, x: float | None, k: int, taken: set) -> list[int]:
    if exp == "x0":                                   # the original scheme, kept so the run X0 configs regenerate
        rng = random.Random(f"afterimage|{exp}|{stage}|{x:.3f}")
        out = []
        while len(out) < k:
            v = rng.randrange(2_000_000, 3_000_000)
            if v not in taken and v not in out:
                out.append(v)
        return sorted(out)
    rng = random.Random(f"{MASTER}|{stage}|{'L' if x is None else f'{x:.3f}'}")
    out = []
    while len(out) < k:                               # skip any seed already used anywhere in configs/
        v = rng.randrange(MASTER * 10, MASTER * 10 + 10_000)
        if v not in taken and v not in out:
            out.append(v)
    return sorted(out)


def qarm(stage: str, q: float, k_seeds: list[int], changes: dict) -> dict:
    b = base(1.55)
    c = {k: b[k] for k in ("phase", "prompt", "sampling", "firm1")}
    return {"stage": stage, "arm": "Q", **c, "rounds": 100, "fixed_price": q, "seeds": k_seeds, "parallel_games": len(k_seeds),
            **changes}


def cold(stage: str, k_seeds: list[int], changes: dict) -> dict:
    """Arm L cold start, copied from make_v2_configs.conf_extras (no opening)."""
    c = {**base(1.55), "stage": stage, "arm": "L", "seeds": k_seeds, **changes}
    for f in ("opening", "opening_source", "window"):
        c.pop(f, None)
    return {**c, "window": 150}


def main() -> int:
    taken = used_seeds()
    n = 0
    for exp, stage, x, k, changes in cells():
        assert not stage.startswith(("v2", "cls-"))
        if exp == "x0":                               # already run: its files are frozen, never rewritten
            ex = OUT / exp / f"O_{stage}_x{x:.3f}.yaml"
            if ex.exists():
                taken |= {int(v) for v in yaml.safe_load(ex.read_text())["seeds"]}
                continue
        mine = {f for f in (OUT / exp).glob(f"*_{stage}_*.yaml")} if (OUT / exp).exists() else set()
        own = {int(v) for f in mine for v in (yaml.safe_load(f.read_text()) or {}).get("seeds", [])}
        sd = seeds(exp, stage, x, k, taken - own)
        taken |= set(sd)
        if x is None:
            c, p = cold(stage, sd, changes), OUT / exp / f"L_{stage}_coldstart.yaml"
        elif x < 0:                                   # arm Q at fixed rival price -x (copied from make_v2_configs)
            c, p = qarm(stage, -x, sd, changes), OUT / exp / f"Q_{stage}_q{-x:.3f}.yaml"
        else:
            c, p = {**base(x), "stage": stage, "seeds": sd, **changes}, OUT / exp / f"O_{stage}_x{x:.3f}.yaml"
        p.parent.mkdir(parents=True, exist_ok=True)
        new = yaml.safe_dump(c, sort_keys=False)
        if p.exists() and p.read_text() != new:
            sys.exit(f"refusing to overwrite {p} (content differs)")
        if not p.exists():
            p.write_text(new)
            n += 1
    print(f"wrote {n} configs under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
