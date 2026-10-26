"""Strict config validation. Unknown keys, wrong types, missing arm requirements and out-of-spec values are ERRORS:
a typo like `temprature: 0` must never silently fall back to a default.

validate(cfg) returns a list of error strings (empty == valid). load_config() raises ConfigError on any error.
"""
from __future__ import annotations
from .prompts import HINT_LINES

from pathlib import Path

import yaml

ARMS = {"L", "P", "FN", "FM", "X", "D", "Dfrozen", "Dplacebo", "S", "A", "O", "T", "Q", "STEP", "IMP"}
LIVE_FIRM2 = {"L", "X", "D", "Dfrozen", "Dplacebo", "S", "A", "O", "T", "IMP"}
# injected openings: S (v1) and O (plan v2) symmetric (x1 == x2), A asymmetric; T = transplanted real opening;
# Q = rival fixed at fixed_price (a ladder of q). PREFIX_ARMS carry a replayed history before live play.
OPENING_ARMS = {"S", "A", "O", "STEP", "IMP"}  # STEP / IMP: mechanism arms (addendum N3), opening at x
PREFIX_ARMS = OPENING_ARMS | {"T"}
PHASES = {"disc", "conf"}
PROVIDERS = {"vllm", "openai", "anthropic", "mock", "ifm"}

TOP = {
    "stage": str, "arm": str, "prompt": str, "rounds": int, "sampling": dict, "firm1": dict, "firm2": dict,
    "seeds": list, "seed_assignment": str, "seeds_from": str, "n_runs": int, "source_run_template": str,
    "replay_map": dict, "shock_round": int, "window": int, "parallel_games": int, "est_output_tokens": int,
    "market": dict, "identity1": dict, "identity2": dict, "identity_label": str, "strict": bool, "health": dict,
    "notes": str, "opening": list, "opening_source": dict, "phase": str, "fixed_price": (int, float), "view": dict, "step": dict, "impulse": dict, "supervision": dict, "pressure": dict,
}
OPENING_SOURCE = {"template": str, "pool": list, "perm_seed": int, "cross_model": bool}
SAMPLING = {"temperature": (int, float), "top_p": (int, float), "max_tokens": int}
FIRM = {"provider": str, "model": str, "base_url": str, "strategy": str, "latency_s": (int, float),
        "reasoning_effort": (str, type(None)), "extra_body": dict, "send_top_p": bool, "verbosity": str, "effort": str, "thinking": str, "timeout_s": (int, float), "max_attempts": int, "api_key_env": str}
MARKET = {"a": (int, float), "a0": (int, float), "c": (int, float), "mu": (int, float), "c2": (int, float)}
IDENTITY = {"self_name": (str, type(None)), "rival_name": (str, type(None))}
HEALTH = {"window": int, "max_fallback_rate": (int, float), "max_truncation_rate": (int, float),
          "max_consecutive_transport_errors": int, "frozen_price_rounds": int}
INTERNAL = {"_config_path"}  # set by the runner


class ConfigError(ValueError):
    pass


def _check(d: dict, schema: dict, where: str, errs: list[str]) -> None:
    for k, v in d.items():
        if k in INTERNAL:
            continue
        if k not in schema:
            errs.append(f"{where}: unknown key {k!r} (allowed: {sorted(schema)})")
        elif not isinstance(v, schema[k]) or (schema[k] is int and isinstance(v, bool)):
            errs.append(f"{where}.{k}: expected {schema[k]}, got {type(v).__name__} {v!r}")


def validate(cfg: dict) -> list[str]:
    errs: list[str] = []
    _check(cfg, TOP, "config", errs)
    for k in ("stage", "arm", "prompt", "rounds", "sampling", "firm1"):
        if k not in cfg:
            errs.append(f"config: missing required key {k!r}")
    if errs:
        return errs
    arm = cfg["arm"]
    if arm not in ARMS:
        errs.append(f"arm {arm!r} not in {sorted(ARMS)}")
    if cfg["prompt"] not in ("E0", "E1"):
        errs.append(f"prompt {cfg['prompt']!r} not in E0/E1")
    if not 1 <= cfg["rounds"] <= 1000:
        errs.append(f"rounds={cfg['rounds']} outside [1, 1000]")
    _check(cfg["sampling"], SAMPLING, "sampling", errs)
    for k in SAMPLING:
        if k not in cfg["sampling"]:
            errs.append(f"sampling: missing {k!r} (no silent defaults)")
    s = cfg["sampling"]
    if isinstance(s.get("temperature"), (int, float)) and not 0 <= s["temperature"] <= 2:
        errs.append(f"sampling.temperature={s['temperature']} outside [0, 2]")
    if isinstance(s.get("top_p"), (int, float)) and not 0 < s["top_p"] <= 1:
        errs.append(f"sampling.top_p={s['top_p']} outside (0, 1]")
    for f in ("firm1", "firm2"):
        if f in cfg:
            _check(cfg[f], FIRM, f, errs)
            if cfg[f].get("provider") not in PROVIDERS:
                errs.append(f"{f}.provider {cfg[f].get('provider')!r} not in {sorted(PROVIDERS)}")
            if not cfg[f].get("model"):
                errs.append(f"{f}.model missing")
    if arm in LIVE_FIRM2 and "firm2" not in cfg:
        errs.append(f"arm {arm} needs firm2")
    if arm in ("FN", "FM", "P", "Q") and "firm2" in cfg:
        errs.append(f"arm {arm}: firm2 is not an agent; remove firm2 to avoid confusion")
    if arm == "L" and cfg.get("firm2", {}).get("model") != cfg["firm1"].get("model"):
        errs.append("arm L requires the SAME model for both firms (use arm X for cross-family)")
    if arm == "X" and cfg.get("firm2", {}).get("model") == cfg["firm1"].get("model"):
        errs.append("arm X requires DIFFERENT models")
    has_seeds, has_assign = "seeds" in cfg, "seed_assignment" in cfg
    if has_seeds == has_assign:
        errs.append("give exactly one of `seeds` or `seed_assignment`(+`seeds_from`)")
    if has_seeds:
        seeds = cfg["seeds"]
        if not seeds or not all(isinstance(x, int) and not isinstance(x, bool) for x in seeds):
            errs.append("seeds must be a non-empty list of ints")
        elif len(set(seeds)) != len(seeds):
            errs.append("seeds contain duplicates")
    if has_assign:
        want = {"L": "L", "D": "L", "Dfrozen": "L", "Dplacebo": "L", "P": "P"}.get(arm)
        if want is None:
            errs.append(f"arm {arm} must use explicit `seeds`, not the L/P seed_assignment")
        elif cfg.get("seeds_from") != want:
            errs.append(f"arm {arm} must use seeds_from: {want} (got {cfg.get('seeds_from')!r})")
    if "replay_map" in cfg:
        rm = cfg["replay_map"]
        if not isinstance(rm, dict) or not all(str(k).lstrip("-").isdigit() and isinstance(v, int) for k, v in rm.items()):
            errs.append("replay_map must map P seed -> L seed (ints)")
        elif has_seeds and {str(x) for x in cfg["seeds"]} - {str(k) for k in rm}:
            errs.append("replay_map is missing some P seeds")
    if arm in ("P", "D", "Dfrozen", "Dplacebo") and "source_run_template" not in cfg:
        errs.append(f"arm {arm} needs source_run_template")
    if "source_run_template" in cfg and "{seed}" not in cfg["source_run_template"]:
        errs.append("source_run_template must contain {seed}")
    if arm in OPENING_ARMS:
        op = cfg.get("opening")
        if not (isinstance(op, list) and len(op) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in op)):
            errs.append("opening must be [x1, x2] (numbers)")
        else:
            if not all(1.0 <= v <= 2.5 for v in op):
                errs.append(f"opening {op} outside [1, 2.5]")
            if arm in ("S", "O", "STEP", "IMP") and op[0] != op[1]:
                errs.append(f"arm {arm} is symmetric: opening needs x1 == x2 (use arm A)")
            if arm == "A" and op[0] == op[1]:
                errs.append("arm A is asymmetric: opening needs x1 != x2 (use arm S)")
    if arm in PREFIX_ARMS:
        src = cfg.get("opening_source")
        if not isinstance(src, dict):
            errs.append(f"arm {arm} needs opening_source {{template, pool, perm_seed}}")
        else:
            _check(src, OPENING_SOURCE, "opening_source", errs)
            for k in ("template", "pool", "perm_seed"):
                if k not in src:
                    errs.append(f"opening_source: missing {k!r}")
            if "{seed}" not in str(src.get("template", "")):
                errs.append("opening_source.template must contain {seed}")
            pool = src.get("pool")
            if not (isinstance(pool, list) and pool and all(isinstance(x, int) and not isinstance(x, bool) for x in pool)
                    and len(set(pool)) == len(pool)):
                errs.append("opening_source.pool must be a non-empty list of distinct int seeds")
        w = cfg.get("window")
        if not isinstance(w, int) or w < 1:
            errs.append(f"arm {arm} needs window = number of live rounds (the continuation; int >= 1)")
        elif (cfg["rounds"] - w) < 20 or (cfg["rounds"] - w) % 20:
            errs.append(f"arm {arm}: rounds - window = {cfg['rounds'] - w} prefix rows; must be a positive multiple of 20")
    if arm == "T" and "opening" in cfg:
        errs.append("arm T transplants a real opening as-is: no `opening` level")
    if arm not in PREFIX_ARMS and ("opening" in cfg or "opening_source" in cfg):
        errs.append(f"opening/opening_source only apply to arms {sorted(PREFIX_ARMS)}, not {arm}")
    if arm in ("O", "T", "Q") and cfg.get("phase") not in PHASES:
        errs.append(f"arm {arm} needs phase in {sorted(PHASES)} (discovery vs confirmation)")
    if "phase" in cfg and cfg["phase"] not in PHASES:
        errs.append(f"phase must be one of {sorted(PHASES)}")
    if arm == "STEP":  # firm 1 live; firm 2 scripted: the opening level x, then x + delta from round `round`
        st = cfg.get("step")
        if not (isinstance(st, dict) and set(st) == {"delta", "round"} and isinstance(st["delta"], (int, float))
                and isinstance(st["round"], int) and cfg["rounds"] - cfg.get("window", 0) < st["round"] <= cfg["rounds"]):
            errs.append("arm STEP needs step {delta: number, round: int inside the live rounds}")
        if "firm2" in cfg:
            errs.append("arm STEP: firm 2 is scripted, remove firm2")
    elif "step" in cfg:
        errs.append("step only applies to arm STEP")
    if arm == "IMP":  # both live; firm 1 forced to p1(round-1) + delta for `hold` rounds
        im = cfg.get("impulse")
        if not (isinstance(im, dict) and set(im) - {"hold"} == {"delta", "round"} and isinstance(im["delta"], (int, float))
                and isinstance(im["round"], int) and cfg["rounds"] - cfg.get("window", 0) < im["round"] <= cfg["rounds"]
                and isinstance(im.get("hold", 1), int) and im.get("hold", 1) >= 1):
            errs.append("arm IMP needs impulse {delta: number, round: int inside the live rounds, hold: int >= 1}")
    elif "impulse" in cfg:
        errs.append("impulse only applies to arm IMP")
    if arm in ("STEP", "IMP") and cfg.get("phase") not in PHASES:
        errs.append(f"arm {arm} needs phase")
    if arm == "Q":
        q = cfg.get("fixed_price")
        if not isinstance(q, (int, float)) or not 1.0 <= q <= 2.5:
            errs.append("arm Q needs fixed_price in [1, 2.5]")
        if "firm2" in cfg:
            errs.append("arm Q: firm 2 is fixed, remove firm2")
    elif "fixed_price" in cfg:
        errs.append("fixed_price only applies to arm Q")
    if "view" in cfg:
        v = cfg["view"]
        if set(v) - {"label", "anchor", "price_only", "line"} or ("label" in v and v["label"] is not True) or \
                ("price_only" in v and v["price_only"] is not True) or ("line" in v and v["line"] not in HINT_LINES) or \
                ("anchor" in v and not (isinstance(v["anchor"], (int, float)) and 1.0 <= v["anchor"] <= 2.5)):
            errs.append("view must be {label: true} / {anchor: <number in [1, 2.5]>} / {price_only: true} / {line: br_hint|sham}")
    if "pressure" in cfg:
        pr = cfg["pressure"]
        if set(pr) - {"frame", "target"} or pr.get("frame") not in ("gain", "loss") or pr.get("target") not in ("both", "firm1") \
                or arm not in ("L", "O"):
            errs.append("pressure must be {frame: gain|loss, target: both|firm1}, on arm L or O")
    if "supervision" in cfg:
        sv = cfg["supervision"]
        if set(sv) - {"frame", "agent", "supervisor", "max_tokens", "mode", "yoke"} or sv.get("frame") not in ("shared", "individual") \
                or not isinstance(sv.get("agent"), bool) or arm not in ("L", "O"):
            errs.append("supervision must be {frame: shared|individual, agent: true|false, supervisor?: firm spec, "
                        "max_tokens?: int}, on arm L or O")
        mode = sv.get("mode", "note")  # v2 (docs/SUPERVISION.md, supervision v2): note | review | yoked
        if mode not in ("note", "review", "yoked") or (mode == "review" and sv.get("agent") is not True) \
                or (mode == "yoked" and sv.get("agent") is not False):
            errs.append("supervision.mode must be note | review (agent: true) | yoked (agent: false)")
        if mode == "yoked":
            yk = sv.get("yoke")
            if not isinstance(yk, dict) or sorted(int(k) for k in yk) != sorted(cfg.get("seeds") or []):
                errs.append("yoked supervision needs yoke: {<seed>: <review run_id>} covering exactly the config's seeds")
            else:
                for k, rid in yk.items():
                    if not isinstance(rid, str) or rid.endswith(f"_s{int(k)}") or "_supSR_" not in rid and "_supIR_" not in rid:
                        errs.append(f"yoke {k} -> {rid!r}: must be a review run (supSR/supIR) with a DIFFERENT seed")
        elif "yoke" in sv:
            errs.append("supervision.yoke only applies to mode yoked")
    if arm in ("D", "Dfrozen", "Dplacebo"):
        k = cfg.get("shock_round")
        if not isinstance(k, int) or not 2 <= k <= cfg["rounds"]:
            errs.append(f"shock_round={k!r} must be an int in [2, rounds]")
    if arm == "P" and has_seeds and "replay_map" not in cfg:
        errs.append("arm P with explicit seeds needs replay_map {P_seed: L_seed}")
    for sect, schema in (("market", MARKET), ("identity1", IDENTITY), ("identity2", IDENTITY), ("health", HEALTH)):
        if sect in cfg:
            _check(cfg[sect], schema, sect, errs)
    if ("identity1" in cfg or "identity2" in cfg) and "identity_label" not in cfg:
        errs.append("identity arms need identity_label (goes into run_id)")
    for f in ("firm1", "firm2"):
        if cfg.get(f, {}).get("provider") in ("openai", "anthropic") and "est_output_tokens" not in cfg:
            errs.append(f"{f} is a paid API: set est_output_tokens so the dry-run cost estimate is meaningful")
    return errs


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path}: not a YAML mapping")
    errs = validate(cfg)
    if errs:
        raise ConfigError(f"{path}: {len(errs)} config error(s):\n  - " + "\n  - ".join(errs))
    if "replay_map" in cfg:  # YAML may give int keys; the runner looks keys up as strings
        cfg["replay_map"] = {str(k): v for k, v in cfg["replay_map"].items()}
    cfg["_config_path"] = str(path)
    return cfg
