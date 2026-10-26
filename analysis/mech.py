"""Mechanism arms STEP and IMP (plan v2 addendum): configs, the N4 mock gate, and the H4/H5 analysis.
Frozen at tag `prereg-mech` before any STEP/IMP LLM call.

  .venv/bin/python analysis/mech.py configs           # write configs/mech/*.yaml (fixed once written)
  .venv/bin/python analysis/mech.py gate              # N4: every control through STEP + IMP (mock), vs the hand table
  .venv/bin/python analysis/mech.py analyse --runs-dir runs_cluster   # H4/H5 for every LLM stage present

Design: 20-row injected opening at x = 1.70 (the shared plan-v2 window pool), then 150 live rounds (21-170). The
SAME seeds in every cell of an arm (common random numbers). Shock at round 41.
  STEP: firm 1 live; firm 2 scripted = 1.70, then 1.70 + delta from round 41; delta in {-0.15, 0, +0.15}.
  IMP : both live; firm 1 forced to p1(40) + delta at round 41 only (runtime), then released; delta in {-0.15, 0, +0.15}.
Outcomes: STEP = mean p1 over rounds 131-170; IMP = mean of both firms over rounds 131-170 ("late market").
H4 (STEP): rho_up = (up - hold)/0.15, rho_down = (hold - down)/0.15, A = rho_down - rho_up, all per SEED (CRN
  contrasts). Inference: exact sign-flip test on per-seed contrasts and one-sided t bounds.
  ratchet: A_hat > 0.5, LB95(A) > 0, LB95(rho_down) > 0.5. Sub-label "matching" if 90% CI(rho_down) is inside
           (0.7, 1.15); "BR-landing" if LB95(rho_down) > 1.15.
  reverse: A_hat < -0.5 and UB95(A) < 0.
  symmetric: 90% CI(A) inside +-0.5; then matcher-like if LB95(mean rho) > 0.7, BR-like if the 90% CI of mean rho is
             inside (0.2, 0.65), rival-blind if UB95(mean rho) < 0.2.
  else inconclusive.
H5 (IMP): pi_dir = dp2(42)/dp1(41) vs placebo; H_dir = (late - late_placebo)/delta, per seed.
  one-way hysteresis: LB95(H_down - H_up) > 0, mean H_down > 0.5, UB95(H_up) < 0.3
  two-way persistence: both LB95(H) > 0.3 and |H_down - H_up| CI inside +-0.3
  transient: both UB95(H) < 0.25;  else inconclusive.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from collusion.runlog import read_jsonl  # noqa: E402

X, K, D = 1.70, 41, 0.15
DELTAS = (-D, 0.0, D)
LATE = (131, 170)
POOL = "s3-qwen25_14b_L_E1_Qwen25-14B-Instruct-AWQ_s{seed}"
POOL_SEEDS = [s for s in json.loads((ROOT / "configs/seeds/qwen25_14b.json").read_text())["L"] if s not in (61860, 210629)]
SAMP = lambda mt: {"temperature": 0.7, "top_p": 1.0, "max_tokens": mt}  # noqa: E731
LLM = {"qwen3off": ({"provider": "vllm", "model": "Qwen/Qwen3-14B-AWQ",
                     "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}, 2048, SAMP(2048), 20),
       "qwen25": ({"provider": "vllm", "model": "Qwen/Qwen2.5-14B-Instruct-AWQ"}, 2048, SAMP(2048), 20),
       "sonnet55": ({"provider": "anthropic", "model": "claude-sonnet-5-5", "effort": "low", "thinking": "adaptive"},
                    2048, {"temperature": 1.0, "top_p": 1.0, "max_tokens": 2048}, 8)}
# N4 controls (hand table, addendum). Noise sd 0.005 = "small noise". (hold, up, down) late p1 for STEP.
CONTROLS = {"match": "match", "averager": "averager", "cutmatch": "cutmatch", "winstay": "winstay",
            "satisfice": "satisfice:0.025", "raisematch": "raisematch", "br": "br", "grim": "grim:1.80",
            "hillclimb": "hillclimb:0.02"}
HAND_STEP = {"match": (1, 1, "symmetric, matcher-like"), "averager": (1, 1, "symmetric, matcher-like"),
             "cutmatch": (0, 1, "ratchet, matching"), "winstay": (0, 1, "ratchet, matching"),
             "satisfice": (0, 1.29, "ratchet, BR-landing"), "raisematch": (1, 0, "reverse"),
             "br": (0.41, 0.44, "symmetric, BR-like"), "grim": (2.18, 0, "reverse"), "hillclimb": (0.4, 0.4, "symmetric, BR-like")}
HAND_IMP = {"averager": (0.5, 0.5, 0.5, 0.5), "match": (1, 1, 0.5, 0.5), "cutmatch": (1, 0, 1, 0), "winstay": (1, 0, 1, 0),
            "raisematch": (0, 1, 0, 1), "br": (0.43, 0.43, 0, 0)}  # (pi_down, pi_up, H_down, H_up)


def seeds(key: str, n: int, master: int = 97000) -> list[int]:
    return sorted(random.Random(f"{master}|{key}").sample(range(master * 10, master * 10 + 9999), n))


def step_cfg(stage, firm, samp, sd, d, phase="conf", est=None) -> dict:
    c = {"stage": stage, "arm": "STEP", "phase": phase, "prompt": "E1", "rounds": 170, "window": 150, "opening": [X, X],
         "opening_source": {"template": POOL, "pool": POOL_SEEDS, "perm_seed": 4201, "cross_model": True},
         "step": {"delta": d, "round": K}, "seeds": sd, "sampling": samp, "parallel_games": min(len(sd), 20), "firm1": firm}
    if est:
        c["est_output_tokens"] = est
    return c


def imp_cfg(stage, firm, samp, sd, d, phase="conf", hold=1) -> dict:
    im = {"delta": d, "round": K}
    if hold != 1:
        im["hold"] = hold
    return {"stage": stage, "arm": "IMP", "phase": phase, "prompt": "E1", "rounds": 170, "window": 150, "opening": [X, X],
            "opening_source": {"template": POOL, "pool": POOL_SEEDS, "perm_seed": 4201, "cross_model": True},
            "impulse": im, "seeds": sd, "sampling": samp, "parallel_games": min(len(sd), 30), "firm1": firm, "firm2": firm}


def write_configs() -> int:
    out = ROOT / "configs" / "mech"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for m, (firm, _mt, samp, ns) in LLM.items():
        sd = seeds(f"{m}|STEP", ns)
        for d in DELTAS:
            c = step_cfg(f"mSTEP-{m}", firm, samp, sd, d, est=1100 if m == "sonnet55" else None)
            p = out / f"STEP_{m}_d{d:+.2f}.yaml"
            if p.exists():
                sys.exit(f"refusing to overwrite {p}")
            p.write_text(yaml.safe_dump(c, sort_keys=False))
            n += 1
        if m == "sonnet55":
            continue  # Sonnet: STEP only (its O band cells are separate configs)
        sd = seeds(f"{m}|IMP", ns)
        for d in DELTAS:
            p = out / f"IMP_{m}_d{d:+.2f}.yaml"
            if p.exists():
                sys.exit(f"refusing to overwrite {p}")
            p.write_text(yaml.safe_dump(imp_cfg(f"mIMP-{m}", firm, samp, sd, d), sort_keys=False))
            n += 1
    firm, _mt, samp, _ = LLM["sonnet55"]                                  # Sonnet arm O band x 4 (plan v2 paid (b))
    for x in (1.55, 1.75, 1.95):
        c = {"stage": "v2O-sonnet55", "arm": "O", "phase": "conf", "prompt": "E1", "rounds": 170, "window": 150,
             "opening": [x, x], "opening_source": {"template": POOL, "pool": POOL_SEEDS, "perm_seed": 4102, "cross_model": True},
             "seeds": seeds(f"sonnetO|{x}", 4), "sampling": samp, "parallel_games": 4, "est_output_tokens": 1100,
             "firm1": firm, "firm2": firm}
        p = out / f"O_sonnet55_x{x:.3f}.yaml"
        if p.exists():
            sys.exit(f"refusing to overwrite {p}")
        p.write_text(yaml.safe_dump(c, sort_keys=False))
        n += 1
    print(f"wrote {n} configs to {out}")
    return 0


# ------------------------------------------------------------------ analysis
def _rounds(d: Path) -> dict[int, dict]:
    return {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}


def _ok(d: Path) -> bool:
    s, v = d / "summary.json", d / "validation.json"
    return s.exists() and json.loads(s.read_text()).get("status") == "complete" and v.exists() and json.loads(v.read_text()).get("ok")


def collect(runs: Path, stage: str, arm: str) -> dict[float, dict[int, dict]]:
    """cell delta -> seed -> rounds (complete + valid only)."""
    out: dict[float, dict[int, dict]] = {}
    for d in runs.glob(f"{stage}_{arm}_*"):
        if not _ok(d):
            continue
        man = json.loads((d / "manifest.json").read_text())
        c = man["config"]
        dl = float((c.get("step") or c.get("impulse"))["delta"])
        out.setdefault(dl, {})[man["seed"]] = _rounds(d)
    return out


def _signflip_p(x: np.ndarray, two_sided=True) -> float:
    n = len(x)
    if n == 0:
        return float("nan")
    if n <= 20:
        s = 1 - 2 * ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1)
    else:
        s = np.random.default_rng(7).choice([-1, 1], (100_000, n))
    m = (s * x).mean(1)
    return float((np.abs(m) >= abs(x.mean()) - 1e-12).mean()) if two_sided else float((m >= x.mean() - 1e-12).mean())


def t95(df: int) -> float:
    """One-sided 95% Student-t quantile by the standard series expansion in 1/df (error < 1e-3 for df >= 4)."""
    z = 1.6448536269514722
    return (z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)
            + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * df**3))


def _bounds(x: np.ndarray) -> dict:
    n, mu = len(x), float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t95_ = t90 = t95(n - 1)  # the one-sided 95% bound = the edge of the two-sided 90% CI
    return {"mean": mu, "se": se, "n": n, "lb95": mu - t95_ * se, "ub95": mu + t95_ * se, "ci90": [mu - t90 * se, mu + t90 * se],
            "p_signflip": _signflip_p(x)}


def late(R: dict, firm: str = "1") -> float:
    return float(np.mean([R[t][f"p{firm}"] for t in range(LATE[0], LATE[1] + 1)]) if firm != "both"
                 else np.mean([(R[t]["p1"] + R[t]["p2"]) / 2 for t in range(LATE[0], LATE[1] + 1)]))


def h4(cells: dict) -> dict:
    up, hold, down = cells.get(D, {}), cells.get(0.0, {}), cells.get(-D, {})
    common = sorted(set(up) & set(hold) & set(down))
    if len(common) < 3:
        return {"n": len(common), "label": "insufficient"}
    lu = np.array([late(up[s]) for s in common])
    lh = np.array([late(hold[s]) for s in common])
    ld = np.array([late(down[s]) for s in common])
    r_up, r_dn = (lu - lh) / D, (lh - ld) / D
    A, rbar = r_dn - r_up, (r_dn + r_up) / 2
    bA, bd, bu, bm = _bounds(A), _bounds(r_dn), _bounds(r_up), _bounds(rbar)
    if bA["mean"] > 0.5 and bA["lb95"] > 0 and bd["lb95"] > 0.5:
        lab = "ratchet" + (", matching" if 0.7 < bd["ci90"][0] and bd["ci90"][1] < 1.15 else
                           ", BR-landing" if bd["lb95"] > 1.15 else "")
    elif bA["mean"] < -0.5 and bA["ub95"] < 0:
        lab = "reverse"
    elif -0.5 < bA["ci90"][0] and bA["ci90"][1] < 0.5:
        lab = ("symmetric, matcher-like" if bm["lb95"] > 0.7 else
               "symmetric, BR-like" if 0.2 < bm["ci90"][0] and bm["ci90"][1] < 0.65 else
               "symmetric, rival-blind" if bm["ub95"] < 0.2 else "symmetric")
    else:
        lab = "inconclusive"
    trans = {f"{d:+.2f}": float(np.mean([np.mean([cells[d][s][t]["p1"] for t in range(42, 47)]) for s in common])) for d in DELTAS}
    rn = float(np.mean([abs(cells[0.0][s][t]["p1"] * 100 - round(cells[0.0][s][t]["p1"] * 100)) < 1e-6
                        for s in common for t in range(21, 171)]))
    return {"n": len(common), "late": {"hold": float(lh.mean()), "up": float(lu.mean()), "down": float(ld.mean())},
            "rho_up": bu, "rho_down": bd, "A": bA, "rho_mean": bm, "label": lab, "transient_42_46": trans,
            "round_number_share_hold": rn, "hold_drift_vs_1.70": float(lh.mean() - X),
            "hold_by_r41": float(np.mean([hold[s][K - 1]["p1"] for s in common]) - X)}


def h5(cells: dict) -> dict:
    up, pl, dn = cells.get(D, {}), cells.get(0.0, {}), cells.get(-D, {})
    common = sorted(set(up) & set(pl) & set(dn))
    if len(common) < 3:
        return {"n": len(common), "label": "insufficient"}

    def per(cell, d):
        pi = [(cell[s][42]["p2"] - pl[s][42]["p2"]) / (cell[s][41]["p1"] - pl[s][41]["p1"])
              if abs(cell[s][41]["p1"] - pl[s][41]["p1"]) > 1e-9 else np.nan for s in common]
        H = [(late(cell[s], "both") - late(pl[s], "both")) / d for s in common]
        return np.array(pi, float), np.array(H, float)
    pd_, Hd = per(dn, -D)
    pu, Hu = per(up, D)
    bd, bu, bdiff = _bounds(Hd), _bounds(Hu), _bounds(Hd - Hu)
    if bdiff["lb95"] > 0 and bd["mean"] > 0.5 and bu["ub95"] < 0.3:
        lab = "one-way hysteresis"
    elif bd["lb95"] > 0.3 and bu["lb95"] > 0.3 and -0.3 < bdiff["ci90"][0] and bdiff["ci90"][1] < 0.3:
        lab = "two-way persistence"
    elif bd["ub95"] < 0.25 and bu["ub95"] < 0.25:
        lab = "transient"
    else:
        lab = "inconclusive"
    return {"n": len(common), "pi_down": float(np.nanmean(pd_)), "pi_up": float(np.nanmean(pu)), "H_down": bd, "H_up": bu,
            "H_down_minus_up": bdiff, "label": lab, "placebo_late": float(np.mean([late(pl[s], "both") for s in common]))}


def analyse(runs: Path, out: Path) -> dict:
    res = {"STEP": {}, "IMP": {}}
    stages = {d.name.split("_")[0] for d in runs.glob("m*_STEP_*")} | {d.name.split("_")[0] for d in runs.glob("m*_IMP_*")}
    for st in sorted(stages):
        if "_STEP_" in next(iter(runs.glob(f"{st}_*")), Path("x")).name or list(runs.glob(f"{st}_STEP_*")):
            if list(runs.glob(f"{st}_STEP_*")):
                res["STEP"][st] = h4(collect(runs, st, "STEP"))
        if list(runs.glob(f"{st}_IMP_*")):
            res["IMP"][st] = h5(collect(runs, st, "IMP"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "mech.json").write_text(json.dumps(res, indent=1, default=float))
    for st, e in res["STEP"].items():
        if "A" in e:
            print(f"H4 {st:<26} n={e['n']} late h/u/d {e['late']['hold']:.3f}/{e['late']['up']:.3f}/{e['late']['down']:.3f} "
                  f"rho_up {e['rho_up']['mean']:+.2f} rho_down {e['rho_down']['mean']:+.2f} A {e['A']['mean']:+.2f} "
                  f"[LB {e['A']['lb95']:+.2f}] -> {e['label']}")
    for st, e in res["IMP"].items():
        if "H_down" in e:
            print(f"H5 {st:<26} n={e['n']} pi d/u {e['pi_down']:+.2f}/{e['pi_up']:+.2f} H d/u {e['H_down']['mean']:+.2f}/"
                  f"{e['H_up']['mean']:+.2f} -> {e['label']}")
    return res


# ------------------------------------------------------------------ N4 mock gate
def gate(runs: Path, noise: float = 0.0) -> int:
    """N4. The hand table assumes NOISE-FREE policies: the EXACT pass (noise 0, stage prefix mgate0-) must match
    it. The noisy pass (noise 0.005, stage prefix mgate-) is robustness only: path-dependent controls
    (satisfice, winstay, cutmatch, raisematch) random-walk under noise, which is real behaviour, not a code bug."""
    from collusion.run import main as run_main
    pre = "mgate0" if noise == 0 else "mgate"
    tmp = ROOT / "results" / "mech" / f"_{pre}_configs"
    tmp.mkdir(parents=True, exist_ok=True)
    sd = seeds("gate", 20, master=98000)
    for pol, strat in CONTROLS.items():
        f = {"provider": "mock", "model": f"gate-{pol}", "strategy": strat + (f"~{noise}" if noise else "")}
        for d in DELTAS:
            for kind, c in (("STEP", step_cfg(f"{pre}-{pol}", f, SAMP(2048), sd, d)),
                            ("IMP", imp_cfg(f"{pre}-{pol}", f, SAMP(2048), sd, d))):
                p = tmp / f"{kind}_{pol}_{d:+.2f}.yaml"
                p.write_text(yaml.safe_dump(c))
                with contextlib.redirect_stdout(io.StringIO()):
                    if run_main([str(p), "--runs-dir", str(runs)]) != 0:
                        print(f"RUN FAILED {kind} {pol} {d}")
    res = analyse(runs, ROOT / "results" / "mech" / pre)
    res = {k: {st: e for st, e in v.items() if st.startswith(pre + "-")} for k, v in res.items()}
    rows, miss = [], []
    for pol in CONTROLS:
        e = res["STEP"].get(f"{pre}-{pol}", {})
        hu, hd, lab = HAND_STEP[pol]
        ok = (abs(e["rho_up"]["mean"] - hu) <= 0.1 and abs(e["rho_down"]["mean"] - hd) <= 0.1 and
              abs(e["A"]["mean"] - (hd - hu)) <= 0.1 and e["label"].startswith(lab.split(",")[0]))
        rows.append({"policy": pol, "arm": "STEP", "rho_up": e["rho_up"]["mean"], "rho_down": e["rho_down"]["mean"],
                     "A": e["A"]["mean"], "label": e["label"], "hand": HAND_STEP[pol], "pass": ok})
        if not ok:
            miss.append(f"STEP {pol}")
        if pol in HAND_IMP:
            i = res["IMP"][f"{pre}-{pol}"]
            pdn, pup, Hd, Hu = HAND_IMP[pol]
            ok = all(abs(a - b) <= 0.1 for a, b in ((i["pi_down"], pdn), (i["pi_up"], pup), (i["H_down"]["mean"], Hd), (i["H_up"]["mean"], Hu)))
            rows.append({"policy": pol, "arm": "IMP", "pi_down": i["pi_down"], "pi_up": i["pi_up"], "H_down": i["H_down"]["mean"],
                         "H_up": i["H_up"]["mean"], "label": i["label"], "hand": HAND_IMP[pol], "pass": ok})
            if not ok:
                miss.append(f"IMP {pol}")
    (ROOT / "results" / "mech" / pre / "gate_table.json").write_text(json.dumps({"rows": rows, "misses": miss}, indent=1, default=float))
    for r in rows:
        print(r["arm"], r["policy"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("arm", "policy", "hand")}, "hand", r["hand"])
    print("GATE PASS" if not miss else f"GATE MISSES: {miss}")
    return 0 if not miss else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["configs", "gate", "analyse"])
    ap.add_argument("--runs-dir", default=str(ROOT / "runs_cluster"))
    ap.add_argument("--noise", type=float, default=0.0)
    a = ap.parse_args(argv)
    if a.cmd == "configs":
        return write_configs()
    if a.cmd == "gate":
        return gate(Path(a.runs_dir), a.noise)
    analyse(Path(a.runs_dir), ROOT / "results" / "mech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
