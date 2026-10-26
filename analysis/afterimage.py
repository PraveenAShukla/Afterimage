"""Price Afterimage A(W): how much of a planted opening x is still in the price in window W (Track A;
EXPLORATORY unless a later tag pre-registers it).

  A(W) = OLS slope, across runs of one stage, of the window game-mean price on the nominal opening x (x in BAND).
  A copier keeps all of the plant (A = 1); a best responder keeps none of it (A = 0).

Estimators are the frozen ones in analysis/v2.py (slope, holm, ok, BAND, SEED); the within-level bootstrap repeats
v2.boot_slope_within's draws (resample runs within each x level), here for every window at once.

  .venv/bin/python analysis/afterimage.py --runs-dir runs_cluster [--runs-dir runs_afterimage] [--gate-only]

GATE (must pass before any number is used): this script's own data path recomputes beta_band (late = rounds 121-170)
and carry_slope_41_60 for every arm-O stage in results/v2/conf/numbers.json, and must match them to 9 significant
digits. On a miss it prints the differences and exits 2 without writing anything.

Output: results/afterimage/<exp>/numbers.json, exp in {conf, e1, e1b, e2a, e5, x0}, chosen by stage prefix.
Per stage: A(W) for 10-round bins and 20-round windows (both firms' mean, and per firm), n per window, CIs from the
within-level bootstrap and from a window-cluster bootstrap (resample the 18 opening window_ids), exclusion counts.
Window-paired contrasts (CONTRASTS) use a paired window-cluster bootstrap; each contrast is one Holm family over its
20-round windows. X0 (estimator positive control, rule in docs/AFTERIMAGE.md): the target for mock partial:w is
A_q(W) from analysis/x0_mc.py (the quantized process the mock actually runs); the analytic mean of w**(t - 20) is kept
as A_analytic. lambda-hat / half-life / D_0.5 are added to the model stages only when X0 passes.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2 import BAND, ROOT, SEED, holm, ok, slope  # noqa: E402

from collusion.runlog import read_jsonl  # noqa: E402

N_BOOT = 10_000
BINS = [(a, a + 9) for a in range(21, 171, 10)]                        # 21-30 ... 161-170
WINDOWS = [(41, 60), (61, 80), (81, 100), (101, 120), (121, 140), (141, 170)]
LONG = [(171, 220), (221, 320), (321, 470), (471, 620)]               # only runs that reach them (x0long)
RULE_WINDOWS = [(121, 170)]                                           # the pre-registered rule window (PREREG_AFTERIMAGE)
GATE = {"late": (121, 170), "carry": (41, 60)}
EXP = [("v2O", "conf"), ("ait-d2", "d2"), ("ait-e1b", "e1b"), ("ait-e1", "e1"), ("ait-e5", "e5"), ("ait-e4", "e4"), ("ait-e2b", "e2b"),
       ("relay-nash", "e1b"), ("relay-", "e1"), ("ai-", "e5"), ("rsn-", "e2a"), ("x0", "x0")]
CONTRASTS = [  # (exp, name, stage a, stage b, price key): A_a(W) - A_b(W), window-paired (PREREG_AFTERIMAGE.md)
    ("e1", "D_BR", "ait-e1-br-qwen3off", "v2O-qwen3off", "p1"),      # firm 1 = Qwen3; comparator = conf self-play
    ("e1", "D_COPY", "ait-e1-copy-qwen3off", "v2O-qwen3off", "p1"),
    ("e5", "D_PO", "ait-e5-po-qwen3off", "ait-e5-full-qwen3off", "p"),
    ("e2b", "D_high", "ait-e2b-luna-high", "ait-e2b-luna-none", "p"),
    ("e2b", "D_high8k", "ait-e2b-luna-high8k", "ait-e2b-luna-none", "p"),   # POST-DATA amendment (tag D3)
    ("d2", "D_hint", "ait-d2-hint-qwen3off", "ait-d2-sham-qwen3off", "p"),      # lever 1 primary (same server)
    ("d2", "D_sham_base", "ait-d2-sham-qwen3off", "ait-e5-full-qwen3off", "p"),  # secondary (E5 full, other server)
]
PARTIAL = re.compile(r"^partial:([\d.]+)")


def wname(w) -> str:
    return f"{w[0]}-{w[1]}"


def exp_of(stage: str) -> str | None:
    return next((e for p, e in EXP if stage.startswith(p)), None)


def wmean(by: dict, w, key: str) -> float | None:
    """Mean over rounds w[0]..w[1] of the game-mean price (key "p") or one firm's price; None if any round is missing."""
    ts = range(w[0], w[1] + 1)
    if not all(t in by for t in ts):
        return None
    return float(np.mean([(by[t]["p1"] + by[t]["p2"]) / 2 if key == "p" else by[t][key] for t in ts]))


def load(runs_dirs: list[Path]) -> tuple[list[dict], dict]:
    """One row per complete+valid arm-O conf run: nominal x, window_id, and per-window mean prices (None if short)."""
    rows, excl, seen = [], defaultdict(int), set()
    for rd in runs_dirs:
        for d in sorted(rd.iterdir()):
            mf = d / "manifest.json"
            if not mf.exists():
                continue
            man = json.loads(mf.read_text())
            c = man["config"]
            if man.get("kind") == "onestep" or c.get("phase") != "conf" or c.get("arm") != "O" or "opening" not in c:
                continue
            st = c["stage"]
            if exp_of(st) is None:
                continue
            if man["run_id"] in seen:                       # the same run pulled into two runs dirs
                excl[f"{st}|duplicate"] += 1
                continue
            if not ok(d):
                excl[f"{st}|not complete+valid"] += 1
                continue
            seen.add(man["run_id"])
            by = {r["t"]: r for r in read_jsonl(d / "rounds.jsonl")}

            ws = BINS + WINDOWS + LONG + RULE_WINDOWS + list(GATE.values())
            m = PARTIAL.match(c["firm1"].get("strategy", "") or "")
            rows.append({"stage": st, "exp": exp_of(st), "run_id": man["run_id"], "seed": man["seed"],
                         "model1": c["firm1"]["model"], "model2": c.get("firm2", {}).get("model"),
                         "x": float(c["opening"][0]), "window_id": man["meta"].get("window_id"),
                         "rounds": max(by), "w_partial": float(m.group(1)) if m else None,
                         "p": {wname(w): wmean(by, w, "p") for w in ws},
                         "p1": {wname(w): wmean(by, w, "p1") for w in ws},
                         "p2": {wname(w): wmean(by, w, "p2") for w in ws}})
    return rows, dict(excl)


def gate(rows: list[dict], ref_path: Path) -> list[str]:
    ref = json.loads(ref_path.read_text())["H1"]
    errs = []
    for st, e in sorted(ref.items()):
        band = [r for r in rows if r["stage"] == st and r["x"] in BAND]
        if len(band) != e["n_band"]:
            errs.append(f"{st}: n_band {len(band)} != {e['n_band']}")
            continue
        x = [r["x"] for r in band]
        for key, w in (("beta_band", GATE["late"]), ("carry_slope_41_60", GATE["carry"])):
            got = slope(x, [r["p"][wname(w)] for r in band])
            if f"{got:.9g}" != f"{e[key]:.9g}":
                errs.append(f"{st}/{key}: {got!r} != {e[key]!r}")
    return errs


def boot_within(x: np.ndarray, Y: np.ndarray, rng, n=N_BOOT) -> np.ndarray:
    """Within-level bootstrap (v2.boot_slope_within's draws) for every column of Y at once: (n, n_windows)."""
    idx = [np.where(x == v)[0] for v in sorted(set(x))]
    out = np.empty((n, Y.shape[1]))
    for b in range(n):
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx])
        out[b] = slopes(x[ii], Y[ii])
    return out


def slopes(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xc = x - x.mean()
    return (xc[:, None] * (Y - Y.mean(0))).sum(0) / (xc * xc).sum()


def cluster_draws(wids: list, rng, n=N_BOOT) -> list[np.ndarray]:
    """Window-cluster bootstrap: n resamples (with replacement) of the distinct opening window_ids."""
    u = np.array(sorted(set(wids)))
    return [rng.choice(u, len(u)) for _ in range(n)]


def boot_cluster(rows: list[dict], cols: list[str], key: str, draws) -> np.ndarray:
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["window_id"]].append(i)
    x = np.array([r["x"] for r in rows])
    Y = np.array([[r[key][c] for c in cols] for r in rows])
    out = np.full((len(draws), len(cols)), np.nan)
    for b, d in enumerate(draws):
        ii = np.concatenate([by[w] for w in d if w in by]).astype(int)
        if len(set(x[ii])) >= 2:
            out[b] = slopes(x[ii], Y[ii])
    return out


def ci(a: np.ndarray, q=(0.05, 0.95)) -> list:
    a = a[~np.isnan(a)]
    return [float(np.quantile(a, q[0])), float(np.quantile(a, q[1]))] if len(a) else [None, None]


def a_true(w: float, win) -> float:
    return float(np.mean([w ** (t - 20) for t in range(win[0], win[1] + 1)]))


def stage_block(g: list[dict], rng, draws) -> dict:
    cols = [wname(w) for w in BINS + WINDOWS + LONG + RULE_WINDOWS if all(r["p"][wname(w)] is not None for r in g)]
    x = np.array([r["x"] for r in g])
    e = {"model1": g[0]["model1"], "model2": g[0]["model2"], "n": len(g), "levels": sorted(set(x.tolist())),
         "n_per_level": {f"{v:.3f}": int((x == v).sum()) for v in sorted(set(x))},
         "window_ids": len({r["window_id"] for r in g}), "windows": cols, "A": {}}
    if len(set(x)) < 2:
        return e
    for key in ("p", "p1", "p2"):
        Y = np.array([[r[key][c] for c in cols] for r in g])
        pt = slopes(x, Y)
        bw = boot_within(x, Y, rng)
        bc = boot_cluster(g, cols, key, draws) if all(r["window_id"] is not None for r in g) else None
        e["A"][key] = {c: {"A": float(pt[j]), "ci90_within": ci(bw[:, j]), "ci95_within": ci(bw[:, j], (.025, .975)),
                           "p_le_half": float((bw[:, j] <= 0.5).mean()),
                           "ci90_cluster": ci(bc[:, j]) if bc is not None else None,
                           "ci95_cluster": ci(bc[:, j], (.025, .975)) if bc is not None else None}
                       for j, c in enumerate(cols)}
    e["mean_price"] = {f"{v:.3f}": {c: float(np.mean([r["p"][c] for r in g if r["x"] == v])) for c in cols}
                       for v in sorted(set(x))}
    if g[0]["w_partial"] is not None:
        w = g[0]["w_partial"]
        e["A_analytic"] = {c: a_true(w, tuple(map(int, c.split("-")))) for c in cols}
    return e


def contrast(ga: list[dict], gb: list[dict], rng, key: str = "p") -> dict:
    """Window-paired: only window_ids both stages ran; resample those ids, A_a - A_b per window (20-round windows
    form the Holm family; the rule window 121-170 is reported beside it, outside the family)."""
    common = sorted({r["window_id"] for r in ga} & {r["window_id"] for r in gb})
    ga = [r for r in ga if r["window_id"] in common]
    gb = [r for r in gb if r["window_id"] in common]
    cols = [wname(w) for w in WINDOWS + RULE_WINDOWS if all(r[key][wname(w)] is not None for r in ga + gb)]
    if len(common) < 2 or not cols:
        return {"n_windows": len(common), "windows": cols}
    draws = cluster_draws(common, rng)
    da = boot_cluster(ga, cols, key, draws)
    db = boot_cluster(gb, cols, key, draws)
    d = da - db
    pa = slopes(np.array([r["x"] for r in ga]), np.array([[r[key][c] for c in cols] for r in ga]))
    pb = slopes(np.array([r["x"] for r in gb]), np.array([[r[key][c] for c in cols] for r in gb]))
    p = {}
    for j, c in enumerate(cols):
        dj = d[~np.isnan(d[:, j]), j]
        p[c] = float(min(1.0, 2 * min((dj <= 0).mean(), (dj >= 0).mean()))) if len(dj) else 1.0
    fam = [wname(w) for w in WINDOWS if wname(w) in p]
    hp = holm({c: p[c] for c in fam})
    return {"n_windows": len(common), "n_a": len(ga), "n_b": len(gb), "windows": cols,
            "key": key, "diff": {c: {"d": float(pa[j] - pb[j]), "ci90": ci(d[:, j]), "ci95": ci(d[:, j], (.025, .975)),
                                     "p_boot": p[c], "p_holm": hp.get(c)}
                     for j, c in enumerate(cols)}}


def main(argv=None) -> int:
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", action="append", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "afterimage"))
    ap.add_argument("--ref", default=str(ROOT / "results" / "v2" / "conf" / "numbers.json"))
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    N_BOOT = a.n_boot
    rows, excl = load([Path(p) for p in (a.runs_dir or [str(ROOT / "runs_cluster")])])
    errs = gate(rows, Path(a.ref))
    if errs:
        print("GATE FAILED: afterimage.py does not reproduce results/v2/conf/numbers.json")
        print("\n".join(errs))
        return 2
    print(f"GATE PASSED: beta_band and carry_slope_41_60 reproduce to 9 significant digits for "
          f"{len(json.loads(Path(a.ref).read_text())['H1'])} conf stages")
    if a.gate_only:
        return 0
    rng = np.random.default_rng(SEED)
    out = Path(a.out)
    x0_pass = False
    for exp in sorted({r["exp"] for r in rows}, key=lambda e: (e != "x0", e)):
        res = {"exp": exp, "n_boot": N_BOOT, "excluded": {k: v for k, v in excl.items() if exp_of(k.split("|")[0]) == exp},
               "stages": {}, "contrasts": {}}
        for st in sorted({r["stage"] for r in rows if r["exp"] == exp}):
            g = [r for r in rows if r["stage"] == st and r["x"] in BAND]
            if not g:
                continue
            draws = cluster_draws([r["window_id"] for r in g], rng) if all(r["window_id"] is not None for r in g) else None
            res["stages"][st] = stage_block(g, rng, draws)
        for e, name, sa, sb, key in CONTRASTS:
            if e != exp:
                continue
            ga = [r for r in rows if r["stage"] == sa and r["x"] in BAND]
            gb = [r for r in rows if r["stage"] == sb and r["x"] in BAND]
            if ga and gb:
                res["contrasts"][name] = {"a": sa, "b": sb, "holm_family": f"{name}: its 20-round windows",
                                          **contrast(ga, gb, rng, key)}
        if exp == "x0":
            res["x0_verdict"] = x0_verdict(res["stages"], out / "x0" / "mc.json")
            x0_pass = res["x0_verdict"]["pass"]
        elif x0_pass:
            for e in res["stages"].values():
                if e["A"]:
                    e["half_life"] = fit_half_life(e)
        if exp == "d2":
            res["d2_extras"] = d2_extras([Path(p) for p in (a.runs_dir or [str(ROOT / "runs_cluster")])])
        res["rules"] = rules(exp, res)
        (out / exp).mkdir(parents=True, exist_ok=True)
        (out / exp / "numbers.json").write_text(json.dumps(res, indent=1, sort_keys=True))
        print(f"{exp}: {len(res['stages'])} stages -> {out / exp / 'numbers.json'}")
    return 0


def _ci(res: dict, st: str, w: str, key: str = "p") -> tuple | None:
    """(A, LB90, UB90) of one stage and window, within-level bootstrap (the CI X0 validated)."""
    e = res["stages"].get(st, {}).get("A", {}).get(key, {}).get(w)
    return (e["A"], *e["ci90_within"]) if e else None


def _band(lo, hi, eps=0.15) -> bool:
    return lo is not None and -eps < lo and hi < eps


def rules(exp: str, res: dict) -> dict:
    """The pre-registered labels (docs/PREREG_AFTERIMAGE.md, tag prereg-afterimage-D1). Single-stage bounds use the
    within-level 90% CI; contrasts use the window-paired cluster 90% CI. Nothing here may change after any D1 data."""
    L, C, W, E = {}, res["contrasts"], "121-170", "41-60"
    if exp == "e5" and (a := _ci(res, "ait-e5-full-qwen3off", W)) and (b := _ci(res, "ait-e5-full-qwen3off", E)):
        L["E5_confirmed"] = bool(a[1] > 0.5 and 0.75 <= a[0] <= 0.93 and 0.92 <= b[0] <= 1.00)
        L["E5_inputs"] = {"A_121_170": a, "A_41_60": b, "PI_121_170": [0.75, 0.93], "PI_41_60": [0.92, 1.00]}
    if exp == "e5" and "D_PO" in C and W in C["D_PO"].get("diff", {}):
        lo, hi = C["D_PO"]["diff"][W]["ci90"]
        L["D_PO"] = ("price columns suffice" if _band(lo, hi) else "share and profit carry part" if hi < -0.15
                     else "hiding profit lengthens it" if lo > 0.15 else "unresolved")
    if exp == "e1":
        if a := _ci(res, "ait-e1-br-qwen3off", W, "p1"):
            L["E1_BR"] = "loop-carried" if a[2] < 0.2 else "single-agent carry" if a[1] > 0.5 else "partial"
            if (s := C.get("D_BR", {}).get("diff", {}).get(W)) is not None:
                L["E1_BR_share_of_self"] = a[0] / (a[0] - s["d"]) if a[0] - s["d"] else None
        if (s := C.get("D_COPY", {}).get("diff", {}).get(W)) is not None:
            L["E1_COPY"] = "preserved" if _band(*s["ci90"]) else "not preserved"
        ps = {k: C[k]["diff"][W]["p_boot"] for k in ("D_BR", "D_COPY") if W in C.get(k, {}).get("diff", {})}
        L["E1_holm_121_170"] = holm(ps) if ps else {}
    if exp == "e1b" and (a := _ci(res, "ait-e1b-nash-qwen3off", W, "p1")):
        L["E1b"] = "own-column carry" if a[1] > 0.5 else "none" if a[2] < 0.2 else "partial"
    if exp == "e4":
        ps = {}
        for st in res["stages"]:
            a, b = _ci(res, st, W), _ci(res, st, E)
            if not (a and b):
                continue
            L[st] = ("relays" if a[1] > 0.5 else "lets go" if b[1] > 0.5 and a[2] < 0.5 else "none" if b[2] < 0.2
                     else "unresolved")
            ps[st] = res["stages"][st]["A"]["p"][W].get("p_le_half")
        L["E4_holm_relays"] = holm({k: v for k, v in ps.items() if v is not None})
    if exp == "d2" and (a := _ci(res, "ait-d2-hint-qwen3off", W)) and (s := C.get("D_hint", {}).get("diff", {}).get(W)):
        fn = res.get("d2_extras", {}).get("fn_gap_q1.473")
        L["D2_inputs"] = {"A_hint_121_170": a, "D_hint_ci90": s["ci90"], "fn_gap_q1.473": fn}
        L["D2_WORKS"] = bool(a[2] < 0.2 and s["ci90"][1] < -0.15 and fn is not None and fn <= 0.05)  # + the run gates
    if exp == "e2b" and (s := C.get("D_high", {}).get("diff", {}).get(W)) is not None:
        L["E2b"] = "reasoning shortens" if s["ci90"][1] < -0.15 else "no shortening shown"
    if exp == "e2b" and (s := C.get("D_high8k", {}).get("diff", {}).get(W)) is not None:   # D3, post-data amendment
        n_ok = res["stages"].get("ait-e2b-luna-high8k", {}).get("n", 0)
        n_ex = sum(v for k, v in res["excluded"].items() if k.startswith("ait-e2b-luna-high8k|"))
        L["E2b_amend_8k"] = ("not scored (G1)" if n_ex > 0.10 * (n_ok + n_ex) else
                             "reasoning shortens" if s["ci90"][1] < -0.15 else "no shortening shown")
    return L


def d2_extras(runs_dirs: list[Path]) -> dict:
    """Lever-1 descriptives (PREREG_AFTERIMAGE D2): FN gap under the hint at q = 1.473 (mean |p1 - BR(q)| over the
    last 50 rounds of the Q-hint runs), hint uptake (share of firm decisions within 0.01 of the hinted b, rounds > 20),
    and the late level per plant for hint and sham."""
    from collusion.market import Market  # noqa: PLC0415
    m, out, gaps, up = Market(), {}, defaultdict(list), defaultdict(list)
    rx = re.compile(r"highest at a price of (\d\.\d{3})\.")
    for rd in runs_dirs:
        for d in sorted(rd.glob("ait-d2-*")):
            mf = d / "manifest.json"
            if not mf.exists() or not ok(d):
                continue
            c = json.loads(mf.read_text())["config"]
            R = read_jsonl(d / "rounds.jsonl")
            if c["arm"] == "Q":
                q = c["fixed_price"]
                gaps[q].append(float(np.mean([abs(r["p1"] - m.best_response(q)) for r in R[-50:]])))
            elif c["stage"] == "ait-d2-hint-qwen3off":
                for x in read_jsonl(d / "calls.jsonl"):
                    mh = rx.search(x.get("user_prompt") or "")
                    if mh and x.get("parsed_price") is not None and x.get("round", 0) > 20:
                        up[c["opening"][0]].append(abs(x["parsed_price"] - float(mh.group(1))) <= 0.01)
    for q, g in gaps.items():
        out[f"fn_gap_q{q:.3f}"] = float(np.mean(g))
    out["uptake"] = {f"{x:.3f}": float(np.mean(v)) for x, v in sorted(up.items())}
    return out


def x0_verdict(stages: dict, mc_path: Path) -> dict:
    """Empirical half of the X0 rule (docs/AFTERIMAGE.md), on the mock runs: |A - A_q| <= 0.01 per window at sigma 0
    and <= 0.05 at sigma > 0 (A_q from x0_mc.py); averager within 0.05 of 1 and br within 0.05 of 0 from 41-60 on.
    X0 passes only if this AND x0_mc.py (data path, coverage) pass."""
    if not mc_path.exists():
        return {"pass": False, "reason": f"{mc_path} missing: run analysis/x0_mc.py first"}
    mc = json.loads(mc_path.read_text())
    v = {"per_stage": {}, "mc_pass": mc["pass"]}
    for st, e in stages.items():
        A = e["A"].get("p", {})
        m = re.match(r"^x0(long)?-h(\d+)-sd([\d.]+)$", st)
        if m:
            key = f"h{m.group(2)}{'-long' if m.group(1) else ''}-sd{float(m.group(3)):g}"
            if key not in mc["cells"]:
                v["per_stage"][st] = {"pass": False, "reason": f"no MC cell {key}"}
                continue
            tgt, tol = mc["cells"][key]["A_q"], (0.01 if float(m.group(3)) == 0 else 0.05)
            dev = {c: abs(A[c]["A"] - tgt[c]) for c in A if c in tgt}
        elif st in ("x0-averager-sd0", "x0-br-sd0"):
            t0, tol = (1.0 if "averager" in st else 0.0), 0.05
            dev = {c: abs(A[c]["A"] - t0) for c in A if c in [wname(w) for w in WINDOWS]}
        else:
            continue
        v["per_stage"][st] = {"tol": tol, "max_dev": max(dev.values()) if dev else None,
                              "pass": bool(dev) and max(dev.values()) <= tol}
    v["pass"] = bool(v["per_stage"]) and mc["pass"] and all(s["pass"] for s in v["per_stage"].values())
    return v


def fit_half_life(e: dict) -> dict:
    """lambda-hat from log A on bin centre over 10-round bins with A > 0.05; h = ln 0.5 / ln lambda; D_0.5 = first
    bin whose A < 0.5."""
    A = e["A"]["p"]
    pts = [((int(c.split("-")[0]) + int(c.split("-")[1])) / 2 - 20, A[c]["A"]) for c in A if c in [wname(w) for w in BINS]]
    use = [(t, v) for t, v in pts if v > 0.05]
    if len(use) < 3:
        return {"lambda": None, "h": None, "D_0.5": None}
    s = slope([t for t, _ in use], [math.log(v) for _, v in use])
    lam = math.exp(s)
    d05 = next((c for c in A if c in [wname(w) for w in BINS] and A[c]["A"] < 0.5), None)
    return {"lambda": lam, "h": (math.log(0.5) / math.log(lam)) if lam < 1 else None, "D_0.5": d05}


if __name__ == "__main__":
    raise SystemExit(main())
