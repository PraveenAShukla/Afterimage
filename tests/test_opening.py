"""Arms S/A (injected openings): window construction, config rules, run identity (E1) and validation (E2)."""
import json
import sys
from pathlib import Path

import pytest
import yaml

from collusion.config import validate
from collusion.opening import shifted_window, source_order
from collusion.run import main
from collusion.runlog import read_jsonl
from collusion.validate import validate_run

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from analyze import design_key  # noqa: E402

SRC = [{"t": t, "p1": 1.60 + 0.01 * (t % 3), "p2": 1.62 - 0.01 * (t % 2)} for t in range(1, 301)]


def _mean(xs):
    return sum(xs) / len(xs)


def test_symmetric_window_mean_and_shape():
    rows, info = shifted_window(SRC, 1.90, 1.90)
    assert [r["t"] for r in rows] == list(range(1, 201))
    last = rows[-20:]
    assert _mean([r["p1"] for r in last] + [r["p2"] for r in last]) == pytest.approx(1.90)
    # ONE delta for both firms: the within-window gap between firms is preserved exactly
    src = {r["t"]: r for r in SRC}
    for r in last:
        assert r["p1"] - r["p2"] == pytest.approx(src[r["t"]]["p1"] - src[r["t"]]["p2"])
    assert info["delta"][0] == info["delta"][1] and info["n_clipped"] == 0
    # rows 1-180 repeat the visible window
    assert all(rows[i]["p1"] == rows[180 + i % 20]["p1"] for i in range(180))


def test_asymmetric_window_per_firm_means():
    rows, info = shifted_window(SRC, 1.40, 1.90)
    last = rows[-20:]
    assert _mean([r["p1"] for r in last]) == pytest.approx(1.40)
    assert _mean([r["p2"] for r in last]) == pytest.approx(1.90)
    assert info["delta"][0] != info["delta"][1]


def test_clipping_is_recorded():
    rows, info = shifted_window(SRC, 2.50, 2.50)  # window spread straddles the 2.5 bound
    assert info["n_clipped"] > 0 and max(r["p1"] for r in rows) <= 2.5


def test_source_order_is_a_recorded_permutation():
    pool = [5, 6, 7, 8]
    assert sorted(source_order(pool, 1)) == pool
    assert source_order(pool, 1) == source_order(pool, 1)
    with pytest.raises(ValueError):
        source_order([5, 5], 1)


BASE = {"stage": "t", "prompt": "E1", "rounds": 300, "window": 100, "seeds": [1],
        "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 64},
        "firm1": {"provider": "mock", "model": "mock", "strategy": "collude"},
        "firm2": {"provider": "mock", "model": "mock", "strategy": "collude"},
        "opening_source": {"template": "t_L_E1_mock_s{seed}", "pool": [11, 12], "perm_seed": 3}}


@pytest.mark.parametrize("arm, opening, ok", [
    ("S", [1.7, 1.7], True), ("S", [1.6, 1.9], False), ("A", [1.6, 1.9], True), ("A", [1.7, 1.7], False),
    ("S", [0.5, 0.5], False), ("S", "1.7", False),
])
def test_config_rules(arm, opening, ok):
    errs = validate({**BASE, "arm": arm, "opening": opening})
    assert (not errs) == ok, errs


def test_config_needs_source_and_prefix_multiple_of_20():
    assert validate({**{k: v for k, v in BASE.items() if k != "opening_source"}, "arm": "S", "opening": [1.7, 1.7]})
    assert validate({**BASE, "arm": "S", "opening": [1.7, 1.7], "window": 95})
    assert validate({**BASE, "arm": "L", "opening": [1.7, 1.7]})  # opening outside S/A is an error


def _write(tmp_path, name, **kw):
    c = {**BASE, **kw}
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(c))
    return str(p)


def test_two_levels_same_seed_are_two_runs_and_two_design_keys(tmp_path):
    """E1: the injected opening must be in the run_id and the design key, else level 2 is skipped as complete."""
    runs = tmp_path / "runs"
    Lcfg = {k: v for k, v in BASE.items() if k not in ("opening_source", "window")}
    p = tmp_path / "L2.yaml"
    p.write_text(yaml.safe_dump({**Lcfg, "arm": "L", "rounds": 200, "seeds": [11, 12]}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    for x in (1.30, 2.10):
        assert main([_write(tmp_path, f"S{x}", arm="S", opening=[x, x], seeds=[1, 2]), "--runs-dir", str(runs)]) == 0
    dirs = sorted(d.name for d in runs.glob("t_S_*"))
    assert len(dirs) == 4, dirs  # 2 levels x 2 seeds, none skipped
    keys = set()
    for d in runs.glob("t_S_*"):
        man = json.loads((d / "manifest.json").read_text())
        keys.add(design_key(man["config"]))
        assert validate_run(d) == [], d.name
        R = read_jsonl(d / "rounds.jsonl")
        assert [r["t"] for r in R] == list(range(1, 301))
        assert all(r["is_prefix"] for r in R[:200]) and not any(r["is_prefix"] for r in R[200:])
        x = man["config"]["opening"][0]
        win = R[180:200]
        assert _mean([r["p1"] for r in win] + [r["p2"] for r in win]) == pytest.approx(x, abs=1e-9)
        assert man["meta"]["opening_source_run"] in ("t_L_E1_mock_s11", "t_L_E1_mock_s12")
    assert len(keys) == 2


def test_validator_catches_a_tampered_prefix(tmp_path):
    runs = tmp_path / "runs"
    Lcfg = {k: v for k, v in BASE.items() if k not in ("opening_source", "window")}
    p = tmp_path / "L.yaml"
    p.write_text(yaml.safe_dump({**Lcfg, "arm": "L", "rounds": 200, "seeds": [11, 12]}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    assert main([_write(tmp_path, "S", arm="S", opening=[1.9, 1.9]), "--runs-dir", str(runs)]) == 0
    d = next(runs.glob("t_S_*"))
    # tamper the recorded opening in the manifest: the rebuilt prefix no longer matches the played one
    man = json.loads((d / "manifest.json").read_text())
    man["meta"]["opening"] = [1.5, 1.5]
    (d / "manifest.json").write_text(json.dumps(man))
    assert any("opening" in e for e in validate_run(d))


def test_design_key_separates_firm2_models():
    c = {"stage": "s", "prompt": "E1", "sampling": {"temperature": 0.7, "top_p": 1.0, "max_tokens": 9},
         "firm1": {"model": "a/A"}, "firm2": {"model": "b/B"}}
    c2 = {**c, "firm2": {"model": "c/C"}}
    assert design_key(c) != design_key(c2)
    same = {**c, "firm2": {"model": "a/A"}}
    assert "|f2:" not in design_key(same)  # self-play keys are unchanged from before


def _controls_setup(tmp_path):
    runs = tmp_path / "runs"
    Lcfg = {k: v for k, v in BASE.items() if k not in ("opening_source", "window")}
    p = tmp_path / "L.yaml"
    p.write_text(yaml.safe_dump({**Lcfg, "arm": "L", "rounds": 200, "seeds": [11, 12]}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    return runs


@pytest.mark.parametrize("strategy, x, want", [
    ("averager", 1.30, 1.30), ("averager", 2.10, 2.10),   # averager: the opening persists (slope 1)
    ("br", 1.30, 1.4729), ("br", 2.10, 1.4729),           # static BR: Nash whatever the opening (slope 0)
    ("grim", 2.10, 1.80), ("grim", 1.30, 1.4729),         # grim trigger at 1.80: a step in the opening
])
def test_scripted_controls_through_the_pipeline(tmp_path, strategy, x, want):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": strategy}
    assert main([_write(tmp_path, "S", arm="S", opening=[x, x], firm1=f, firm2=f), "--runs-dir", str(runs)]) == 0
    d = next(runs.glob("t_S_*"))
    assert validate_run(d) == []
    R = read_jsonl(d / "rounds.jsonl")
    final = _mean([r["p1"] for r in R[260:]] + [r["p2"] for r in R[260:]])
    assert final == pytest.approx(want, abs=0.01)


def test_hillclimb_control_runs_and_validates(tmp_path):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "hillclimb"}
    assert main([_write(tmp_path, "S", arm="S", opening=[1.7, 1.7], firm1=f, firm2=f), "--runs-dir", str(runs)]) == 0
    assert validate_run(next(runs.glob("t_S_*"))) == []


def test_cross_model_source_requires_the_flag(tmp_path):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "other", "strategy": "averager"}
    with pytest.raises(RuntimeError, match="model differs"):
        main([_write(tmp_path, "S", arm="S", opening=[1.7, 1.7], firm1=f, firm2=f), "--runs-dir", str(runs)])
    src = {**BASE["opening_source"], "cross_model": True}
    assert main([_write(tmp_path, "S2", arm="S", opening=[1.7, 1.7], firm1=f, firm2=f, opening_source=src),
                 "--runs-dir", str(runs)]) == 0


def test_onestep_map_on_controls(tmp_path):
    """E4: a one-step map of the averager returns the window level (own=rival=x); of BR, BR(last rival)."""
    from collusion.market import Market
    from collusion.onestep import main as onestep
    runs = _controls_setup(tmp_path)
    for strat in ("averager", "br"):
        c = {"stage": f"m{strat}", "prompt": "E1", "sampling": BASE["sampling"], "mode": "diag",
             "firm1": {"provider": "mock", "model": "mock", "strategy": strat}, "levels": [1.3, 2.1], "samples": 3,
             "seed_base": 5, "opening_source": BASE["opening_source"]}
        p = tmp_path / f"{strat}.yaml"
        p.write_text(yaml.safe_dump(c))
        assert onestep([str(p), "--runs-dir", str(runs), "--workers", "4"]) == 0
        S = read_jsonl(runs / f"m{strat}_onestep_diag_mock" / "samples.jsonl")
        assert len(S) == 6 and all(s["status"] == "ok" for s in S)
        for s in S:
            want = (s["last_own"] + s["last_rival"]) / 2 if strat == "averager" else Market().best_response(s["last_rival"])
            assert s["price"] == pytest.approx(want, abs=1e-3)
            assert (s["win_mean_own"] + s["win_mean_rival"]) / 2 == pytest.approx(s["own"], abs=1e-9)
        assert len(read_jsonl(runs / f"m{strat}_onestep_diag_mock" / "calls.jsonl")) == 6


# ---------------- plan v2: arms O / T / Q, phase, window id, views ----------------
V2 = {**BASE, "rounds": 170, "window": 150, "phase": "disc"}


def test_arm_O_twenty_row_opening_and_continuation(tmp_path):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "O.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "O", "opening": [1.95, 1.95], "seeds": [1, 2], "firm1": f, "firm2": f}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    ds = sorted(runs.glob("t_O_E1_mock_disc_x1.950-1.950_w*_s*"))
    assert len(ds) == 2, [d.name for d in runs.iterdir()]
    for d in ds:
        assert validate_run(d) == []
        R = read_jsonl(d / "rounds.jsonl")
        assert [r["t"] for r in R] == list(range(1, 171)) and sum(r["is_prefix"] for r in R) == 20
        man = json.loads((d / "manifest.json").read_text())
        meta = man["meta"]
        assert len(meta["prefix_sha256"]) == 64 and "window_slope" in meta and len(meta["realised_last5"]) == 2
        assert f"_w{meta['window_id']}_" in d.name
        u = read_jsonl(d / "calls.jsonl")[0]["user_prompt"]
        assert u.startswith("Round 21. ") and "\n1, " in u and "\n20, " in u  # rounds 1-20 shown, live from 21


def test_arm_T_transplants_the_source_opening(tmp_path):
    runs = _controls_setup(tmp_path)
    p = tmp_path / "T.yaml"
    p.write_text(yaml.safe_dump({**V2, "arm": "T", "seeds": [7]}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    d = next(runs.glob("t_T_E1_mock_disc_w*_s7"))
    assert validate_run(d) == []
    man = json.loads((d / "manifest.json").read_text())
    src = {r["t"]: r for r in read_jsonl(runs / man["meta"]["transplant_source_run"] / "rounds.jsonl")}
    R = read_jsonl(d / "rounds.jsonl")
    assert all(R[t - 1]["p1"] == src[t]["p1"] and R[t - 1]["p2"] == src[t]["p2"] for t in range(1, 21))


def test_arm_Q_fixed_rival_ladder_and_views(tmp_path):
    runs = tmp_path / "runs"
    f = {"provider": "mock", "model": "mock", "strategy": "br"}
    for q, view in ((1.70, None), (1.70, {"label": True}), (1.70, {"anchor": 2.10})):
        c = {"stage": "t", "arm": "Q", "prompt": "E1", "rounds": 30, "fixed_price": q, "phase": "disc", "seeds": [3],
             "sampling": BASE["sampling"], "firm1": f}
        if view:
            c["view"] = view
        p = tmp_path / f"Q{len(list(tmp_path.glob('Q*.yaml')))}.yaml"
        p.write_text(yaml.safe_dump(c))
        assert main([str(p), "--runs-dir", str(runs)]) == 0
    ds = sorted(runs.glob("t_Q_*"))
    assert [d.name for d in ds] == ["t_Q_E1_mock_disc_q1.700_s3", "t_Q_E1_mock_disc_q1.700_vanchor2.100_s3",
                                    "t_Q_E1_mock_disc_q1.700_vlabel_s3"]
    from collusion.market import Market
    for d in ds:
        assert validate_run(d) == [], d.name
        R = read_jsonl(d / "rounds.jsonl")
        assert all(r["p2"] == 1.70 for r in R) and R[-1]["p1"] == pytest.approx(Market().best_response(1.70), abs=1e-3)
    u = {d.name: read_jsonl(d / "calls.jsonl")[3]["user_prompt"] for d in ds}
    assert "unrelated product" in u["t_Q_E1_mock_disc_q1.700_vlabel_s3"]
    assert u["t_Q_E1_mock_disc_q1.700_vanchor2.100_s3"].splitlines()[1].endswith(", 2.100")
    assert "competitor's price" in u["t_Q_E1_mock_disc_q1.700_s3"] and "reference" not in u["t_Q_E1_mock_disc_q1.700_s3"]


def test_disc_and_conf_are_different_runs_and_design_keys(tmp_path):
    runs = _controls_setup(tmp_path)
    keys = set()
    for ph in ("disc", "conf"):
        p = tmp_path / f"{ph}.yaml"
        p.write_text(yaml.safe_dump({**V2, "arm": "O", "phase": ph, "opening": [1.75, 1.75], "seeds": [1]}))
        assert main([str(p), "--runs-dir", str(runs)]) == 0
    for d in runs.glob("t_O_*"):
        keys.add(design_key(json.loads((d / "manifest.json").read_text())["config"]))
    assert len(list(runs.glob("t_O_*"))) == 2 and len(keys) == 2


def test_v2_configs_require_phase_and_valid_views():
    assert validate({**V2, "arm": "O", "opening": [1.7, 1.7], "phase": "x"})
    assert validate({k: v for k, v in {**V2, "arm": "O", "opening": [1.7, 1.7]}.items() if k != "phase"})
    assert validate({"stage": "t", "arm": "Q", "prompt": "E1", "rounds": 30, "fixed_price": 3.0, "phase": "disc",
                     "seeds": [1], "sampling": BASE["sampling"], "firm1": BASE["firm1"]})
    assert validate({"stage": "t", "arm": "Q", "prompt": "E1", "rounds": 30, "fixed_price": 1.7, "phase": "disc",
                     "seeds": [1], "sampling": BASE["sampling"], "firm1": BASE["firm1"], "view": {"anchor": 9}})


def test_v2_disc_and_conf_seed_sets_do_not_overlap():
    """Plan v2 M1b: discovery and confirmation never share a seed or a window permutation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mk", Path(__file__).resolve().parents[1] / "scripts/make_v2_configs.py")
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    seeds = {ph: {s for c in mk.configs(ph).values() for s in c["seeds"]} for ph in ("disc", "conf")}
    assert seeds["disc"] and seeds["conf"] and not (seeds["disc"] & seeds["conf"])
    assert mk.PHASE["disc"]["perm"] != mk.PHASE["conf"]["perm"]
    assert 61860 not in mk.POOL and 210629 not in mk.POOL and len(mk.POOL) == 18


# ---------------- plan v2 addendum: mechanism arms STEP / IMP, E1-PO ----------------
MECH = {**V2, "opening": [1.70, 1.70], "phase": "conf", "seeds": [1, 2]}


def test_step_arm_scripted_rival_and_validation(tmp_path):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "cutmatch"}
    for d in (-0.15, 0.0, 0.15):
        p = tmp_path / f"STEP{d}.yaml"
        p.write_text(yaml.safe_dump({**{k: v for k, v in MECH.items() if k != "firm2"}, "arm": "STEP", "firm1": f,
                                     "step": {"delta": d, "round": 41}}))
        assert main([str(p), "--runs-dir", str(runs)]) == 0
    ds = sorted(runs.glob("t_STEP_*"))
    assert len(ds) == 6
    for d in ds:
        assert validate_run(d) == [], d.name
        R = read_jsonl(d / "rounds.jsonl")
        dl = json.loads((d / "manifest.json").read_text())["config"]["step"]["delta"]
        assert all(r["p2"] == pytest.approx(1.70) for r in R[20:40]) and R[40]["p2"] == pytest.approx(1.70 + dl)
        assert R[-1]["p1"] == pytest.approx(1.70 + min(dl, 0.0), abs=1e-6)  # cutmatch follows the cut only


def test_imp_arm_runtime_impulse_and_validation(tmp_path):
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "cutmatch"}
    p = tmp_path / "IMP.yaml"
    p.write_text(yaml.safe_dump({**MECH, "arm": "IMP", "firm1": f, "firm2": f, "impulse": {"delta": -0.15, "round": 41}}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    d = next(runs.glob("t_IMP_*"))
    assert validate_run(d) == []
    R = read_jsonl(d / "rounds.jsonl")
    assert R[40]["source1"] == "impulse" and R[40]["p1"] == pytest.approx(R[39]["p1"] - 0.15)
    assert R[41]["p2"] == pytest.approx(R[40]["p1"])  # the rival matches the cut


def test_price_only_view_hides_share_and_profit(tmp_path):
    from collusion.prompts import user_message
    h = [{"round": 1, "own_price": 1.7, "rival_price": 1.6, "own_share": 0.3, "own_profit": 0.2}]
    full, po = user_message(2, h), user_message(2, h, {"price_only": True})
    assert full == user_message(2, h, None) and "your quantity share" in full
    assert "share" not in po and po.splitlines()[1] == "1, 1.700, 1.600"
    runs = _controls_setup(tmp_path)
    f = {"provider": "mock", "model": "mock", "strategy": "averager"}
    p = tmp_path / "A_PO.yaml"
    p.write_text(yaml.safe_dump({**MECH, "arm": "A", "opening": [1.60, 2.00], "firm1": f, "firm2": f,
                                 "view": {"price_only": True}}))
    assert main([str(p), "--runs-dir", str(runs)]) == 0
    d = next(runs.glob("t_A_*vPO*"))
    assert validate_run(d) == []


def test_mechanism_control_policies():
    import random as _r
    from collusion.backends.mock import MockBackend
    down = "Round 42. H\n40, 1.700, 1.700, 0.300, 0.200\n41, 1.700, 1.550, 0.250, 0.150"
    up = "Round 42. H\n40, 1.700, 1.700, 0.300, 0.200\n41, 1.700, 1.850, 0.350, 0.250"
    P = lambda s, u: MockBackend("m", s)._price(u, _r.Random(1))  # noqa: E731
    assert (P("cutmatch", down), P("cutmatch", up)) == (1.55, 1.70)
    assert (P("raisematch", down), P("raisematch", up)) == (1.70, 1.85)
    assert (P("winstay", down), P("winstay", up)) == (1.55, 1.70)
    assert P("satisfice:0.025", up) == 1.70 and P("satisfice:0.025", down) == pytest.approx(1.5059, abs=1e-3)
    assert P("step:1.70:41:0.15", "Round 41. x") == pytest.approx(1.85) and P("step:1.70:41:0.15", "Round 40. x") == 1.70
