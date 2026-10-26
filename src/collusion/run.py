"""CLI: expand a YAML config into games and run them in parallel (vLLM batches concurrent requests).

  python -m collusion.run configs/stage0_smoke.yaml --dry-run   # print jobs, first prompt, cost estimate; no calls
  python -m collusion.run configs/stage0_smoke.yaml             # run

Completed run_ids (summary.json present) are skipped, so re-invoking a config resumes a stage.
A run_id whose directory exists WITHOUT summary.json (crashed) is reported and left untouched; re-run it
with --tag to get a fresh id. Nothing is ever overwritten.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


from .agent import Agent
from .arms import Fixed, Impulse, Live, Policy, Replay, Scripted, deviation_overrides
from .backends import Sampling, make_backend
from .budget import Budget
from .config import ConfigError, load_config, validate
from .env import load_env
from .game import play
from .market import Market
from .opening import prefix_fingerprint, shifted_window, source_order, transplant_prefix
from .prompts import Identity, system_prompt, user_message
from .supervisor import REVIEW_TEMPERATURE, Supervisor, YokedSupervisor
from .runlog import REPO, RunLog, load_run, read_jsonl
from .seeds import load_assignment
from .validate import write_validation


@dataclass
class Job:
    run_id: str
    seed: int
    p1: Policy
    p2: Policy
    prefix: list[dict] | None
    meta: dict


def _agent(cfg: dict, firm: int) -> Agent:
    spec = cfg[f"firm{firm}"]
    ident = cfg.get(f"identity{firm}")
    return Agent(backend=make_backend(spec), prompt_variant=cfg["prompt"], sampling=Sampling(**cfg["sampling"]),
                 identity=Identity(**ident) if ident else None, view=cfg.get("view"),
                 cost=Market(**cfg.get("market", {})).cost(firm), supervision=(cfg.get("supervision") or {}).get("frame"),
                 pressure=pressure_for(cfg, firm))


def pressure_for(cfg: dict, firm: int) -> str | None:
    """The pressure line this firm gets: target both = both firms, firm1 = firm 1 only (firm 2 is the spec prompt)."""
    pr = cfg.get("pressure")
    return pr["frame"] if pr and (pr["target"] == "both" or firm == 1) else None


def _supervisor(cfg: dict, runs_dir: Path | None = None, seed: int | None = None):
    """The supervisor for a run. mode note (v1): free-text advice notes, firm 1's model and sampling. mode review
    (v2): label only, temperature REVIEW_TEMPERATURE. mode yoked (v2): no model; labels copied from the review run
    paired with this seed (supervision.yoke). None when there is no supervisor (claim-only arms)."""
    sv = cfg.get("supervision") or {}
    mode = sv.get("mode", "note")
    if mode == "yoked":
        src = sv["yoke"].get(seed, sv["yoke"].get(str(seed)))
        return YokedSupervisor.from_run(sv["frame"], Path(runs_dir) / src)
    if not sv.get("agent"):
        return None
    smp = dict(cfg["sampling"])
    if sv.get("max_tokens"):
        smp["max_tokens"] = sv["max_tokens"]
    if mode == "review":
        smp["temperature"] = REVIEW_TEMPERATURE
    return Supervisor(make_backend(sv.get("supervisor") or cfg["firm1"]), sv["frame"], Sampling(**smp), mode)


def _short(model: str) -> str:
    return model.split("/")[-1].replace(".", "")[:24]


def build_jobs(cfg: dict, runs_dir: Path, tag: str = "") -> list[Job]:
    m = Market(**cfg.get("market", {}))
    arm, b = cfg["arm"], m.benchmarks()
    name = f"{cfg['stage']}_{arm}_{cfg['prompt']}_{_short(cfg['firm1']['model'])}"
    if cfg.get("identity1"):
        name += "_id-" + cfg["identity_label"]
    # Everything that distinguishes a design cell is in the run_id, so a reused seed is never "already complete"
    # in another cell (plan v2 M1): phase, opening level, fixed rival price, prompt view; window id per job below.
    if cfg.get("phase"):
        name += f"_{cfg['phase']}"
    if arm in ("S", "A", "O", "STEP", "IMP"):
        x1, x2 = cfg["opening"]
        name += f"_x{x1:.3f}-{x2:.3f}"
    if arm == "STEP":
        name += f"_step{cfg['step']['delta']:+.3f}r{cfg['step']['round']}"
    if arm == "IMP":
        im = cfg["impulse"]
        name += f"_imp{im['delta']:+.3f}r{im['round']}" + (f"k{im['hold']}" if im.get("hold", 1) != 1 else "")
    if arm == "Q":
        name += f"_q{cfg['fixed_price']:.3f}"
    pr = cfg.get("pressure")
    if pr:  # prG/prL = gain / loss pressure; 2 = both firms, 1 = firm 1 only
        name += f"_pr{pr['frame'][0].upper()}{'2' if pr['target'] == 'both' else '1'}"
    sv = cfg.get("supervision")
    if sv:  # supS/supI = shared / individual monitoring line; A = real supervisor agent, N = none
        letter = {"review": "R", "yoked": "Y"}.get(sv.get("mode", "note"), "A" if sv["agent"] else "N")
        name += f"_sup{sv['frame'][0].upper()}{letter}"  # A/N v1 notes; R review, Y yoked (v2)
    v = cfg.get("view") or {}
    if v.get("label"):
        name += "_vlabel"
    if v.get("price_only"):
        name += "_vPO"
    if "anchor" in v:
        name += f"_vanchor{v['anchor']:.3f}"
    if v.get("line"):
        name += {"br_hint": "_vBRH", "sham": "_vSHM"}[v["line"]]
    seeds = _seeds(cfg)
    prefixed = arm in ("S", "A", "O", "T", "STEP", "IMP")
    order = source_order(cfg["opening_source"]["pool"], cfg["opening_source"]["perm_seed"]) if prefixed else None
    jobs = []
    for i, seed in enumerate(seeds):
        rid = f"{name}_s{seed}{tag}"
        if prefixed and (arm in ("O", "T", "STEP", "IMP") or cfg.get("phase")):  # window id (v2 arms; v1 S/A ids unchanged)
            rid = f"{name}_w{order[i % len(order)]}_s{seed}{tag}"
        meta = {"arm": arm, "index": i}
        if arm in ("L", "X"):
            jobs.append(Job(rid, seed, Live(_agent(cfg, 1)), Live(_agent(cfg, 2)), None, meta))
        elif arm in ("FN", "FM"):
            jobs.append(Job(rid, seed, Live(_agent(cfg, 1)), Fixed(b["pN"] if arm == "FN" else b["pM"]), None, meta))
        elif arm == "Q":
            jobs.append(Job(rid, seed, Live(_agent(cfg, 1)), Fixed(float(cfg["fixed_price"])), None, meta))
        elif arm == "P":
            src = _source_run(cfg, runs_dir, cfg["replay_map"][str(seed)])
            p2 = [r["p2"] for r in src["rounds"] if not r["is_prefix"]]
            if len(p2) < cfg["rounds"]:
                raise RuntimeError(f"source L run too short: {len(p2)} < {cfg['rounds']}")
            meta["replays"] = src["manifest"]["run_id"]
            jobs.append(Job(rid, seed, Live(_agent(cfg, 1)), Replay(p2), None, meta))
        elif arm in ("D", "Dfrozen", "Dplacebo"):
            src = _source_run(cfg, runs_dir, seed)  # D uses the L run's own seed: branch of that run
            k = cfg["shock_round"]
            prefix = [r for r in src["rounds"] if r["t"] < k]
            f1, f2 = deviation_overrides(m, prefix, k, frozen=(arm == "Dfrozen"), window=cfg.get("window", 10),
                                         placebo=(arm == "Dplacebo"))
            meta.update({"branch_of": src["manifest"]["run_id"], "forced1": f1, "forced2": f2})
            jobs.append(Job(rid, seed, Scripted(Live(_agent(cfg, 1)), f1),
                            Scripted(Live(_agent(cfg, 2)), f2), prefix, meta))
        elif prefixed:
            src_seed = order[i % len(order)]
            src = _source_run(cfg, runs_dir, src_seed, template=cfg["opening_source"]["template"])
            n_pre = cfg["rounds"] - cfg["window"]
            if arm == "T":  # transplant: the source run's own first n_pre rounds, as played
                prefix = transplant_prefix(src["rounds"], n_pre)
                meta.update({"transplant_source_run": src["manifest"]["run_id"]})
            else:
                x1, x2 = cfg["opening"]
                prefix, info = shifted_window(src["rounds"], x1, x2, n_prefix=n_pre)
                meta.update({"opening": [x1, x2], "opening_source_run": src["manifest"]["run_id"], "opening_info": info})
            meta.update({"opening_source_order": order, "window_id": src_seed, **prefix_fingerprint(prefix)})
            if arm == "STEP":  # firm 2 scripted: the opening level until `round`, then level + delta
                x = float(cfg["opening"][0])
                k, d = cfg["step"]["round"], float(cfg["step"]["delta"])
                ov = {t: x + d for t in range(k, cfg["rounds"] + 1)}
                meta.update({"forced2": {str(t): v for t, v in ov.items()}, "step_level": x})
                jobs.append(Job(rid, seed, Live(_agent(cfg, 1)), Scripted(Fixed(x), ov), prefix, meta))
            elif arm == "IMP":
                im = cfg["impulse"]
                p1 = Impulse(Live(_agent(cfg, 1)), im["round"], float(im["delta"]), im.get("hold", 1))
                jobs.append(Job(rid, seed, p1, Live(_agent(cfg, 2)), prefix, meta))
            else:
                jobs.append(Job(rid, seed, Live(_agent(cfg, 1)), Live(_agent(cfg, 2)), prefix, meta))
        else:
            raise ValueError(f"unknown arm {arm}")
    return jobs


def _seeds(cfg: dict) -> list[int]:
    if "seeds" in cfg:
        return list(cfg["seeds"])
    a = load_assignment(REPO / cfg["seed_assignment"])
    key = cfg["seeds_from"]  # "L" or "P"
    if key == "P":
        cfg["replay_map"] = a["P_replays_L"]
    return a[key][: cfg.get("n_runs", len(a[key]))]


def _source_run(cfg: dict, runs_dir: Path, seed: int, template: str | None = None) -> dict:
    rid = (template or cfg["source_run_template"]).format(seed=seed)
    run = load_run(runs_dir, rid)
    if run["summary"] is None or run["summary"].get("status") != "complete":
        raise RuntimeError(f"source run {rid} is not a complete run; refusing to build on it")
    src = run["manifest"]["config"]
    cross = cfg["arm"] in ("S", "A", "O", "T", "STEP", "IMP") and cfg["opening_source"].get("cross_model")
    for k in ("market",) if cross else ("prompt", "sampling", "market"):  # a cross-model window is price data only
        if src.get(k) != cfg.get(k):
            raise RuntimeError(f"source run {rid}: {k} differs ({src.get(k)!r} vs {cfg.get(k)!r})")
    firms = ("firm1", "firm2") if cfg["arm"] in ("D", "Dfrozen", "Dplacebo") else ("firm1",)
    if cfg["arm"] in ("S", "A", "O", "T", "STEP", "IMP") and cfg["opening_source"].get("cross_model"):
        firms = ()  # scripted controls fed another model's real windows: declared in the config, recorded in the manifest
    for f in firms:
        if (src[f]["provider"], src[f]["model"]) != (cfg[f]["provider"], cfg[f]["model"]):
            raise RuntimeError(f"source run {rid}: {f} model differs")
    if src["arm"] != "L":
        raise RuntimeError(f"source run {rid} is arm {src['arm']}, expected L")
    return run


def estimate(cfg: dict, n_jobs: int) -> dict:
    """Rough token budget: system ~80 tok, user ~40 + 25/row (<=20 rows); output assumed = cfg est_output_tokens."""
    rounds = cfg["rounds"] - (cfg.get("shock_round", 1) - 1 if cfg["arm"] in ("D", "Dfrozen", "Dplacebo") else 0)
    if cfg["arm"] in ("D", "Dfrozen", "Dplacebo"):
        rounds = min(rounds, cfg.get("window", 10))
    if cfg["arm"] in ("S", "A", "O", "T", "STEP", "IMP"):
        rounds = cfg["window"]
    live_firms = 2 if cfg["arm"] in ("L", "X", "D", "Dfrozen", "Dplacebo", "S", "A", "O", "T", "IMP") else 1
    calls = n_jobs * rounds * live_firms
    inp = calls * (80 + 40 + 25 * 20)
    out = calls * cfg.get("est_output_tokens", cfg["sampling"]["max_tokens"])
    return {"games": n_jobs, "calls_no_retries": calls, "input_tokens": inp, "output_tokens_est": out}


def _resume_state(d: Path) -> tuple[list[dict], dict]:
    """Completed rounds and the stats they imply, recounted from the logs (calls of the interrupted round excluded)."""
    rounds = read_jsonl(d / "rounds.jsonl")
    last = rounds[-1]["t"] if rounds else 0
    calls = [c for c in read_jsonl(d / "calls.jsonl") if c["round"] <= last]
    st = {"calls": len(calls), "parse_fail_calls": sum(c["parse_status"] != "ok" for c in calls),
          "truncated": sum(bool(c["truncated"]) for c in calls), "transport_errors": sum(bool(c["transport_error"]) for c in calls),
          "price_from_think": sum(bool(c.get("price_from_think")) for c in calls),
          "input_tokens": sum(c["input_tokens"] or 0 for c in calls), "output_tokens": sum(c["output_tokens"] or 0 for c in calls),
          "usd": sum(c["usd"] for c in calls),
          "decisions": sum(1 for r in rounds for f in "12" if r.get("source" + f) == "live"),
          "fallbacks": sum(1 for r in rounds for f in "12" if str(r.get("status" + f, "")).startswith("fallback")),
          "clipped": sum(1 for r in rounds for f in "12" if r.get("clipped" + f))}
    return rounds, st


def run_job(job: Job, cfg: dict, runs_dir: Path, budget: Budget, debug_calls: int = 0, resume: bool = False) -> dict:
    if resume:
        if job.prefix:
            raise RuntimeError(f"{job.run_id}: resume is not supported for branched (D) runs; rerun with --tag")
        d = runs_dir / job.run_id
        _assert_same_design(d, cfg)
        log = RunLog(runs_dir, job.run_id, resume=True)
        done, base = _resume_state(d)
        log.event("resume", from_round=(done[-1]["t"] if done else 0) + 1)
        log.debug(f"RESUME from round {(done[-1]['t'] if done else 0) + 1}", echo=True)
        m = Market(**cfg.get("market", {}))
        try:
            summary = play(log, m, job.p1, job.p2, cfg["rounds"], job.seed, cfg["arm"], budget, None,
                           strict=cfg.get("strict", True), health_cfg=cfg.get("health"), debug_calls=debug_calls,
                           resumed=done, base_stats=base, supervisor=_supervisor(cfg, runs_dir, job.seed))
        except Exception as e:
            log.event("crash", error=repr(e), traceback=traceback.format_exc())
            raise
        log.finish(summary)
        errs = write_validation(log.dir)
        summary["validation_errors"] = len(errs)
        return summary
    log = RunLog(runs_dir, job.run_id)
    m = Market(**cfg.get("market", {}))
    server = {}
    for f in (1, 2):
        spec = cfg.get(f"firm{f}")
        if spec and spec["provider"] == "vllm":
            server[f"firm{f}"] = make_backend(spec).server_info()
    log.write_manifest({"config": cfg, "seed": job.seed, "meta": job.meta, "market": m.benchmarks(),
                        "server_info": server, "system_prompts": {f: system_prompt(cfg["prompt"], Identity(**cfg[f"identity{f}"]) if cfg.get(f"identity{f}") else None, m.cost(f),
                                                          (cfg.get("supervision") or {}).get("frame"),
                                                          pressure_for(cfg, f)) for f in (1, 2)}})
    try:
        summary = play(log, m, job.p1, job.p2, cfg["rounds"] if not job.prefix else
                       min(cfg["rounds"], len(job.prefix) + cfg.get("window", 10)),
                       job.seed, cfg["arm"], budget, job.prefix, strict=cfg.get("strict", True),
                       health_cfg=cfg.get("health"), debug_calls=debug_calls, supervisor=_supervisor(cfg, runs_dir, job.seed))
    except Exception as e:  # leave the run incomplete (no summary.json), with the traceback recorded
        log.event("crash", error=repr(e), traceback=traceback.format_exc())
        raise
    log.finish(summary)
    errs = write_validation(log.dir)  # re-derive everything from the logs; analysis refuses invalid runs
    summary["validation_errors"] = len(errs)
    if errs:
        log.debug(f"VALIDATION FAILED ({len(errs)}): {errs[:3]}", echo=True)
    return summary


DESIGN_KEYS = ("stage", "arm", "prompt", "rounds", "sampling", "firm1", "firm2", "market", "identity1", "identity2",
               "shock_round", "window", "source_run_template", "opening", "opening_source", "phase", "fixed_price", "view",
               "step", "impulse", "supervision", "pressure")


def _assert_same_design(run_dir: Path, cfg: dict) -> None:
    old = json.loads((run_dir / "manifest.json").read_text())["config"]
    diff = [k for k in DESIGN_KEYS if old.get(k) != cfg.get(k)]
    if diff:
        raise RuntimeError(f"{run_dir.name} already exists with a different design ({diff}); use --tag for a new run_id")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for run_ids (needed to re-run a crashed id)")
    ap.add_argument("--runs-dir", default=str(REPO / "runs"))
    ap.add_argument("--rounds", type=int, help="override rounds (smoke tests only; recorded in manifest)")
    ap.add_argument("--approve-usd", type=float, default=None,
                    help="REQUIRED for paid models: the spend the team approved for this invocation (hard cap)")
    ap.add_argument("--resume", action="store_true", help="continue interrupted runs (no summary.json) in place")
    ap.add_argument("--debug", type=int, default=0, metavar="N", help="echo the first N calls (prompt + output) per run")
    args = ap.parse_args(argv)
    load_env()
    cfg = load_config(args.config)  # strict: unknown keys / bad values abort before anything runs
    if args.rounds is not None:  # smoke override: validated like the config, and kept apart from the real stage
        cfg["rounds"] = args.rounds
        errs = validate(cfg)
        if errs:
            raise ConfigError(f"--rounds {args.rounds}: " + "; ".join(errs))
        args.tag = f"{args.tag}_r{args.rounds}"
    runs_dir = Path(args.runs_dir)
    jobs = build_jobs(cfg, runs_dir, args.tag)
    todo, crashed = [], []
    for j in jobs:
        d = runs_dir / j.run_id
        if (d / "summary.json").exists():
            _assert_same_design(d, cfg)  # a completed run_id only counts if it ran THIS design
            continue
        if d.exists() and args.resume:
            _assert_same_design(d, cfg)
            j.meta["resume"] = True
            todo.append(j)
        else:
            (crashed if d.exists() else todo).append(j)
    est = estimate(cfg, len(todo))
    paid = [cfg[f] for f in ("firm1", "firm2") if f in cfg and cfg[f]["provider"] not in ("vllm", "mock")]
    if paid:
        b0 = Budget(runs_dir / "_ledger", REPO / "configs" / "prices.yaml")
        share = len(paid) / sum(1 for f in ("firm1", "firm2") if f in cfg)  # fraction of calls that are paid
        cost = lambda out: sum(b0.cost(s["model"], est["input_tokens"] * share / len(paid), out * share / len(paid)) for s in paid)  # noqa: E731
        est["usd_expected"] = round(cost(est["output_tokens_est"]), 2)
        est["usd_worst_no_retries"] = round(cost(est["calls_no_retries"] * cfg["sampling"]["max_tokens"]), 2)
        est["usd_worst_all_retries"] = round(3 * est["usd_worst_no_retries"], 2)
    print(json.dumps({"config": args.config, "jobs": len(jobs), "todo": len(todo),
                      "crashed_untouched": [j.run_id for j in crashed], "estimate": est}, indent=1))
    if paid and not args.dry_run and args.approve_usd is None:
        print(f"REFUSED: paid models {[s['model'] for s in paid]} need explicit approval. Show the team the estimate "
              f"above, then rerun with --approve-usd <amount>.", file=sys.stderr)
        return 2
    if args.dry_run:
        if todo:
            print("--- firm-1 system prompt ---\n" + system_prompt(cfg["prompt"]) + "\n--- round-1 user ---\n" + user_message(1, []))
        return 0
    budget = Budget(runs_dir / "_ledger", REPO / "configs" / "prices.yaml", session_cap_usd=args.approve_usd)
    failures = 0
    with ThreadPoolExecutor(max_workers=cfg.get("parallel_games", 8)) as ex:
        futs = {ex.submit(run_job, j, cfg, runs_dir, budget, args.debug, j.meta.get("resume", False)): j for j in todo}
        for f in as_completed(futs):
            j = futs[f]
            try:
                s = f.result()
                print(f"[done] {j.run_id} delta1={s.get('delta1', float('nan')):.3f} "
                      f"fail={s['parse_fail_rate_calls']:.3f} trunc={s['truncation_rate']:.3f} status={s['status']} "
                      f"invalid={s['validation_errors']}", flush=True)
                failures += s["status"] != "complete" or s["validation_errors"] > 0
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"[CRASH] {j.run_id}: {e!r}", file=sys.stderr, flush=True)
    if crashed:
        print(f"UNFINISHED: {len(crashed)} interrupted run(s) left untouched; rerun with --resume to continue them: "
              f"{[j.run_id for j in crashed][:5]}", file=sys.stderr)
    return 1 if failures or crashed else 0


if __name__ == "__main__":
    raise SystemExit(main())
