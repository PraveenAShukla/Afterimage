"""Post-run validator: re-derives everything from the raw logs and checks it against what was logged.

  python -m collusion.validate runs/            # every run
  python -m collusion.validate runs/<run_id>    # one run

Runs automatically at the end of every run (validation.json). analysis/analyze.py refuses runs that did not pass.
Checks (each failure is a precise message, never a silent fix):
  1. manifest has the fields docs/DESIGN.md 3.6 requires (sampling config, server/vLLM info, seed, code version)
  2. rounds are 1..n with no gaps; prefix rounds come first
  3. shares/profits/BRs/cumulative profits recompute exactly from the played prices
  4. every live decision has 1-3 call lines with attempts 0..k; the decision's played price follows the rules
     (ok -> clip(parsed); fallback_previous -> previous round's price); non-live rounds have NO calls
  5. every call line carries the spec's required fields
  6. the history inside every prompt equals the played history (no leaks, no peeking at the current round)
  7. summary.json's headline numbers recompute from rounds.jsonl
  8. P: firm 2's prices equal the source L run's firm 2; D: the prefix equals the source L run
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

from .market import Market, collusion_index
from .prompts import SHAM_LINE

HINT_RE = re.compile(r"Note: if the competitor repeats its last price \((\d\.\d{3})\), your profit this round is "
                     r"highest at a price of (\d\.\d{3})\.")
from .prompts import PRESSURE, REVIEW_LABELS, SUPERVISION, review_line
from .parse import clip, parse_price
from .runlog import read_jsonl

CALL_FIELDS = ["run_id", "arm", "seed", "round", "firm", "model", "provider", "sampling", "system_prompt",
               "user_prompt", "raw_output", "parsed_price", "parse_status", "attempt", "call_seed",
               "final_price", "decision_status", "input_tokens", "output_tokens", "finish_reason"]
ROW = re.compile(r"^(\d+), (-?[\d.]+), (-?[\d.]+)(?:, (-?[\d.]+), (-?[\d.]+))?(?:, -?[\d.]+)?$", re.M)  # share/profit hidden under E1-PO; trailing = anchor
TOL = 1e-9


def _close(a, b, tol=TOL) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=0, abs_tol=tol)


def validate_run(d: Path) -> list[str]:
    d = Path(d)
    errs: list[str] = []
    try:
        man = json.loads((d / "manifest.json").read_text())
        rounds = read_jsonl(d / "rounds.jsonl")
        calls = read_jsonl(d / "calls.jsonl")
        summ = json.loads((d / "summary.json").read_text())
    except FileNotFoundError as e:
        return [f"missing file: {e.filename}"]
    cfg = man.get("config", {})
    for k in ("config", "seed", "code", "env", "market", "server_info", "system_prompts"):
        if k not in man:
            errs.append(f"manifest missing {k!r}")
    if "sampling" not in cfg:
        errs.append("manifest config has no sampling block")
    m = Market(**cfg.get("market", {}))
    b = m.benchmarks()

    # 2. contiguity
    ts = [r["t"] for r in rounds]
    if ts != list(range(1, len(ts) + 1)):
        errs.append(f"rounds not contiguous from 1: first gap near {next((i + 1 for i, t in enumerate(ts) if t != i + 1), '?')}")
    pre = [r.get("is_prefix", False) for r in rounds]
    if pre != sorted(pre, reverse=True):
        errs.append("prefix rounds are not all at the start")

    # 3. market recomputation
    cum = [0.0, 0.0]
    for r in rounds:
        x = m.clear(r["p1"], r["p2"])
        for k, v in x.items():
            if not _close(r[k], v):
                errs.append(f"t={r['t']}: {k} logged {r[k]} != recomputed {v}")
        if r.get("is_prefix"):
            continue
        cum[0] += x["pi1"]; cum[1] += x["pi2"]
        for k, v in (("br1", m.best_response(r["p2"], firm=1)), ("br2", m.best_response(r["p1"], firm=2)),
                     ("cum_pi1", cum[0]), ("cum_pi2", cum[1])):
            if not _close(r.get(k), v, 1e-7):
                errs.append(f"t={r['t']}: {k} logged {r.get(k)} != recomputed {v}")

    # 4-6. calls vs decisions. Calls logged before a resume for rounds that never completed are orphans: kept on
    # disk (append-only) but excluded from checks.
    events = read_jsonl(d / "events.jsonl") if (d / "events.jsonl").exists() else []
    resumes = [(e["ts"], e["from_round"]) for e in events if e.get("event") == "resume"]
    orphan = lambda c: any(c["ts"] < ts and c["round"] >= fr for ts, fr in resumes)  # noqa: E731
    calls = [c for c in calls if not orphan(c)]
    by = defaultdict(list)
    for c in calls:
        if c.get("decision_status") == "budget_stop":  # attempts of the round the budget stopped (no round line)
            continue
        miss = [f for f in CALL_FIELDS if f not in c]
        if miss:  # reported once, then excluded so no deeper check can crash on it (shows up as an attempt gap)
            errs.append(f"call t={c.get('round')} firm={c.get('firm')}: missing fields {miss}")
            continue
        by[(c["round"], c["firm"])].append(c)
    rmap = {r["t"]: r for r in rounds}
    for r in rounds:
        for firm in (1, 2):
            src = r.get(f"source{firm}")
            cs = sorted(by.get((r["t"], firm), []), key=lambda c: c["attempt"])
            if src != "live":
                if cs:
                    errs.append(f"t={r['t']} firm={firm}: source={src} but {len(cs)} call lines exist")
                want = None
                if src == "fixed":
                    want = (b["pN"] if cfg.get("arm") == "FN" else b["pM"] if cfg.get("arm") == "FM"
                            else cfg.get("fixed_price") if cfg.get("arm") == "Q"
                            else man.get("meta", {}).get("step_level") if cfg.get("arm") == "STEP" else None)
                    if want is None:
                        errs.append(f"t={r['t']} firm={firm}: fixed price in arm {cfg.get('arm')}")
                elif src == "forced":
                    want = man.get("meta", {}).get(f"forced{firm}", {}).get(str(r["t"]))
                    if want is None:
                        errs.append(f"t={r['t']} firm={firm}: forced price not recorded in manifest meta")
                elif src == "impulse":  # arm IMP: own pre-shock price + delta, clipped (decided at runtime)
                    im = cfg.get("impulse", {})
                    base = rmap.get(im.get("round", 0) - 1, {}).get(f"p{firm}")
                    want = None if base is None else min(2.5, max(1.0, base + float(im.get("delta", 0))))
                    if want is None:
                        errs.append(f"t={r['t']} firm={firm}: impulse without a pre-shock price")
                elif src not in ("replay", "prefix"):
                    errs.append(f"t={r['t']} firm={firm}: unknown price source {src!r}")
                if want is not None and not _close(r[f"p{firm}"], want):
                    errs.append(f"t={r['t']} firm={firm}: {src} price {r[f'p{firm}']} != expected {want}")
                continue
            if not 1 <= len(cs) <= 3 or [c["attempt"] for c in cs] != list(range(len(cs))):
                errs.append(f"t={r['t']} firm={firm}: bad attempt sequence {[c['attempt'] for c in cs]}")
                continue
            played, status = r[f"p{firm}"], r.get(f"status{firm}")
            last = cs[-1]
            if not _close(last["final_price"], played):
                errs.append(f"t={r['t']} firm={firm}: call final_price {last['final_price']} != played {played}")
            if status in ("ok", "ok_after_retry"):
                if last["parse_status"] != "ok" or not _close(clip(last["parsed_price"])[0], played):
                    errs.append(f"t={r['t']} firm={firm}: status {status} but played != clip(parsed)")
                if any(c["parse_status"] == "ok" for c in cs[:-1]):
                    errs.append(f"t={r['t']} firm={firm}: retried after a successful parse")
            elif status == "fallback_previous":
                if len(cs) != 3 or any(c["parse_status"] == "ok" for c in cs):
                    errs.append(f"t={r['t']} firm={firm}: fallback without 3 failed attempts")
                prev = rmap.get(r["t"] - 1)
                if prev is None or not _close(prev[f"p{firm}"], played):
                    errs.append(f"t={r['t']} firm={firm}: fallback price != previous round's price")
            elif status != "fallback_initial":
                errs.append(f"t={r['t']} firm={firm}: unknown decision status {status!r}")
            own, riv = (str(firm), str(3 - firm))
            sv = cfg.get("supervision") or {}
            smode = sv.get("mode", "note")
            v1note = smode == "note" and sv.get("agent")
            want_note = r.get(f"note{firm}") if v1note else None
            want_rev = r.get(f"review{firm}") if smode in ("review", "yoked") else None
            for c in cs:
                up = c["user_prompt"]
                if smode in ("review", "yoked"):
                    if f"review{firm}" not in r:
                        errs.append(f"t={r['t']} firm={firm}: review arm but the round has no review field")
                    ok_line = up.endswith("\n" + review_line(sv["frame"], want_rev)) if want_rev in REVIEW_LABELS \
                        else "Supervisor's review" not in up
                    if want_rev not in (None, *REVIEW_LABELS) or not ok_line:
                        errs.append(f"t={r['t']} firm={firm}: review line in the prompt does not match the round record")
                if v1note and f"note{firm}" not in r:
                    errs.append(f"t={r['t']} firm={firm}: supervisor arm but the round has no note field")
                if (up.endswith(f"\nSupervisor's note: {want_note}") if want_note else "Supervisor's note:" not in up) is False:
                    errs.append(f"t={r['t']} firm={firm}: supervisor note in the prompt does not match the round record")
                pr = cfg.get("pressure") or {}
                want_pr = pr.get("frame") if pr and (pr["target"] == "both" or firm == 1) else None
                if any((PRESSURE[k] in c.get("system_prompt", "")) != (k == want_pr) for k in PRESSURE):
                    errs.append(f"t={r['t']} firm={firm}: pressure line in the system prompt does not match the config")
                if bool(sv) != (SUPERVISION.get(sv.get("frame"), "\0") in c.get("system_prompt", "")):
                    errs.append(f"t={r['t']} firm={firm}: supervision line in the system prompt does not match the config")
                if not up.startswith(f"Round {r['t']}. "):
                    errs.append(f"t={r['t']} firm={firm}: prompt header is {up[:20]!r}")
                shown = [int(x[0]) for x in ROW.findall(up)]
                if shown != list(range(max(1, r["t"] - 20), r["t"])):
                    errs.append(f"t={r['t']} firm={firm}: prompt shows rounds {shown[:3]}..{shown[-3:]} (expected last 20 before t)")
                kind = (cfg.get("view") or {}).get("line")
                got = [x for x in up.splitlines() if x.startswith("Note: ")]
                if r["t"] == 1 or not kind:
                    if got:
                        errs.append(f"t={r['t']} firm={firm}: unexpected view line {got[0][:40]!r}")
                elif kind == "sham" and got != [SHAM_LINE]:
                    errs.append(f"t={r['t']} firm={firm}: sham line missing or wrong")
                elif kind == "br_hint":
                    rows = ROW.findall(up)
                    rr = round(float(rows[-1][2]), 3) if rows else None
                    mh = HINT_RE.fullmatch(got[0]) if len(got) == 1 else None
                    if rr is None or not mh or abs(float(mh.group(1)) - rr) > 6e-4 or \
                            abs(float(mh.group(2)) - m.best_response(rr, firm=firm)) > 6e-4:
                        errs.append(f"t={r['t']} firm={firm}: br_hint line does not match the shown rival price")
                if (r["t"] == 1) != up.rstrip().endswith("No history yet."):
                    errs.append(f"t={r['t']} firm={firm}: 'No history yet.' wrongly present/absent")
                if c.get("transport_error"):
                    if c["parse_status"] != "transport_error":
                        errs.append(f"t={r['t']} firm={firm}: transport error not marked as such")
                elif parse_price(c.get("raw_output")) != (c.get("parsed_price"), c.get("parse_status")):
                    errs.append(f"t={r['t']} firm={firm} attempt={c['attempt']}: re-parsing raw_output gives "
                                f"{parse_price(c.get('raw_output'))}, logged ({c['parsed_price']}, {c['parse_status']})")
                for tt, op, rp, sh, pf in ROW.findall(c["user_prompt"]):
                    h = rmap.get(int(tt))
                    if h is None:
                        errs.append(f"t={r['t']} firm={firm}: prompt shows round {tt}, which is missing from rounds.jsonl")
                        break
                    po = (cfg.get("view") or {}).get("price_only")
                    if bool(po) != (sh == ""):
                        errs.append(f"t={r['t']} firm={firm}: share/profit columns {'present' if sh else 'missing'} against the view")
                        break
                    if not (_close(float(op), h["p" + own], 6e-4) and _close(float(rp), h["p" + riv], 6e-4)
                            and (po or (_close(float(sh), h["s" + own], 6e-4) and _close(float(pf), h["pi" + own], 6e-4)))):
                        errs.append(f"t={r['t']} firm={firm}: prompt row {tt} disagrees with played history")
                        break

    # 6c. supervision v2: review labels come from supervisor.jsonl (review) or the yoke source run (yoked)
    sv2 = cfg.get("supervision") or {}
    if sv2.get("mode") == "review":
        sp = Path(d) / "supervisor.jsonl"
        final = {}
        for c in (read_jsonl(sp) if sp.exists() else []):
            for f in ((1, 2) if c["to"] == "both" else (c["to"],)):
                if c.get("review") or (c["round"], f) not in final:
                    final[(c["round"], f)] = c.get("review")
        for r in rounds:
            if not r.get("is_prefix"):
                for f in (1, 2):
                    if final.get((r["t"], f)) != r.get(f"review{f}"):
                        errs.append(f"t={r['t']} firm={f}: round review differs from supervisor.jsonl")
                        break
    elif sv2.get("mode") == "yoked":
        src = sv2["yoke"].get(man["seed"], sv2["yoke"].get(str(man["seed"])))
        sd = Path(d).parent / str(src)
        if not (sd / "rounds.jsonl").exists():
            errs.append(f"yoke source {src} not found next to this run")
        else:
            srcr = {x["t"]: x for x in read_jsonl(sd / "rounds.jsonl") if not x.get("is_prefix")}
            for r in rounds:
                if not r.get("is_prefix"):
                    for f in (1, 2):
                        if (srcr.get(r["t"]) or {}).get(f"review{f}") != r.get(f"review{f}"):
                            errs.append(f"t={r['t']} firm={f}: yoked review differs from source run {src}")
                            break
    # 6b. supervisor notes: the note each firm saw is the supervisor's last successful note for that round
    if (cfg.get("supervision") or {}).get("agent") and (cfg.get("supervision") or {}).get("mode", "note") == "note":
        sp = Path(d) / "supervisor.jsonl"
        final = {}
        for c in (read_jsonl(sp) if sp.exists() else []):
            for f in ((1, 2) if c["to"] == "both" else (c["to"],)):
                if c["note"] or (c["round"], f) not in final:
                    final[(c["round"], f)] = c["note"]
        for r in rounds:
            if not r.get("is_prefix"):
                for f in (1, 2):
                    if final.get((r["t"], f)) != r.get(f"note{f}"):
                        errs.append(f"t={r['t']} firm={f}: round note differs from supervisor.jsonl")
                        break
    # 7. summary recomputation
    live = [r for r in rounds if not r.get("is_prefix")][-50:]
    if live:
        mean = lambda k: sum(r[k] for r in live) / len(live)  # noqa: E731
        pi1, pi2 = mean("pi1"), mean("pi2")
        want = {"lastK": len(live), "mean_p1": mean("p1"), "mean_p2": mean("p2"), "mean_pi1": pi1, "mean_pi2": pi2,
                "delta1": collusion_index(pi1, b["piN"], b["piM"]), "delta2": collusion_index(pi2, b["piN"], b["piM"]),
                "delta_avg": collusion_index((pi1 + pi2) / 2, b["piN"], b["piM"]),
                "profit_ratio1": pi1 / b["piN"], "profit_ratio2": pi2 / b["piN"],
                "mean_abs_gap_br1": sum(abs(r["p1"] - m.best_response(r["p2"], firm=1)) for r in live) / len(live)}
        for k, v in want.items():
            if k not in summ or not _close(summ[k], v, 1e-7):
                errs.append(f"summary {k} {summ.get(k)} != recomputed {v}")
        n_calls = len(calls)  # includes budget_stop attempts, as stats does
        if summ.get("calls") != n_calls:
            errs.append(f"summary calls {summ.get('calls')} != call lines {n_calls}")

    # 8. lineage
    meta = man.get("meta", {})
    runs_dir = d.parent
    if "replays" in meta:
        src = read_jsonl(runs_dir / meta["replays"] / "rounds.jsonl")
        sp2 = [r["p2"] for r in src if not r.get("is_prefix")]
        if [r["p2"] for r in rounds] != sp2[: len(rounds)]:
            errs.append(f"P replay: firm-2 prices differ from source {meta['replays']}")
    if "branch_of" in meta:
        src = {r["t"]: r for r in read_jsonl(runs_dir / meta["branch_of"] / "rounds.jsonl")}
        for r in rounds:
            if r.get("is_prefix") and not (_close(r["p1"], src[r["t"]]["p1"]) and _close(r["p2"], src[r["t"]]["p2"])):
                errs.append(f"D prefix t={r['t']} differs from source {meta['branch_of']}")
                break
    if "prefix_sha256" in meta:  # every v2 prefixed run: the played prefix must hash to the recorded value
        from .opening import prefix_fingerprint
        fp = prefix_fingerprint([r for r in rounds if r.get("is_prefix")])
        if fp["prefix_sha256"] != meta["prefix_sha256"]:
            errs.append("prefix sha256 differs from the manifest")
    if "transplant_source_run" in meta:  # arm T: the prefix must equal the source run's own opening rounds
        src = {r["t"]: r for r in read_jsonl(runs_dir / meta["transplant_source_run"] / "rounds.jsonl")}
        for r in rounds:
            if r.get("is_prefix") and not (_close(r["p1"], src.get(r["t"], {}).get("p1")) and _close(r["p2"], src.get(r["t"], {}).get("p2"))):
                errs.append(f"T prefix t={r['t']} differs from source {meta['transplant_source_run']}")
                break
    if "opening_source_run" in meta:  # arms S/A/O: the prefix must be the recorded level-shift of the source window
        from .opening import shifted_window
        pre_rows = [r for r in rounds if r.get("is_prefix")]
        try:
            want, info = shifted_window(read_jsonl(runs_dir / meta["opening_source_run"] / "rounds.jsonl"),
                                        *meta["opening"], n_prefix=len(pre_rows))
        except Exception as e:  # noqa: BLE001 - report, never crash the validator
            errs.append(f"opening: cannot rebuild prefix from {meta['opening_source_run']}: {e!r}")
        else:
            if [r["t"] for r in pre_rows] != [w["t"] for w in want]:
                errs.append("opening: prefix rounds differ from the rebuilt prefix")
            elif any(not (_close(r["p1"], w["p1"]) and _close(r["p2"], w["p2"])) for r, w in zip(pre_rows, want, strict=True)):
                errs.append(f"opening: prefix prices differ from the level-shifted source {meta['opening_source_run']}")
            if any(not _close(a, b) for a, b in zip(info["delta"], meta.get("opening_info", {}).get("delta", [None, None]), strict=True)):
                errs.append("opening: recorded delta differs from the rebuilt delta")
            if list(meta["opening"]) != list(cfg.get("opening", [])):
                errs.append("opening: manifest meta opening differs from config opening")
    return errs


def write_validation(d: Path) -> list[str]:
    errs = validate_run(d)
    (Path(d) / "validation.json").write_text(json.dumps({"ok": not errs, "n_errors": len(errs), "errors": errs[:200]}, indent=1))
    return errs


def main(argv=None) -> int:
    target = Path((argv or sys.argv[1:])[0])
    dirs = [target] if (target / "manifest.json").exists() else sorted(
        p for p in target.iterdir() if p.is_dir() and (p / "summary.json").exists())
    bad = 0
    for p in dirs:
        errs = write_validation(p)
        bad += bool(errs)
        print(f"{'OK  ' if not errs else 'FAIL'} {p.name}" + ("" if not errs else f"  ({len(errs)}) {errs[0]}"))
    print(f"{len(dirs) - bad}/{len(dirs)} runs valid")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
