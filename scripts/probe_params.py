"""Which request parameters does each paid model accept? Tiny live probe, through the budget guard.

  python scripts/probe_params.py --approve-usd 0.05 [models...]

Every variant is sent once with a tiny max output. A 4xx means "not accepted" (4xx responses are not billed). The
results go to docs/api/param_probe.json and docs/api/param_probe.md. Keys are never printed.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from collusion.budget import Budget, BudgetExceeded  # noqa: E402
from collusion.env import load_env  # noqa: E402
from collusion.ratelimit import get_limiter  # noqa: E402

SYSTEM = 'You are a pricing agent. End with a JSON object {"price": <number>}.'
USER = "Round 1. No history yet."
MAXOUT = 64

OPENAI_VARIANTS = {
    "baseline (max_completion_tokens)": {},
    "temperature=0.7": {"temperature": 0.7},
    "temperature=1.0": {"temperature": 1.0},
    "top_p=0.9": {"top_p": 0.9},
    "seed=123": {"seed": 123},
    "reasoning_effort=none": {"reasoning_effort": "none"},
    "reasoning_effort=minimal": {"reasoning_effort": "minimal"},
    "reasoning_effort=low": {"reasoning_effort": "low"},
    "reasoning_effort=none + temperature=0.7": {"reasoning_effort": "none", "temperature": 0.7},
    "verbosity=low": {"verbosity": "low"},
    "logprobs=true": {"logprobs": True},
    "legacy max_tokens": {"_legacy_max_tokens": True},
}
ANTHROPIC_VARIANTS = {
    "baseline (effort=low)": {},
    "temperature=0.7": {"temperature": 0.7},
    "temperature=1.0": {"temperature": 1.0},
    "top_p=0.9": {"top_p": 0.9},
    "top_k=5": {"top_k": 5},
    "thinking=disabled": {"thinking": {"type": "disabled"}},
    "thinking=between_tools": {"thinking": {"type": "between_tools"}},
    "thinking=adaptive+summarized": {"thinking": {"type": "adaptive", "display": "summarized"}},
    "effort=high": {"output_config": {"effort": "high"}},
}


def probe_openai(model: str, extra: dict) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]}
    if extra.pop("_legacy_max_tokens", False):
        body["max_tokens"] = MAXOUT
    else:
        body["max_completion_tokens"] = MAXOUT
    body.update(extra)
    r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, timeout=60,
                   headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    if r.status_code != 200:
        err = r.json().get("error", {}) if "json" in r.headers.get("content-type", "") else {}
        return {"accepted": False, "status": r.status_code, "error": (err.get("message") or r.text)[:240]}
    js = r.json()
    u = js.get("usage") or {}
    return {"accepted": True, "status": 200, "finish": js["choices"][0].get("finish_reason"),
            "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "fingerprint": js.get("system_fingerprint"), "model_returned": js.get("model")}


def probe_anthropic(model: str, extra: dict) -> dict:
    import anthropic
    c = anthropic.Anthropic(max_retries=0, timeout=60)
    body = {"model": model, "max_tokens": MAXOUT, "system": SYSTEM, "messages": [{"role": "user", "content": USER}]}
    raw = {"output_config": {"effort": "low"}, **extra}   # sent raw so the SERVER (not the SDK) decides acceptance
    try:
        m = c.messages.create(**body, extra_body=raw)
    except anthropic.APIStatusError as e:
        return {"accepted": False, "status": e.status_code, "error": str(e)[:240]}
    return {"accepted": True, "status": 200, "finish": m.stop_reason, "in": m.usage.input_tokens,
            "out": m.usage.output_tokens, "blocks": [b.type for b in m.content], "model_returned": m.model}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["gpt-6-luna", "gpt-6.1-sol", "claude-haiku-5-5", "claude-sonnet-5-5"])
    ap.add_argument("--approve-usd", type=float, required=True)
    a = ap.parse_args()
    load_env()
    budget = Budget(REPO / "runs" / "_ledger", REPO / "configs" / "prices.yaml", session_cap_usd=a.approve_usd)
    results = {}
    for model in a.models:
        prov = "anthropic" if model.startswith("claude") else "openai"
        variants = ANTHROPIC_VARIANTS if prov == "anthropic" else OPENAI_VARIANTS
        lim = get_limiter(prov)
        for name, extra in variants.items():
            try:
                rid = budget.reserve(prov, model, 60, MAXOUT)
            except BudgetExceeded as e:
                print(f"STOP: {e}")
                break
            ev = lim.acquire(MAXOUT)
            try:
                res = (probe_anthropic if prov == "anthropic" else probe_openai)(model, dict(extra))
            except Exception as e:  # client-side failure: request not sent, nothing billed
                res = {"accepted": False, "status": "client", "error": f"{type(e).__name__}: {e}"[:240]}
            lim.settle(ev, res.get("out"))
            res["usd"] = budget.record(prov, model, "param-probe", res.get("in"), res.get("out"), rid)
            results.setdefault(model, {})[name] = res
            print(f"{model:<18} {name:<42} {'OK ' if res['accepted'] else res['status']} "
                  f"{'' if res['accepted'] else res['error'][:110]}")
            time.sleep(0.2)
    out = REPO / "docs" / "api"
    (out / "param_probe.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "results": results}, indent=1))
    lines = ["# Parameter acceptance probe (live, " + time.strftime("%Y-%m-%d") + ")", "",
             "| model | variant | accepted | detail |", "|---|---|---|---|"]
    for m, vs in results.items():
        for n, r in vs.items():
            det = (f"finish={r.get('finish')} out={r.get('out')}" + (f" reasoning={r.get('reasoning_tokens')}" if r.get("reasoning_tokens") is not None else "")
                   if r["accepted"] else f"{r['status']}: {r['error'][:140]}")
            lines.append(f"| {m} | {n} | {'yes' if r['accepted'] else 'NO'} | {det.replace('|', '/')} |")
    total = sum(r["usd"] for vs in results.values() for r in vs.values())
    lines += ["", f"Total spend: ${total:.5f} (approved cap ${a.approve_usd:.2f})."]
    (out / "param_probe.md").write_text("\n".join(lines) + "\n")
    print(f"total ${total:.5f} -> docs/api/param_probe.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
