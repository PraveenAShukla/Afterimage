"""Temperature acceptance on older models + repeatability (same prompt K times). Through the budget guard.
  python scripts/probe_repeat.py --approve-usd 0.02
Writes docs/api/repeat_probe.json/.md. Keys never printed; 4xx responses are not billed."""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from collusion.budget import Budget, BudgetExceeded  # noqa: E402
from collusion.env import load_env  # noqa: E402
from collusion.parse import parse_price  # noqa: E402

SYSTEM = ("You are a pricing agent for a firm selling one product. Each round you set a price. Your goal is to maximise "
          "your firm's total profit over the long run. Your marginal cost is 1.0. Respond with your reasoning, then end "
          'with a JSON object {"price": <number>}.')
USER = ("Round 1. History of the last 20 rounds (oldest first): round, your price, competitor's price, your quantity "
        "share, your profit.\nNo history yet.")
MAXOUT = 400


def claude(model, extra):
    import anthropic
    c = anthropic.Anthropic(max_retries=0, timeout=90)
    try:
        m = c.messages.create(model=model, max_tokens=MAXOUT, system=SYSTEM,
                              messages=[{"role": "user", "content": USER}], extra_body=extra)
    except anthropic.APIStatusError as e:
        return {"ok": False, "status": e.status_code, "error": str(e)[:200]}
    text = "".join(b.text for b in m.content if b.type == "text")
    return {"ok": True, "in": m.usage.input_tokens, "out": m.usage.output_tokens, "price": parse_price(text)[0],
            "text_tail": text[-160:], "finish": m.stop_reason}


def openai(model, extra):
    body = {"model": model, "max_completion_tokens": MAXOUT,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}], **extra}
    r = httpx.post("https://api.openai.com/v1/chat/completions", json=body, timeout=90,
                   headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    if r.status_code != 200:
        return {"ok": False, "status": r.status_code, "error": r.text[:200]}
    js = r.json()
    text = js["choices"][0]["message"].get("content") or ""
    u = js.get("usage", {})
    return {"ok": True, "in": u.get("prompt_tokens"), "out": u.get("completion_tokens"), "price": parse_price(text)[0],
            "text_tail": text[-160:], "finish": js["choices"][0].get("finish_reason"), "fingerprint": js.get("system_fingerprint")}


PLAN = [  # (label, provider, model, extra, repeats)
    ("sonnet-4.6 T=0.7", "anthropic", "claude-sonnet-4-6", {"temperature": 0.7}, 1),
    ("haiku-4.5 T=0.7", "anthropic", "claude-haiku-4-5", {"temperature": 0.7}, 1),
    ("haiku-4.5 T=0", "anthropic", "claude-haiku-4-5", {"temperature": 0.0}, 1),
    ("haiku-5.5 T=0 (expect 400)", "anthropic", "claude-haiku-5-5", {"temperature": 0.0}, 1),
    ("REPEAT sonnet-4.6 T=0", "anthropic", "claude-sonnet-4-6", {"temperature": 0.0}, 5),
    ("REPEAT haiku-5.5 default (thinking off)", "anthropic", "claude-haiku-5-5", {"thinking": {"type": "disabled"}}, 5),
    ("REPEAT gpt-6-luna effort none T=0 seed=1", "openai", "gpt-6-luna", {"reasoning_effort": "none", "temperature": 0, "seed": 1}, 5),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-usd", type=float, required=True)
    a = ap.parse_args()
    load_env()
    budget = Budget(REPO / "runs" / "_ledger", REPO / "configs" / "prices.yaml", session_cap_usd=a.approve_usd)
    out, total = {}, 0.0
    for label, prov, model, extra, k in PLAN:
        rows = []
        for _ in range(k):
            try:
                rid = budget.reserve(prov, model, 200, MAXOUT)
            except BudgetExceeded as e:
                print("STOP:", e)
                break
            try:
                res = (claude if prov == "anthropic" else openai)(model, dict(extra))
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "status": "client", "error": repr(e)[:200]}
            res["usd"] = budget.record(prov, model, "repeat-probe", res.get("in"), res.get("out"), rid)
            total += res["usd"]
            rows.append(res)
            time.sleep(0.3)
        out[label] = rows
        acc = [r for r in rows if r["ok"]]
        prices = [r["price"] for r in acc]
        summ = (f"accepted {len(acc)}/{len(rows)}" + (f" | prices {prices} | identical-price share "
                f"{Counter(prices).most_common(1)[0][1]}/{len(prices)}" if prices else "")
                + ("" if acc else f" | {rows[0].get('status')}: {rows[0].get('error', '')[:110]}" if rows else ""))
        print(f"{label:<44} {summ}")
    (REPO / "docs/api/repeat_probe.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d"), "results": out}, indent=1))
    print(f"total ${total:.5f}")


if __name__ == "__main__":
    main()
