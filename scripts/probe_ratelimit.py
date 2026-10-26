"""Measure an OpenAI model's ACTUAL rate limit by load, not by headers.

  python scripts/probe_ratelimit.py MODEL [N=500] [WORKERS=25]

Sends N one-token requests as fast as WORKERS threads allow (aim: ~N per minute), records every status code, the
time of the first 429, and the FULL text of each distinct 429 message (which names the limit type and value).
max_completion_tokens=1, so each request is billed at most ~1 output token (reasoning models reject it unbilled).
Results: docs/api/ratelimit_<model>.json. The key is never printed.
"""
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import dotenv_values

REPO = Path(__file__).resolve().parents[1]
model = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 500
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 25
key = dotenv_values(REPO / ".env")["OPENAI_API_KEY"]
client = httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60)
t0 = time.time()


def one(i):
    try:
        r = client.post("https://api.openai.com/v1/chat/completions",
                        json={"model": model, "max_completion_tokens": 1, "messages": [{"role": "user", "content": "x"}]})
        msg = ""
        if r.status_code != 200:
            try:
                msg = (r.json().get("error") or {}).get("message", "")
            except Exception:
                msg = r.text[:300]
        hdr = {k.replace("x-ratelimit-", ""): v for k, v in r.headers.items() if k.startswith("x-ratelimit")}
        return {"i": i, "t": round(time.time() - t0, 2), "status": r.status_code, "msg": msg, "hdr": hdr}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "t": round(time.time() - t0, 2), "status": "client", "msg": repr(e)[:200], "hdr": {}}


with ThreadPoolExecutor(workers) as ex:
    res = list(ex.map(one, range(n)))
codes = Counter(r["status"] for r in res)
first429 = next((r for r in sorted(res, key=lambda r: r["t"]) if r["status"] == 429), None)
msgs429 = sorted({r["msg"] for r in res if r["status"] == 429})
out = {"model": model, "n": n, "elapsed_s": round(time.time() - t0, 1), "status_counts": dict(codes),
       "first_429_at_s": first429["t"] if first429 else None,
       "accepted_before_first_429": sum(1 for r in res if r["status"] in (200, 400) and (not first429 or r["t"] <= first429["t"])),
       "distinct_429_messages": msgs429, "header_samples": [res[0]["hdr"], res[-1]["hdr"]]}
(REPO / "docs/api" / f"ratelimit_{model}.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "header_samples"}, indent=1)[:1500])
