"""Parse Lee's JSON reply {price, plan, insight}. The LAST well-formed JSON object containing "price" wins; if no
object parses, a bare `"price": <num>` is accepted with empty plan/insight (status "price_only"). Prices are clipped
to [pmin, pmax], and clipping is reported.
"""
from __future__ import annotations

import json
import math
import re

_OBJ = re.compile(r"\{[^{}]*\}", re.S)
_PRICE = re.compile(r'"price"\s*:\s*"?\s*(-?\d+(?:\.\d+)?)')


def parse_reply(text: str | None, pmin: float = 0.0, pmax: float = 5.0) -> dict:
    if not text or not text.strip():
        return {"status": "empty", "price": None, "plan": None, "insight": None, "clipped": False}
    for m in reversed(list(_OBJ.finditer(text))):
        try:
            o = json.loads(m.group(0))
        except ValueError:
            continue
        if isinstance(o, dict) and "price" in o:
            v = o["price"]
            if isinstance(v, bool):  # JSON true/false is not a price (float(True) would be 1.0)
                continue
            try:
                p = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(p):  # "nan" / "inf" strings parse as floats
                continue
            c = min(pmax, max(pmin, p))
            return {"status": "ok", "price": c, "raw_price": p, "plan": str(o.get("plan", ""))[:600],
                    "insight": str(o.get("insight", ""))[:600], "clipped": c != p}
    m = list(_PRICE.finditer(text))
    if m:
        p = float(m[-1].group(1))
        c = min(pmax, max(pmin, p))
        return {"status": "price_only", "price": c, "raw_price": p, "plan": None, "insight": None, "clipped": c != p}
    return {"status": "no_price", "price": None, "plan": None, "insight": None, "clipped": False}
