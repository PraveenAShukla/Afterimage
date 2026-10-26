"""Extract the LAST {"price": x} object from model output. Returns (price | None, status)."""
from __future__ import annotations

import math
import re

# Tolerates whitespace, quotes around the number, and extra QUOTED keys after price; requires the "price" key.
# A decimal comma ({"price": 1,5}) does not match: it is a parse failure (retry), never silently 1.0.
_PRICE_OBJ = re.compile(r'\{\s*"price"\s*:\s*"?\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*"?\s*(?:\s*,\s*"[^{}]*)?\}')

STATUS_OK = "ok"
STATUS_EMPTY = "empty_output"
STATUS_NO_JSON = "no_price_json"
STATUS_NONFINITE = "non_finite"


def parse_price(text: str | None) -> tuple[float | None, str]:
    if not text or not text.strip():
        return None, STATUS_EMPTY
    matches = _PRICE_OBJ.findall(text)
    if not matches:
        return None, STATUS_NO_JSON
    price = float(matches[-1])
    if not math.isfinite(price):
        return None, STATUS_NONFINITE
    return price, STATUS_OK


def price_in_think(text: str | None) -> bool:
    """True if the LAST price object lies before the last </think>, i.e. it is a draft from the reasoning and the
    final answer has no price. Spec 3.2 still uses it (parse the last JSON in the full output); we flag and count it."""
    if not text or "</think>" not in text:
        return False
    last = None
    for m in _PRICE_OBJ.finditer(text):
        last = m
    return last is not None and last.start() < text.rfind("</think>")


def clip(price: float, lo: float = 1.0, hi: float = 2.5) -> tuple[float, bool]:
    clipped = min(max(price, lo), hi)
    return clipped, clipped != price
