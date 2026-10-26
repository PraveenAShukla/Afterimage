"""Check API keys at ZERO cost: list the models each key can see (no generation, no tokens billed).

  python -m collusion.keycheck

Prints, per provider: key present (never the value), HTTP status, and which of our candidate models are available.
"""
from __future__ import annotations

import os
import sys

import httpx

from .env import load_env

CANDIDATES = {
    "anthropic": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"],
    "openai": [],  # filled once we pick OpenAI models; listing shows what the key can use
}


def check_anthropic() -> tuple[bool, str, list[str]]:
    import anthropic
    from .backends.http import DEFAULT_ANTHROPIC_KEY_ENV as K
    if not os.environ.get(K):
        return False, f"{K} empty", []
    try:
        ids = [m.id for m in anthropic.Anthropic(api_key=os.environ[K], max_retries=1, timeout=30).models.list(limit=100)]
        return True, "ok", ids
    except anthropic.AuthenticationError:
        return False, "401: key rejected", []
    except anthropic.APIError as e:
        return False, f"{type(e).__name__}: {str(e)[:120]}", []


def check_openai() -> tuple[bool, str, list[str]]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return False, "OPENAI_API_KEY empty", []
    try:
        r = httpx.get("https://api.openai.com/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    except httpx.HTTPError as e:
        return False, f"{type(e).__name__}", []
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}", []
    return True, "ok", sorted(m["id"] for m in r.json().get("data", []))


def main() -> int:
    load_env()
    bad = 0
    for prov, fn in (("anthropic", check_anthropic), ("openai", check_openai)):
        ok, msg, ids = fn()
        bad += not ok
        print(f"{prov:<10} {'OK  ' if ok else 'FAIL'} {msg}  ({len(ids)} models visible)")
        for m in CANDIDATES[prov]:
            print(f"    {m:<22} {'available' if any(i == m or i.startswith(m + '-') for i in ids) else 'NOT visible'}")
        if prov == "openai" and ids:
            print("    e.g.", ", ".join(i for i in ids if i.startswith(("gpt-", "o")))[:300])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
