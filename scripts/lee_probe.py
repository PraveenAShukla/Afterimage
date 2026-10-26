"""Before any Lee run: check each provider's key and settings. PAID: one call per model, about $0.001 each; needs
the team's approval.

  .venv/bin/python scripts/lee_probe.py --approve-usd 0.05
Sends Lee's real round-1 prompt with the config's `extra` (no temperature) and reports per model: HTTP OK or the
error text (e.g. an unknown reasoning parameter), the model the server reports, tokens, and whether the reply parses
as Lee's JSON. Keys are never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from collusion.budget import Budget  # noqa: E402
from collusion.env import load_env  # noqa: E402
from lee.backend import ChatBackend  # noqa: E402
from lee.parse import parse_reply  # noqa: E402
from lee.prompts import system_prompt, user_message  # noqa: E402
from make_lee_configs import MODELS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approve-usd", type=float, required=True)
    a = ap.parse_args()
    load_env()
    budget = Budget(ROOT / "runs_lee" / "_ledger", ROOT / "configs" / "prices.yaml", session_cap_usd=a.approve_usd)
    for name, f in MODELS.items():
        b = ChatBackend(f["provider"], f["model"], f["extra"])
        try:
            res = budget.reserve(b.provider, b.model, 400, 600)
        except Exception as e:  # noqa: BLE001 - report the guard's refusal (e.g. no BUDGET_USD_<PROVIDER> cap)
            print(f"{name:<9} REFUSED by budget guard: {e}")
            continue
        c = b.complete(system_prompt(0), user_message(1, None, None, [], 0))
        usd = budget.record(b.provider, b.model, "lee-probe", c.input_tokens, c.output_tokens, res)
        r = parse_reply(c.text)
        print(f"{name:<9} {'OK ' if not c.error else 'ERR'} model={c.model_returned} in={c.input_tokens} out={c.output_tokens} "
              f"usd={usd:.5f} parse={r['status']} price={r['price']} " + (f"error={c.error[:200]}" if c.error else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
