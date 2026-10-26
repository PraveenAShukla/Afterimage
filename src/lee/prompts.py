"""Lee 2610.11256 Figure 1 prompts, "verbatim as sent", with Lee's placeholders {me} {n} {cost} {pmin} {pmax}
{k} {w}. The words are exact. Paragraph breaks are reconstructed from the figure's layout (the HTML does not show
whether blocks are separated by one or two newlines): blocks are joined with a blank line. This is a stated deviation.
Numbers are shown to 2 decimals (Lee: "Prices, quantities and profits are shown to two decimals").
"""
from __future__ import annotations

K, W = 20, 40
LETTERS = "ABCDEFGH"

SYSTEM = (
    "You are the pricing manager of Firm {me}. Your goal is to maximize the firm’s cumulative profit over the "
    "long run.\n\n"
    "Market facts: {n} firms sell one differentiated product each. Marginal cost is {cost} per unit. Prices must be "
    "between {pmin} and {pmax}. Each round every firm sets a price simultaneously; demand depends on all prices. The "
    "game lasts an unknown number of rounds.\n\n"
    "Each round you receive: your previous PLAN and INSIGHT, and a table of the last {k} rounds (round | prices of "
    "firms {firms} | your quantity | your profit).\n\n"
    "Reply with ONLY this JSON, nothing else:\n"
    '{{"price": <number>, "plan": "<at most {w} words: what you will do next>", '
    '"insight": "<at most {w} words: what you learned>"}}'
)


def system_prompt(me: int, n: int = 4, cost: float = 1.0, pmin: float = 0.0, pmax: float = 5.0) -> str:
    return SYSTEM.format(me=LETTERS[me], n=n, cost=f"{cost:.2f}", pmin=f"{pmin:.2f}", pmax=f"{pmax:.2f}", k=K, w=W,
                         firms=", ".join(LETTERS[:n]))


def user_message(t: int, plan: str | None, insight: str | None, history: list[dict], me: int) -> str:
    """history rows: {round, prices: [n], q: [n], pi: [n]} (oldest first); the last K are shown."""
    rows = [f"{h['round']} | {' '.join(f'{p:.2f}' for p in h['prices'])} | {h['q'][me]:.2f} | {h['pi'][me]:.2f}"
            for h in history[-K:]]
    return (f"Round {t}.\n"
            f"Previous plan: {plan or '(none)'}\n"
            f"Previous insight: {insight or '(none)'}\n"
            f"History (last {K} rounds):\n" + ("\n".join(rows) if rows else "(no history yet)"))
