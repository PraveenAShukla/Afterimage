"""Load API keys and budget caps from .env (git-ignored). Keys are only ever read from os.environ.

A key exported by the launching shell (e.g. ~/.zshrc) silently overrode the project key in .env once (: a
paid wave billed the wrong OpenAI account). So any *_API_KEY set in BOTH places with DIFFERENT values is refused:
unset the shell one, or set ALLOW_SHELL_KEYS=1 to keep it deliberately. Values are never printed.
"""
from __future__ import annotations

import os

from dotenv import dotenv_values, load_dotenv

from .runlog import REPO


class KeyConflict(RuntimeError):
    pass


def key_conflicts(env_file=None) -> list[str]:
    vals = dotenv_values(env_file or REPO / ".env")
    return sorted(k for k, v in vals.items()
                  if k.endswith("_API_KEY") and v and os.environ.get(k) and os.environ[k] != v)


def load_env() -> None:
    bad = key_conflicts()
    if bad and os.environ.get("ALLOW_SHELL_KEYS") != "1":
        raise KeyConflict(f"shell environment overrides .env with a different value for {', '.join(bad)}; "
                          f"unset it (env -u {bad[0]} ...) or set ALLOW_SHELL_KEYS=1")
    load_dotenv(REPO / ".env", override=False)
