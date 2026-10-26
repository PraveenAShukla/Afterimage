#!/usr/bin/env bash
# Primary (code) harness. Runs as pre-commit (staged scan) and pre-push (full scan). Any failure blocks.
#   scripts/check.sh            # full: lint + tests + full secret scan + ignore rules
#   scripts/check.sh --staged   # pre-commit: same, but scans only staged content for secrets
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=.venv/bin/python
echo "== lint (ruff: pyflakes, bugbear, undefined names)"
.venv/bin/ruff check src analysis tests scripts
echo "== tests"
$PY -m pytest -q -p no:cacheprovider tests | tail -1
echo "== secrets + private names"
$PY scripts/scan_secrets.py "${1:---all}"
echo "== ignore rules"
for p in .env serve/site.env internal/x runs/x; do git check-ignore -q "$p" || { echo "NOT IGNORED: $p"; exit 1; }; done
echo "ALL CHECKS PASSED"
