"""Golden test: scripts/paper_table.py must print exactly the numbers in the paper (frozen at results-afterimage-final).
If this fails, a change moved a paper number. Refresh tests/golden/paper_table.txt only on purpose."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_paper_table_matches_golden():
    out = subprocess.run([sys.executable, str(REPO / "scripts/paper_table.py")], capture_output=True, text=True, check=True).stdout
    assert out == (REPO / "tests/golden/paper_table.txt").read_text()


def test_headline_numbers():
    """The numbers the README and abstract quote, read from the same output."""
    out = (REPO / "tests/golden/paper_table.txt").read_text()
    for s in ("E5 full A_late         0.786", "E1 BR A1_late          0.001", "E1 COPY A1_late        0.839",
              "D2 D_hint              -0.861"):
        assert s in out, s
