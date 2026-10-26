"""Block commits/pushes that contain API keys or private infrastructure names. Fails CLOSED.

Usage: python scripts/scan_secrets.py MODE
  --staged   the git index, i.e. exactly what will be committed (pre-commit hook)
  --all      every file in the HEAD commit (committed content, NOT the working tree)
  --push     every commit about to be pushed: reads git's pre-push stdin (`<lref> <lsha> <rref> <rsha>` lines);
             with no stdin, every commit not yet on any remote
What counts as a hit:
  - key-shaped strings (collusion.secrets.KEY_PATTERNS);
  - the literal value of any secret-named variable in the environment OR in .env (loaded here, so a custom token
    that does not look like a key is still caught);
  - any regex in internal/private/denylist.txt (git-ignored, so the list itself stays private). If that file is
    missing, the scan FAILS unless COLLUSION_ALLOW_NO_DENYLIST=1 (e.g. a public clone with nothing private).
  - .env, serve/site.env, internal/ and runs/ being committed at all.
No directory is skipped: test files build fake keys at runtime, so they never contain a literal key.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from collusion.secrets import scan_text  # noqa: E402

ZERO = "0" * 40
NEVER_COMMIT = (".env", "serve/site.env", "internal", "runs")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, errors="replace").stdout


def nul_split(out: str) -> list[str]:
    return [x for x in out.split("\0") if x]


def load_dotenv_literals() -> None:
    """Put .env secrets into os.environ for scan_text (hooks do not inherit them)."""
    env = REPO / ".env"
    if not env.exists():
        return
    for line in env.read_text(errors="replace").splitlines():
        m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*['\"]?([^'\"#\s]*)", line)
        if m and m.group(2):
            os.environ.setdefault(m.group(1), m.group(2))


def targets(mode: str) -> list[tuple[str, str]]:
    """(label, git object spec) pairs whose blob content is scanned."""
    if mode == "--staged":
        files = nul_split(git("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR"))
        return [(f, f":{f}") for f in files]
    if mode == "--all":
        if not git("rev-parse", "--verify", "-q", "HEAD").strip():
            return []
        return [(f, f"HEAD:{f}") for f in nul_split(git("ls-tree", "-r", "-z", "--name-only", "HEAD"))]
    if mode == "--push":
        commits: list[str] = []
        lines = [] if sys.stdin.isatty() else sys.stdin.read().split("\n")
        refs = [l.split() for l in lines if l.strip()]
        if not refs:
            commits = git("rev-list", "HEAD", "--not", "--remotes").split()
        for _lref, lsha, _rref, rsha in refs:
            if lsha == ZERO:
                continue  # deleting a remote branch pushes no content
            rng = [lsha, "--not", "--remotes"] if rsha == ZERO else [f"{rsha}..{lsha}"]
            commits += git("rev-list", *rng).split()
        out = []
        for c in dict.fromkeys(commits):
            changed = nul_split(git("diff-tree", "--root", "--no-commit-id", "-r", "-z", "--name-only", "--diff-filter=ACMR", c))
            out += [(f"{c[:8]}:{f}", f"{c}:{f}") for f in changed]
        return out
    sys.exit(f"unknown mode {mode}")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "--staged"
    load_dotenv_literals()
    deny_file = REPO / "internal/private/denylist.txt"
    bad: list[str] = []
    if deny_file.exists():
        deny = [re.compile(l.strip(), re.I) for l in deny_file.read_text().splitlines() if l.strip() and not l.startswith("#")]
    else:
        deny = []
        if os.environ.get("COLLUSION_ALLOW_NO_DENYLIST") != "1":
            bad.append("internal/private/denylist.txt missing: private-name check cannot run "
                       "(copy it from the private machine, or set COLLUSION_ALLOW_NO_DENYLIST=1 for a public clone)")
    items = targets(mode)
    for label, spec in items:
        path = label.split(":", 1)[-1]
        if any(path == p or path.startswith(p + "/") for p in NEVER_COMMIT):
            bad.append(f"{label}: must never be committed")
        text = subprocess.run(["git", "show", spec], cwd=REPO, capture_output=True).stdout.decode("utf-8", "replace")
        for hit in scan_text(text):
            bad.append(f"{label}: possible secret ({hit})")
        for d in deny:
            m = d.search(text)
            if m:
                bad.append(f"{label}: private name /{d.pattern}/ near {text[max(0, m.start() - 30):m.end() + 10]!r}")
    if bad:
        print("BLOCKED:\n  " + "\n  ".join(bad))
        return 1
    print(f"secret/private scan clean ({len(items)} blobs, {len(deny)} private patterns, mode {mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
