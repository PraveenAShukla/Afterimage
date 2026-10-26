"""Single source of truth check (plan v2 M16): recompute every committed numbers file from the raw logs and fail on
any difference. Run before every tag and before citing any number.

  .venv/bin/python analysis/verify.py            # exit 0 = every committed number reproduces from the logs

Checked: results/v2/<phase>/numbers.json (analysis/v2.py) for each phase present, the afterimage numbers files
(analysis/afterimage.py), and results/classify/heldout.json metrics (analysis/classify.py metrics only; runs already
on disk). Numbers are compared to 9 significant digits; bootstrap draws are seeded, so reruns are exact.

Needs the raw per-call logs (runs_cluster/), which are not in this repository. Without them, use
scripts/paper_table.py, which prints every paper number from the committed results/ files.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _cmp(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        errs = [f"{path}/{k}: missing" for k in set(a) ^ set(b)]
        return errs + [e for k in set(a) & set(b) for e in _cmp(a[k], b[k], f"{path}/{k}")]
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [e for i, (x, y) in enumerate(zip(a, b, strict=True)) for e in _cmp(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float):
        return [] if f"{a:.9g}" == f"{b:.9g}" else [f"{path}: {a!r} != {b!r}"]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _without(tmp: str, stages: tuple) -> Path:
    """runs_cluster as symlinks, minus the run dirs of stages added after a numbers file was frozen."""
    d = Path(tmp) / "runs_frozen"
    d.mkdir()
    for r in (ROOT / "runs_cluster").iterdir():
        if not any(r.name.startswith(f"{s}_") for s in stages):
            (d / r.name).symlink_to(r)
    return d


def _rerun(script: str, ref: Path, out: Path, args: list[str], label: str) -> list[str]:
    subprocess.run([PY, str(ROOT / script), *args, "--out", str(out)], check=True, capture_output=True)
    return _cmp(json.loads(ref.read_text()), json.loads((out / "numbers.json").read_text()), label)


def main() -> int:
    if not (ROOT / "runs_cluster").is_dir():
        print("raw logs are not included in this repository (runs_cluster/ is missing); they are available on request.\n"
              "To re-check the paper's numbers from the committed results files, run: python scripts/paper_table.py")
        return 2
    sys.path.insert(0, str(ROOT / "analysis"))
    from k2 import EXTRA_STAGES  # noqa: PLC0415
    errs, checked = [], []
    with tempfile.TemporaryDirectory() as tmp:
        frozen = _without(tmp, EXTRA_STAGES)
        for ph in ("disc", "conf"):
            ref = ROOT / "results" / "v2" / ph / "numbers.json"
            if not ref.exists():
                continue
            out = Path(tmp) / ph
            subprocess.run([PY, str(ROOT / "analysis/v2.py"), "--phase", ph, "--runs-dir", str(frozen),
                            "--out", str(out)], check=True, capture_output=True)
            e = _cmp(json.loads(ref.read_text()), json.loads((out / "numbers.json").read_text()), f"v2/{ph}")
            errs += e
            checked.append(f"results/v2/{ph}/numbers.json ({'OK' if not e else f'{len(e)} diffs'})")
        ref = ROOT / "results" / "v2" / "k2" / "numbers.json"
        if ref.exists():
            e = _rerun("analysis/k2.py", ref, Path(tmp) / "k2", [], "v2/k2")
            errs += e
            checked.append(f"results/v2/k2/numbers.json ({'OK' if not e else f'{len(e)} diffs'})")
        ai = ROOT / "results" / "afterimage"
        if ai.exists():
            dirs = [str(ROOT / "runs_cluster")] + [str(p) for p in sorted(ROOT.glob("runs_afterimage*")) + sorted(ROOT.glob("runs_ait_*")) if p.is_dir()]
            out = Path(tmp) / "afterimage"
            mc = ai / "x0" / "mc.json"            # the Monte Carlo half of X0 is an input here, not recomputed
            if mc.exists():
                (out / "x0").mkdir(parents=True)
                (out / "x0" / "mc.json").write_bytes(mc.read_bytes())
            r = subprocess.run([PY, str(ROOT / "analysis/afterimage.py"), *sum((["--runs-dir", d] for d in dirs), []),
                                "--out", str(out)], capture_output=True, text=True)
            if r.returncode:
                errs.append(f"afterimage: exit {r.returncode}: {r.stdout.strip()[-300:]}")
            for f in sorted(ai.glob("*/numbers.json")):
                g = out / f.parent.name / "numbers.json"
                e = _cmp(json.loads(f.read_text()), json.loads(g.read_text()), f"afterimage/{f.parent.name}") if g.exists() \
                    else [f"afterimage/{f.parent.name}: not recomputed"]
                errs += e
                checked.append(f"results/afterimage/{f.parent.name}/numbers.json ({'OK' if not e else f'{len(e)} diffs'})")
    sys.path.insert(0, str(ROOT / "analysis"))
    import classify  # noqa: E402
    ref = ROOT / "results" / "classify" / "heldout.json"
    if ref.exists():
        want = json.loads(ref.read_text())["metrics"]
        for pol in classify.POLICIES:
            got = classify.metrics(pol, "heldout", ROOT / "runs_cluster")
            e = _cmp({k: v for k, v in want[pol].items() if k in got}, got, f"classify/{pol}")
            errs += e
        checked.append(f"results/classify/heldout.json ({'OK' if not [x for x in errs if 'classify' in x] else 'diffs'})")
    for c in checked:
        print("checked", c)
    for e in errs[:30]:
        print("DIFF", e)
    print("VERIFY OK" if not errs else f"VERIFY FAILED: {len(errs)} differences")
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
