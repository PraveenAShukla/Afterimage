"""Print the paper's frozen-value table straight from the committed numbers files (offline; no runs needed)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results" / "afterimage"


def main() -> int:
    L = lambda e: json.loads((R / e / "numbers.json").read_text())  # noqa: E731
    f = lambda a: f"{a['A']:.3f} [{a['ci90_within'][0]:.3f}, {a['ci90_within'][1]:.3f}]"  # noqa: E731
    g = lambda d: f"{d['d']:+.3f} [{d['ci90'][0]:+.3f}, {d['ci90'][1]:+.3f}]"  # noqa: E731
    rows = []
    e5 = L("e5")
    rows += [("E5 full A_late", f(e5["stages"]["ait-e5-full-qwen3off"]["A"]["p"]["121-170"])),
             ("E5 D_PO", g(e5["contrasts"]["D_PO"]["diff"]["121-170"]))]
    e1 = L("e1")
    rows += [("E1 BR A1_late", f(e1["stages"]["ait-e1-br-qwen3off"]["A"]["p1"]["121-170"])),
             ("E1 COPY A1_late", f(e1["stages"]["ait-e1-copy-qwen3off"]["A"]["p1"]["121-170"])),
             ("E1 D_BR", g(e1["contrasts"]["D_BR"]["diff"]["121-170"]))]
    rows += [("E1b A1_late", f(L("e1b")["stages"]["ait-e1b-nash-qwen3off"]["A"]["p1"]["121-170"]))]
    s = L("e4")["stages"]
    rows += [(f"E4 {m} A_late", f(s[f"ait-e4-{m}"]["A"]["p"]["121-170"])) for m in ("haiku55", "sonnet55", "gpt61sol")]
    d2 = L("d2")
    rows += [("D2 D_hint", g(d2["contrasts"]["D_hint"]["diff"]["121-170"]))]
    for k, v in rows:
        print(f"{k:22s} {v}")
    for e in ("e1", "e1b", "e4", "e5", "d2", "e2b"):
        print(e, "labels:", L(e)["rules"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
