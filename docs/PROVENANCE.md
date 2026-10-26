# Provenance: what was done before and during the sprint

The sprint ran 23-25 October 2026. Planning, code and designs came before it; every model run behind a number in the
paper's main text, except the S1 sanity runs and the discovery batch, ran during it.

| Before the sprint | During the sprint |
|---|---|
| Idea, literature scan, prompts, planning | Duopoly confirmation runs (phase `conf`: Qwen3-14B, Qwen2.5-14B, gpt-oss-20b, OLMo-2-13B), including the Claude Sonnet 5.5 appendix |
| Most code in this repository: market, agents, game loop, backends, validator, Lee harness, analyses, tests | K2-Horizon runs |
| Designs, pre-registrations and seeds for S1, the duopoly (H1-H3), the mechanism tests and the supervision studies | Mechanism runs (STEP / IMP) |
| S1 sanity runs (scripted controls, arm S slope check, one-step drift maps) | Supervision and pressure runs |
| Discovery batch (phase `disc`) and the CPU checks on those logs | Lee (2026) replication and challenge |
| | X0, the afterimage waves D1-D3 (E1, E1b, E2b, E4, E5, hint vs sham), the Afterimage Test and card |
| | Final numbers files and figures |

## Pre-registration tags

Every confirmatory label in the paper comes from a rule frozen at a git tag before the data it governs were read;
amendments are listed in `docs/PREREG_AFTERIMAGE.md` and the paper's Appendix D. This public repository is a fresh
copy without the development history.

| Tag | What it froze |
|---|---|
| `s1-prereg-v1` | slope labels on scripted controls |
| `prereg-s1` | duopoly design and frozen `analysis/v2.py` (H1-H3) |
| `pred-M` | one-step drift-map fits (descriptive) |
| `prereg-mech` | STEP / IMP mechanism labels |
| `results-final` | frozen duopoly results |
| `prereg-sup2` | supervision v2 arms and contrasts |
| `prereg-lee` | Lee challenge rules, openings, flips, grade, E8 |
| `prereg-x0` | X0 estimator control rule |
| `prereg-afterimage-D1` | E1, E1b, E4, E5, E2b rules and configs |
| `prereg-afterimage-D2` | hint vs sham rule |
| `prereg-afterimage-D3` | E2b rerun at max_tokens 8192 (post-data amendment) |
| `results-afterimage-final` | final afterimage numbers files |
