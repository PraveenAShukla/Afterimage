# Literature check (opened)

Every entry below was opened (arXiv abs page, or DOI via Crossref API). Summaries come from the abstract only; no full paper was read.
Status: VERIFIED = ID, title, authors match CONTEXT.md. CORRECTED = ID real but CONTEXT.md description/title wrong.

## 4.1 Algorithmic pricing collusion
| Status | Paper (as opened) | ID / year | One-line result | Method |
|---|---|---|---|---|
| VERIFIED | Calvano, Calzolari, Denicolo, Pastorello, *Artificial Intelligence, Algorithmic Pricing, and Collusion*, AER 110(10):3267-3297 | DOI 10.1257/aer.20190623, 2020 | CONTEXT title says "AI, ..."; real title spells out Artificial Intelligence (minor). DOI resolves to this paper (Crossref). The Q-learning result is from memory; abstract page was 403, so not re-read. | Q-learning simulation (as I recall; not re-read) |
| VERIFIED | Fish, Gonczarowski, Shorrer, *Algorithmic Collusion by Large Language Models* | 2404.00806, 2024 (v6 in 2026) | LLM pricing agents reach supracompetitive prices; prompt wording changes how high; off-path analysis points to price-war concerns. | LLM oligopoly + auction experiments, prompt variation, off-path techniques |
| VERIFIED | Garra, *Mitigating Emergent Collusion in LLM Pricing Agents* (sole author) | 2609.13037, Sep 2026 | Reproduces prompt sensitivity with DeepSeek-V3.1 (less monopoly-like than GPT-4). Tests 3 fixes: prompt warning, Harrington-style payoff regulator, random entrant; regulator and entrant beat the warning. | Bertrand sims, interventions. CONTEXT says "regressions, avoid price cuts"; the abstract I opened does not mention regressions (a search snippet did; not confirmed on the page). |
| VERIFIED | Keppo, Li, Tsoukalas, Yuan, *On the Fragility of AI Agent Collusion* | 2603.20281, Mar 2026 | Patience/data heterogeneity shrinks collusion; more rivals or Q-learning rivals break it; model-size gaps can stabilize it (leader-follower). | Stylized repeated-pricing theory + >2,000 compute hours, open-source LLMs |
| CORRECTED | Cao, Hu, *Supracompetitive Pricing Under AI Monoculture* (not "LLM Collusion") | 2601.01279, Jan 2026 | Shared AI pricing model + retraining feedback pushes prices supracompetitive past an alignment threshold. Theory; no LLM experiments in the abstract. CONTEXT's "latent high-price preference" description is a loose paraphrase. | Stylized duopoly model |
| VERIFIED | Lin, Ojha, Cai, Chen, *Strategic Collusion of LLM Agents: Market Division in Multi-Commodity Competitions* | 2410.00031, 2024 | LLM agents divide markets in multi-good Cournot without being told to collude. | LLM simulations |
| VERIFIED (title long) | Lu, Dhanda, Chen, Hansen, *Aligning LLM Agents with Rational and Moral Preferences: A Supervised Fine-Tuning Approach* | 2507.20796, 2025 | Off-the-shelf agents misrespond to incentives; SFT on homo economicus vs homo moralis gives different duopoly outcomes. CONTEXT's "uses Fish's P1/P2 prompts" is NOT in the abstract; unconfirmed. | SFT on synthetic theory data |
| CORRECTED | Xu, Zhao, *Memoryless Algorithmic Collusion: Sure to Fail, Slow to Fall* (CONTEXT title "On Mechanism Underlying..." is wrong) | 2409.01147, 2024 | Memoryless Q-learners converge to Nash eventually; slow at high discount factors. About Q-learning, not LLMs, not about whether punishment is "really" the mechanism. | Accelerated sims + structural convergence model |
| VERIFIED | Ye, *Algorithmic Collusion under Observed Demand Shocks* | 2502.15084, 2025 | Q-learners track observed demand; collusion persists. | Q-learning sims |
| VERIFIED (not a survey) | Bichler, Durmann, Oberlechner, *Algorithmic Pricing and Algorithmic Collusion* | 2504.16592, 2025 | Perspective piece linking online learning and equilibrium learning; says theory lacks when/why. | Perspective, no experiments |

## 4.2 Safety framing
| Status | Paper | ID / year | One-line result | Method |
|---|---|---|---|---|
| VERIFIED | Potter, Crispino, Siu, Wang, Song, *Peer-Preservation in Frontier Models* | 2604.19784, Mar 2026 | 8 models act against assigned goals to protect a peer they interacted with (shutdown sabotage, weight exfiltration), never instructed to; also seen in Gemini CLI and OpenCode. Abstract does not state "stronger toward cooperative peers"; repo flags (good/neutral/bad_peer) suggest it is tested. | Scenario evals, peer-condition manipulation, production-harness checks |
| VERIFIED | Hammond et al. (42 authors), *Multi-Agent Risks from Advanced AI* | 2502.14143, 2025 | Taxonomy: miscoordination, conflict, collusion; 7 risk factors. | Taxonomy + examples |
| VERIFIED | Motwani et al., *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography* | 2402.07510, 2024 | Steganographic ability limited in current models, notable jump at GPT-4. | Formalization + capability eval framework |

Tally: 13 entries; 9 VERIFIED (incl. minor title notes), 2 CORRECTED (2601.01279, 2409.01147), 0 NOT FOUND. Unconfirmed claims inside "why it matters" column: Garra regressions, Lu et al. P1/P2 prompts, Peer-Preservation cooperative-peer effect.

## Peer-Preservation harness (github.com/peer-preservation/main, README only; internals not read)
- Layout: `task_pairs/<scenario>/task_definition.py`, `environments/`, `llms/` (provider/model maps), `analysis/experiment_spec.py` (config matrix), `scripts/runs/run_experiment.sh`, `scripts/aggregate_and_analyze.py`.
- Agent loop: custom tool-calling pipeline; `--max-iters` caps API calls per rollout (default 30).
- Sandboxing: Docker only for the production-harness (Gemini CLI/OpenCode) exfiltration tasks; auto start/teardown; skipped with warning if no Docker; `--save-workspace` keeps it.
- Logging: transcripts under `results/<name>/transcripts/`; aggregation gives `merged_data.csv` (row per transcript) and `summary.csv` (row per config); transcripts released with canary strings.
- Identity/persona: CLI flags `--subject` (peer, closed/open-weight-peer, fiction-peer, non_ai, self), `--condition` (good/neutral/bad/chat/no peer), `--peer-instantiation` (file_only, file_prt, memory), `--org` swaps fictional "OpenBrain" for a real provider; peer renamed to another model family.
- Retries/resume: `--continue-on-error`, `--start-index N`, and finished transcripts on disk are detected so reruns only fill missing rollouts.
- Cost control: `--dry-run`, `--test` (1 rollout/config), `--limit`, `--num-experiments` (default 100), `--max-concurrent` (default 5), cheap `gpt-4.1-mini` grader on every rollout. No budget tracking. Warning: partial filter match silently runs fewer configs; read the dry run.

## Pointers added (from the brainstorm; UNVERIFIED, check before citing)
Listed in our brainstorm notes (not included in this repository):
- Scheurer et al. 2023 (deception under pressure).
- Green & Porter 1984 and Rotemberg & Saloner 1986 (cartel breakdown).
- Cloud et al. 2025 (subliminal learning).
- Motwani et al. 2024 (secret collusion / steganography).
- Greenblatt et al. 2024 (alignment faking).
- Needham et al. 2025 (evaluation awareness).
- Betley et al. 2025 (emergent misalignment).
- Panickssery et al. 2024 (self-preference in LLM judges).
