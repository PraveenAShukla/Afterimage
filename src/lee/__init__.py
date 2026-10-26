"""Lee 2610.11256 replication + randomised openings: an n-firm logit market played by LLM agents with
plan/insight memory, exactly as in Lee's Figure 1. Separate from the duopoly harness (src/collusion); it reuses only
collusion.budget (spend guard), collusion.runlog (append-only logs with secret redaction) and the HTTP retry helper.
Design: docs/LEE_REPLICATION.md.
"""
