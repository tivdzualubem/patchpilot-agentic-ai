# Accepted Project Scope

## Goal

Repair small Python defects through a bounded
Plan–Act–Observe–Reflect–Verify loop.

## Required capabilities

- inspect a repository and failing tests;
- maintain structured agent state and execution budgets;
- choose restricted debugging tools dynamically;
- apply minimal source-code patches;
- run targeted and full regression tests;
- reflect and replan after unsuccessful attempts;
- roll back invalid or unsafe changes;
- terminate with verified success, bounded failure, or escalation;
- save auditable execution traces.

## Evaluation

The full agent will be compared with:

1. one-shot patch generation;
2. a fixed one-pass repair workflow;
3. a tool-using agent without reflection.

Primary metrics are complete repair rate and full regression-test pass rate.
Secondary metrics include invalid patches, tool calls, repair attempts,
latency, rollback frequency, patch size, and budget exhaustion.

## Scope controls

- Python repositories only;
- approximately 10–12 validated repair tasks;
- mutation-seeded defects reviewed before evaluation;
- one primary agent;
- no unrestricted shell access;
- no editing benchmark tests;
- no success claim without executable verification.
