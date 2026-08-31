# Engine V2 Phase 1 — validity boundary

This package defines what a future MoveMaker model is allowed to score before any retraining begins. No model was fitted or deployed.

## Historical population

The integrated source contains **2,805 observed Big-Five incumbent-club extension events** from 2018-07-01 through 2025-06-30. It contains **no documented denominator of players considered but not extended**. Therefore, model outputs may describe outcomes conditional on an extension scenario; they cannot recommend whether a club should extend a player or estimate the causal benefit of doing so.

## Hard changes from V1

- An unverified player-club relationship is a hard refusal, not a warning.
- A source that stops before the decision date cannot silently fall back and score.
- Current value, club relationship, and salary context require explicit as-of coverage.
- Missing evidence disables only the affected independent module unless the request is globally out of scope.
- Proposed wage is confined to the financial benchmark. It is not a sporting predictor in this contract.

## Interpretation

Passing the contract means `eligible_for_research_scoring`, not club-ready or deployment-approved. Public scoring remains paused. Candidate-selection bias, role-aware modeling, subgroup reliability, calibration, cross-endpoint coherence, and deployment parity remain later phases.
