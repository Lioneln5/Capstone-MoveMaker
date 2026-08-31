# Engine V1 freeze and Phase 0 containment

## Status

Engine V1 is preserved as the presentation-era **research prototype** and is
not the active basis for new production claims. Its 24 deployed artifacts and
verification records were frozen on 2026-08-30 from commit
`fa0d4e3cb437e772cf0dabb3f6a6cf592e8b07ee`.

The exact byte-level inventory is recorded in
[`docs/releases/engine_v1_freeze_2026-08-30.json`](releases/engine_v1_freeze_2026-08-30.json).
Verify it at any time with:

```bash
python scripts/verify_engine_v1_freeze.py
```

The verifier fails on a changed, missing, or unexpected file beneath
`Data/processed/deployment_models/` (excluding the historical `_backups`
directory). V2 experiments and artifacts must therefore live outside that
directory.

## Public containment

The application now fails closed. Unless an operator explicitly sets
`MOVEMAKER_ENABLE_SCORING=true`:

- the API does not load serialized V1 models or large runtime tables;
- player search, input-default, and profile endpoints return HTTP 503;
- `/health` remains HTTP 200 and reports `research_prototype` with
  `scoring_enabled: false`; and
- the hosted site remains available as a portfolio and research record, with
  an unmistakable scoring-paused banner.

This is not a cosmetic disclaimer. It prevents the presentation-era engine
from continuing to emit live player probabilities while the validity issues
identified in the post-presentation stress test remain unresolved.

## Controlled V1 access

V1 can still be enabled for local regression testing or a deliberately
controlled historical demonstration:

```bash
MOVEMAKER_ENABLE_SCORING=true \
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Enabling V1 requires the documented runtime data and does not change its
research-prototype status. It must not be described as a club-ready decision
engine.

## V2 isolation policy

Engine V2 work must satisfy all of the following:

1. use newly versioned experiment and artifact directories, never overwrite
   `Data/processed/deployment_models/`;
2. preserve chronological evaluation and the true decision-time information
   boundary;
3. refuse unsupported player, club, league, date, horizon, and missing-input
   states rather than silently extrapolating or substituting;
4. evaluate candidate-selection bias, role-aware feature specification,
   subgroup reliability, calibration, and cross-endpoint coherence explicitly;
5. keep proposed commercial terms out of sporting-outcome models unless a
   causal and decision-time-valid justification survives a dedicated ablation;
6. produce a new manifest, independent verifier, API contract, model card, and
   deployment parity report before any live scoring is restored; and
7. require an explicit deployment decision—passing an experiment does not
   automatically promote a model.

Phase 0 changes containment and versioning only. It does not repair, retrain,
or reinterpret any V1 model.
