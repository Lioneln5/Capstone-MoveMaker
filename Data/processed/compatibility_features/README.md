# MoveMaker lagged player and teammate-style compatibility features

## Scope

- Broad lagged feature rows: 905,269
- Advanced FBref compatibility rows: 17,790
- Strict prior-club eligible rows: 10,540
- Roster-conditioned eligible rows: 10,069
- Player-preference eligible rows: 7,378

## Timing policy

1. Broad player features use exact `t-1`, `t-2`, and `t-3` seasons only. Rolling features never include the target season.
2. Advanced player styles are standardized within source season and role, then lagged. Current-season advanced values are retained only as `target_` outcomes in the model matrix.
3. Strict club context uses the destination club's observed `t-1` squad style. If the focal player was already there, their contribution is removed.
4. Roster-conditioned context identifies target-season teammates, represents each teammate only with their `t-1` style, and removes the focal player. Because historical target-roster membership may not have been known at the prediction cutoff, these columns are scenario-only and must not be used in strict backtests.
5. Historical preference features use expanding prior seasons only. For each teammate-context dimension, the player's performance-versus-context slope is shrunk by `effective_n / (effective_n + 3)`.

## Style dimensions

Role-season standardized primitives feed eight interpretable dimensions: scoring threat, creation, progression, dribble aggression, passing control, directness, defensive intensity, and aerial physicality.

Compatibility includes direct player-style × teammate-context interactions, historical familiarity/preference distances, cosine similarity, and performance-conditioned preference slopes. These are features for a validated predictive model, not causal estimates or finalized scouting scores.

## Recommended split

- Train: season starts through 2021
- Validation: 2022
- Test: 2023

Never include columns prefixed `target_` in the predictor matrix.
