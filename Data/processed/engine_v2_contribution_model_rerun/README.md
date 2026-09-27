# Engine V2 repaired contribution model — Run 2

Run 2 asks whether the repaired **sustained realized extension-club
contribution** target has a repeatable pre-extension signal. It does not ask
whether a club should extend a player and it does not measure tactical role
when available.

## Design

- Four expanding rolling origins evaluate extensions signed in 2020-2023.
- Evaluation players are removed from all model-fitting rows. Validation
  players are also removed from the training rows used to choose regularization.
- Brier score is the primary loss. Logistic `C` is chosen only on the preceding
  validation year.
- The fixed candidate ladder contains chronology prevalence, basic market
  profile, recent involvement, and optional prior-volume history.
- Goals/assists, advanced-event statistics, wages, proposed contract terms,
  manager/tactical fields, and post-extension predictors are excluded.

## Result

The predeclared primary candidate `B2_core_recent_involvement` has player-disjoint
pooled Brier improvement +0.054602 versus the
chronology baseline, with a 95% player-cluster interval
[+0.041976, +0.067080] and
4/4 origin wins. Its pooled ROC AUC is
0.762, average precision
0.704, adaptive ECE
0.035, and calibration gap
0.011.

Recent involvement adds +0.017690 Brier skill
over market profile alone. Adding prior-volume history changes Brier skill by
-0.000061 versus the primary candidate, with
a 95% interval [-0.002197, +0.002123].

The frozen decision is `advance_to_reliability_audit` and the selected
development candidate is `B2_core_recent_involvement`. This means only that it may enter the
next calibration/OOD/subgroup reliability audit. It is not a deployable model.

The preliminary subgroup table marks 13/14 views
as supported under the existing minimum-size, discrimination, calibration,
and Brier-skill rules. Those results are diagnostic, not a completed release
gate.

## Boundary

Run 2 fits models only in memory and writes no `.joblib` file. The 2024+
extension cohort remains closed, and every file under the current deployment
directory retains its exact pre-run hash. Role/squad, manager/tactical, club
behavior, nonlinear families, and calibration methods were not searched here;
they require separately frozen follow-up tests if the compact signal survives.
