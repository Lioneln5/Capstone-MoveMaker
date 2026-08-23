# Sustained-contribution model repair experiment

Two independent, mechanically-verified problems were found behind the suspicious live scores. Neither fix involves a manual superstar bonus, market-value floor, probability override, or presentation-only correction.

## Finding 1 -- league-encoding bug (fixed this session, in models/canonical_feature_retriever.py)

`canonical_feature_retriever.py` wrote the raw `competition_id` code (e.g. `GB1`) into the `league` feature. Every deployed model was fit on the Capology source's full league display name (e.g. `Premier League`). The fitted preprocessor's `__OTHER__` category has zero training-row support and is masked out at fit time (zero variance), so an unmatched live `league` value contributed **exactly zero** league signal on every live request, on every Phase 1-4 endpoint that uses `league` -- not just this one model. Confirmed directly against the deployed artifact's own `preprocessor.categories['league']`.

Demo-player impact (same C0/M5 model, before vs. after the fix, live request only -- no refit):

- Haaland: 0.710925 -> 0.720375 (delta +0.00945, league='Premier League')
- Mbappe: 0.642772 -> 0.703545 (delta +0.06077, league='La Liga')
- Bellingham: 0.694691 -> 0.750066 (delta +0.05538, league='La Liga')

## Finding 2 -- the deployed M5 still contains a failed feature block

`M3_vs_M2` pooled Brier improvement in the frozen Phase-1 diagnostic is -0.002066, 95% CI [-0.005595, 0.001495], 1/4 origin wins -- no validated incremental signal, yet M5 (=C0) still contains the entire performance-history block. Per-90 features in that block are heavy-tailed (goals-per-90 up to 2.4 on fewer than 40 minutes played) -- exactly the kind of noisy feature a single global regularized coefficient fits unstably.

## Finding 3 -- C3 was still counterintuitive; C5/C6 test removing pre365_goals_per90 specifically

C3 keeps pre365_goals_per90 (a *validated* M2 feature) and its controlled perturbation remained severely counterintuitive: increasing recent goals/90 from 0.0 to 1.0 on a fixed profile lowered the predicted probability from ~0.750 to ~0.673. **C5 excludes pre365_goals_per90 but retains pre365_assists_per90. C6 excludes both.** No further feature search was performed -- these two candidates only.

## Step 6 -- decision

- `C1_M2_only`: route=`none`, advances=False; vs C0: -0.003978, 95% CI [-0.008974, 0.001172], 0/4 origin wins, terminal-two -0.0037842922769778186, majority-cohort -0.0029628540456904627.
- `C2_M2_plus_contract`: route=`none`, advances=False; vs C0: -0.001960, 95% CI [-0.006464, 0.002501], 1/4 origin wins, terminal-two 8.699987529124356e-05, majority-cohort -0.0007807734324167076.
- `C3_M2_contract_relative_financial`: route=`none`, advances=False; vs C0: 0.002178, 95% CI [-0.000696, 0.004953], 3/4 origin wins, terminal-two -0.000717251317760038, majority-cohort 0.0017503495076975123.
- `C4_compact_nested`: route=`none`, advances=False; vs C0: 0.001676, 95% CI [-0.001232, 0.004393], 3/4 origin wins; vs C3: -0.000502 CI [-0.001628, 0.000640], terminal-two -0.0009546612091740898, majority-cohort 0.0009885694369383936.
- `C5_C3_minus_pre365_goals`: route=`compact_non_inferiority`, advances=True; vs C0: 0.002109, 95% CI [-0.000737, 0.004944], 3/4 origin wins; vs C3: -0.000069 CI [-0.000775, 0.000645], terminal-two -0.0006685358388384401, majority-cohort 0.001630490451532273.
- `C6_C3_minus_pre365_goals_assists`: route=`compact_non_inferiority`, advances=True; vs C0: 0.002246, 95% CI [-0.000896, 0.005349], 2/4 origin wins; vs C3: 0.000068 CI [-0.001559, 0.001743], terminal-two -0.0005962727127188298, majority-cohort 0.0019360537561147818.

**Best candidate: `C6_C3_minus_pre365_goals_assists` (advances=True, route=compact_non_inferiority).**

## Corrected injury findings (superseding, not modifying, Data/processed/contribution_model_refinement/)

Coverage is now decided by independent Big-Five appearance history, never by whether a player happens to appear anywhere in the injury CSV (the previous experiment's `player_id in lookup` check could mark a 2018 signing as injury-covered using a 2025 injury record). Games-missed is now prorated to the overlap fraction of each spell, not added in full for boundary-overlapping spells. Club consistency is now date-aware (+/-400 days around the injury), not whole-career. The target-decomposition diagnostic now requires confirmed coverage AND complete two-year post-signing observation AND no right-censoring before Year 2 ends -- see `corrected_injury_decomposition.csv` for the eligible denominator.

Read this descriptively, not as a clean separator: among the 314 completely-observed non-sustainers, 178 of 314 (56.7%) exceeded 30 injury-days, while 136 of 314 (43.3%) did not -- and sustained contributors themselves averaged 59.3 injury-days, above the 30-day threshold used to define the split above. The >30-day threshold mechanically guarantees the two non-sustained groups' means sit on opposite sides of it, so that gap alone is not evidence of separability; injury burden is descriptive context here, not a deterministic gate on sustained contribution.

See `model_comparison_results.csv`, `calibration_audit.csv`, `subgroup_audit.csv`, `missingness_audit.csv`, `coefficient_stability.csv`, `multicollinearity_audit.csv`, `goals_per90_perturbation_audit.csv`, `live_demo_probability_comparison.csv`, and `model_selection_decision.csv` for full detail before making any product claim.
