# Role-contextualization diagnostic for goals/assists-per-90

Bounded, leakage-safe research diagnostic. **Nothing in this directory is
deployed.** No file under `Data/processed/deployment_models/`,
`models/deployment_scorer.py`, `api/`, `html/`, `slides/`, or
`Data/processed/contribution_model_repair/` was modified — see
`protected_artifacts_fingerprint_before.txt` /
`protected_artifacts_fingerprint_after.txt` and
`independent_verification.json`.

Runner: `models/run_role_contextualization_diagnostic.py`
Verifier: `scripts/verify_role_contextualization_diagnostic.py`
Random seed: `20260811` (project standard). Bootstrap: 5000 player-clustered
repetitions (project standard). Rolling origins: `origin_2020` …
`origin_2023`, identical train/validation/evaluation years to every other
Phase 1-4 diagnostic in this repo.

## 1. Which deployed endpoints actually use a goals/assists feature

Derived from the **deployed joblib artifacts themselves**
(`production_scoring_feature_inventory.csv`), not assumed. 12 of the 16
deployed artifacts contain at least one of
`pre365_goals_per90`, `pre365_assists_per90`, `pre1_canonical_goals_per90`,
`pre1_canonical_assists_per90`, `pre2_canonical_goals_per90`,
`pre2_canonical_assists_per90`:

- `year2_opportunity_share`
- `any_outbound` (24m **and** 36m — both read off the same hazard curve)
- `permanent_outbound` (**24m only** — no 36m artifact is deployed for this
  endpoint; a `permanent_outbound_36m` row appears in the research/backtest
  tables below purely as a research extension, never as a live endpoint)
- `value_log_ratio_12m`, `value_log_ratio_24m`
- `downside_10pct_12m`, `downside_10pct_24m`, `downside_25pct_12m`,
  `downside_25pct_24m`, `downside_50pct_24m`
- `contract_duration` — **not in the task's own "expected affected
  families" list.** Found by deriving from the artifacts rather than
  assuming: it's the deployed `B5_compact_nonlinear`
  (GradientBoostingRegressor) peer-benchmark model, and it carries the same
  six scoring features as every M5 linear endpoint.

**Confirmed out of scope, untouched:** `sustained_meaningful_contribution`
(the deployed **C6** artifact) contains zero scoring-rate features
(`variant=C6_C3_minus_pre365_goals_assists`). `annual_wage`,
`fixed_wage_commitment`, and `wage_to_market_value` (the `B2` peer-benchmark
family) also carry no scoring feature.

None of the 12 affected models interact scoring rate with position, and
none standardize scoring rates within position — `mixed_type_preprocessor
.fit_preprocessor` z-scores every numeric feature globally, so a scoring
rate's effect is identical regardless of the player's position. Only
**3 of 12** affected endpoints' scoring-rate coefficients are ever surfaced
in the live "why this score" panel at all (`downside_25pct_24m`,
`any_outbound`@24m, plus `sustained_contribution`, which itself excludes
scoring — so effectively **2** of the 12 affected endpoints ever show a
scoring-rate contribution to a user). The other 10 affected endpoints'
scoring-rate coefficients exist in the artifact and drive the number on
screen, but no explanation panel ever surfaces them.

## 2. Was Bellingham genuinely penalized?

**Yes, verified against the live production artifact**, not assumed.
`live_production_role_audit.csv` reconstructs every scoring-feature logit
contribution exactly (bit-for-bit, via the same linear decomposition
`models/feature_contribution_explainer.py` uses), for the actual deployed
`any_outbound` (Contract-Horizon Stability / Continuity) artifact scoring
the real Jude Bellingham (Real Madrid, Midfield) profile as of today:

| feature | logit contribution to hazard (departure) |
|---|---|
| `pre365_goals_per90` | **+0.0734** |
| `pre1_canonical_goals_per90` | -0.0093 |
| `pre2_canonical_goals_per90` | **+0.0712** |
| `pre365_assists_per90` | +0.0168 |
| `pre1_canonical_assists_per90` | +0.0141 |
| `pre2_canonical_assists_per90` | +0.0126 |

**Read direction carefully**: `_score_hazard`'s model target is the
*hazard* (departure) probability; the displayed continuity/stay figure is
`1 - cumulative hazard`. A **positive** logit contribution to the hazard
therefore **raises predicted departure risk and lowers the displayed
Contract-Horizon Stability number** — the opposite of "favorable." Every
one of Bellingham's recent/prior scoring-rate features pushes his hazard up
except one small prior-season term. This reproduces the user's observation
exactly, from the real deployed artifact, not a re-derived approximation.

## 3. Global or position-aware?

**Global, and stably so.** `goals_per90_origin_sign_stability.csv` refits
the raw `pre365_goals_per90` coefficient independently at each of the 4
rolling origins for every affected endpoint. The sign is **identical across
all 4 origins** for 9 of 12 endpoints — including both hazard endpoints and
`year2_opportunity_share`:

| endpoint | sign, all 4 origins | coefficient range |
|---|---|---|
| `year2_opportunity_share` | negative | -0.0237 to -0.0187 |
| `any_outbound` | **positive** (raises departure hazard) | +0.0171 to +0.0555 |
| `permanent_outbound` | **positive, and *larger* than any_outbound's** | +0.0500 to +0.0732 |
| `downside_10pct_12m`, `downside_25pct_12m`, `downside_25pct_24m`, `downside_50pct_24m` | negative (protective) | -0.114 to -0.0057 |
| `value_log_ratio_12m`, `value_log_ratio_24m` | **flips sign** (positive 2020-21, negative 2022-23) | -0.021 to +0.065 |

This is not one bad training split — it is a structural property of the
current global (non-role-aware) linear specification, present across every
origin tested.

**On the "outbound demand, not low contribution" hypothesis** (explicitly
required by the task, not assumed either way): the effect is **not**
concentrated in temporary moves. `permanent_outbound`'s coefficient is
consistently *larger* than `any_outbound`'s at every origin (e.g.
origin_2023: +0.0587 vs +0.0480). If the relationship were mostly loan/
squad-rotation noise, `any_outbound` (which includes loans) should show the
larger effect. It does not. This is more consistent with genuine transfer/
sale demand for high-scoring players (a real market dynamic) than with
benign temporary-move noise — but this diagnostic cannot separate "the club
sells its best players" from "the model has learned a market pattern that
happens to fire on Bellingham specifically." Both are plausible; this
diagnostic does not adjudicate between them, and no manual correction was
applied either way.

## 4. Position and sub-position coverage

`role_coverage_audit.csv` / `role_scoring_distribution_audit.csv`
(2,805-row master frame, `pre365_goals_per90`):

| broad position | n rows | non-missing | missing rate | median | zero rate |
|---|---|---|---|---|---|
| Attack | 591 | 500 | 15.4% | 0.336 | 15.4% |
| Midfield | 784 | 651 | 17.0% | 0.062 | 35.3% |
| Defender | 931 | 820 | 11.9% | 0.026 | 48.0% |
| Goalkeeper | 420 | 265 | **36.9%** | **0.000** | **99.2%** |
| unmapped (missing position) | 79 | 0 | 100% | — | — |

Goalkeepers are essentially zero-variance on this feature (99.2% exactly
zero) — confirms the task's own warning and is why Goalkeeper is excluded
from every interaction term (R3/R4) rather than getting a spurious,
uninformative coefficient.

`canonical_sub_position` (`role_sub_position_coverage.csv`) has 14 distinct
values. Several are far below any usable training threshold once split
across 4 rolling origins: `Second Striker` (28 total), `Right Midfield`
(24), `Left Midfield` (13), `Midfield` (1, a data-entry variant of
"Midfield" collapsed into the broad Midfield group for this diagnostic —
see `_position_series()`). Per the task's own instruction, sub-position was
**audited, not adopted**: `MIN_ROLE_TRAIN_N = 30` and
`role_origin_stability_audit.csv` confirm several broad-position groups
themselves dip close to that floor at the earliest origin (e.g. origin_2020
Goalkeeper), which would only get worse fragmenting further into
sub-positions. Broad `canonical_position` (Attack/Midfield/Defender/
Goalkeeper) is used as required.

## 5. R0-R4 comparison by endpoint

Full numbers: `origin_model_metrics.csv` (per-origin), `pooled_model_
comparison.csv` (pooled + player-clustered bootstrap, `pooled_four` and
`terminal_two` windows), `position_subgroup_metrics.csv`,
`calibration_audit.csv`. Candidate feature lists: `candidate_feature_
manifest.csv`. R0 reproduces the deployed raw feature list exactly (see
`independent_verification.csv`'s `R0_feature_lists_match_deployed_
artifacts` check). R2's role-relative method (robust within-position
z-score vs. training-only percentile) and R4's representation
(role-relative vs. interactions) are selected **inside each origin**, on
that origin's own validation split only (`role_method_choices.csv`,
`r4_representation_choices.csv`) — never using the evaluation year.

High-level pattern: R1 (remove scoring entirely) is non-inferior-to-better
almost everywhere and, for the two hazard endpoints specifically, is the
*cleanest* and most consistent winner (see §7). R2/R4 (role-relative) help
the continuous/binary M5 endpoints (`year2_opportunity_share`,
`downside_*`) modestly. R3 (raw + position interactions) underperforms R0
on the pooled terminal-two window for nearly every endpoint tested — adding
interaction terms without also decorrelating the raw scoring feature mostly
adds variance, not signal, at this sample size.

## 6. Did role contextualization improve out-of-time performance?

Modestly, and unevenly — not a clean uniform win. See §8's decision table.
Where a role-aware candidate (R2/R3/R4) was selected, the terminal-two-year
pooled improvement is small (typically <0.003 Brier/MAE) and several
individual position subgroups show a *slight* degradation that stayed
inside this diagnostic's 10%-relative-degradation gate without tripping it
(e.g. `downside_25pct_24m`: Defender Brier 0.1523 → 0.1564 under R4, a
2.6% relative increase — see `position_subgroup_metrics.csv`). This is
disclosed, not hidden, because the task explicitly requires reporting
subgroup effects even when they don't block a decision.

## 7. Was removing scoring features better?

**For the two hazard/continuity endpoints, yes — more convincingly than
role-awareness.** `pooled_model_comparison.csv`, `pooled_four` window:

| endpoint | candidate | pooled loss improvement | origin wins |
|---|---|---|---|
| `any_outbound_24m` | R1 (remove) | **+0.00160** | 2/4 |
| `any_outbound_24m` | R4 (role-aware) | +0.00081 | 2/4 |
| `permanent_outbound_24m` | R1 (remove) | **+0.00096, wins 3/4 origins** | **3/4** |
| `permanent_outbound_24m` | R4 (role-aware) | -0.00001 (flat) | 2/4 |

R1 beats R4 on both the pooled metric and origin-win count for both hazard
endpoints. The automated decision gate (§8) still labeled these two
`ADVANCE_ROLE_AWARE` because R4 also cleared the same non-inferiority bar
(a declared, disclosed threshold, matching the project's existing C6
precedent) — **this is flagged explicitly as a case where a human reviewer
should weight R1 over the gate's literal pick** for the continuity family.
For `year2_opportunity_share` and the `downside_*` family, R1 is close to
R2/R4 but does not clearly beat them, so no similar override concern
applies there.

## 8. Demo player score changes (Bellingham, Haaland, Mbappé)

`demo_player_role_comparison.csv` / `demo_player_role_contributions.csv`,
scored against each candidate's own deployment-equivalent final refit
(train ≤2022 + validation 2023, matching the real deployment methodology).
Continuity (`any_outbound`, 24m stay probability — higher is more stable):

| player | position | R0 (actual production) | R1 (remove scoring) | R2 (role-relative) | R4 (compact role-aware) |
|---|---|---|---|---|---|
| Jude Bellingham | Midfield | 58.1% | 63.1% (**+5.0pp**) | 65.4% (**+7.3pp**) | 59.4% (+1.3pp) |
| Erling Haaland | Attack | 39.2% | 52.6% (**+13.4pp**) | 51.3% (**+12.1pp**) | 31.1% (**-8.1pp**) |
| Kylian Mbappé | Attack | 40.8% | 59.1% (**+18.3pp**) | 58.1% (**+17.2pp**) | 32.9% (**-7.9pp**) |

**This is a genuinely important, non-obvious result, reported honestly
rather than smoothed over**: R1 and R2 both raise all three players'
continuity substantially and consistently. **R4 does not** — it raises
Bellingham's very slightly but *lowers* Haaland's and Mbappé's continuity
further below the current production number. R4 was selected per-origin
purely by validation-set loss, with no player-specific tuning, so this is
real evidence that a role-aware candidate chosen by aggregate validation
performance does not automatically "fix" the effect for the specific
players the concern was raised about — confirming the task's explicit
instruction not to present a role-aware score as correct merely because it
looks more intuitive.

`downside_25pct_24m` (asset-value risk — lower is more favorable) moves the
same direction for all three players under every scoring-aware candidate
(R1/R2/R4 all lower it, e.g. Mbappé 89.4% → 86.7-87.5%), i.e. more
consistent than the continuity result.

Representative defender (Virgil van Dijk) and goalkeeper (Thibaut
Courtois) were also scored (`live_production_role_audit.csv`): van Dijk's
`pre365_goals_per90` contribution to `any_outbound` hazard is small and
positive (+0.029, same unfavorable direction as Bellingham, smaller
magnitude, consistent with defenders' lower/less-variable scoring rates);
Courtois' is small and *negative* (-0.030, favorable) on `any_outbound`
despite his `pre365_goals_per90` being exactly 0 — this reflects the
model's fitted intercept/interaction with his other features through the
global (non-position-aware) preprocessing, not a meaningful goalkeeper-
specific scoring effect (see §4's zero-variance note).

## 9. Final decision per endpoint

`model_selection_decision.csv`. Decision gates (all pre-declared, computed
identically for every endpoint — no per-endpoint tuning): a candidate
**ADVANCE**s only if it is non-inferior on the terminal-two-year pooled
loss (within 0.005), wins at least half of the 4 rolling origins on the
pooled-four window, its bootstrap CI lower bound doesn't indicate >0.01
harm, and it doesn't degrade any adequately-sized (n≥30) position subgroup
by more than 10% relative. This script does not deploy anything regardless
of the outcome.

| endpoint | decision | chosen candidate |
|---|---|---|
| `year2_opportunity_share` | ADVANCE_ROLE_AWARE | R4 (compact role-aware) |
| `any_outbound_24m` | ADVANCE_ROLE_AWARE (see §7 override note) | R4 |
| `any_outbound_36m` | ADVANCE_ROLE_AWARE (see §7 override note) | R4 |
| `permanent_outbound_24m` | ADVANCE_ROLE_AWARE (see §7 override note) | R4 |
| `permanent_outbound_36m` *(research-only, not a live endpoint)* | REMOVE_SCORING_FEATURES | R1 |
| `value_log_ratio_12m` | ADVANCE_ROLE_AWARE | R2 (position-relative) |
| `value_log_ratio_24m` | ADVANCE_ROLE_AWARE | R2 |
| `downside_10pct_12m` | REMOVE_SCORING_FEATURES | R1 |
| `downside_10pct_24m` | ADVANCE_ROLE_AWARE | R4 |
| `downside_25pct_12m` | ADVANCE_ROLE_AWARE | R4 |
| `downside_25pct_24m` | REMOVE_SCORING_FEATURES | R1 |
| `downside_50pct_24m` | ADVANCE_ROLE_AWARE | R2 |
| `contract_duration` | **RETAIN_CURRENT_PENDING_MORE_DATA** | — |

`contract_duration` retains current pending more data because it is the
sole GradientBoosting (nonlinear) endpoint: it has no linear coefficient to
role-adjust the same way, its closed-form contribution decomposition
doesn't apply (see §11), and none of R1-R4 cleared the non-inferiority gate
on the terminal-two window for it.

## 10. Limitations and prohibited claims

- This is a **research diagnostic**. No candidate here is deployed, no
  target definition was changed, no chronological cohort was altered, and
  no future information was used (verified — `independent_verification
  .csv`'s `no_future_information_in_any_candidate_feature_list` check
  scans every saved candidate feature list for any `post_y*`/`target_*`
  column and finds none).
- `permanent_outbound_36m` numbers in the backtest tables are a research
  extension only — there is no deployed 36m artifact for that endpoint;
  do not read it as describing a live product number.
- The value-preservation family's sign instability (§3) means any role-
  aware treatment there should be read as more provisional than the hazard
  or opportunity families.
- Do **not** treat this diagnostic's decision table as an approval to
  deploy — every row explicitly requires human review, and §7 identifies a
  case where the literal gate output (R4) is arguably not the strongest
  candidate (R1) by the same evidence.
- Do not present any role-aware candidate as "the fix" for an individual
  player — §8 shows the same role-aware candidate helped one demo player
  and hurt two others.
- The GK "favorable" `any_outbound` contribution reported in §8 for
  Courtois should not be read as "the model understands goalkeepers
  reward"; his scoring rate has essentially zero variance (§4) and the
  contribution reflects the fitted global coefficient applied to a
  structurally-uninformative input, not a validated goalkeeper-specific
  relationship.
- A genuine, previously-undetected bug was found (not fixed in shared
  code — see below) in `models/feature_contribution_explainer.py`.

### Bug found, not fixed (out of scope per the isolation requirement)

`feature_contribution_explainer.raw_contributions()` calls
`model.intercept_[0]`, which is correct for `LogisticRegression`
(`intercept_` is always array-like) but raises `IndexError` for `Ridge` on
a continuous target (`intercept_` is a plain Python float there — no `[0]`
to take). This has never fired in production because
`deployment_scorer.explain_headline_scores()` only ever calls it for the
three binary/hazard cards (`sustained_contribution`, `downside_25pct_24m`,
`any_outbound`@24m) — never for a continuous Ridge endpoint like
`year2_opportunity_share` or `value_log_ratio_*`. This diagnostic is the
first caller to exercise that path and hit it. **Not patched in the shared
file**, per this diagnostic's isolation requirement; this script uses a
local, byte-identical-except-for-the-fix copy
(`run_role_contextualization_diagnostic.raw_contributions()`) instead.
Recommend a real one-line fix (`np.ravel(model.intercept_)[0]`) be applied
to `models/feature_contribution_explainer.py` by a follow-up, reviewed
change — not bundled into this diagnostic.

## 11. Files created

All under `Data/processed/role_contextualization_diagnostic/` unless noted.
New source files: `models/run_role_contextualization_diagnostic.py`,
`scripts/verify_role_contextualization_diagnostic.py`. See
`output_manifest.csv` for a hash of every file listed below, and
`candidate_artifacts_manifest.csv` for the 44 saved (endpoint, candidate)
research artifacts under `candidate_artifacts/` (R1-R4 only — R0 is a
reproduction check, not a new artifact, so it is not re-saved), each
labeled `not_deployed_pending_review`.

`README.md`, `run_summary.json`, `source_manifest.csv`,
`production_scoring_feature_inventory.csv`, `role_coverage_audit.csv`,
`role_scoring_distribution_audit.csv`, `role_origin_stability_audit.csv`,
`role_sub_position_coverage.csv`, `role_reference_fallback_log.csv`,
`role_method_choices.csv`, `r4_representation_choices.csv`,
`candidate_feature_manifest.csv`, `origin_model_metrics.csv`,
`pooled_model_comparison.csv`, `position_subgroup_metrics.csv`,
`calibration_audit.csv`, `role_specific_perturbation_audit.csv`,
`goals_per90_origin_sign_stability.csv`, `live_production_role_audit.csv`,
`demo_player_role_comparison.csv`, `demo_player_role_contributions.csv`,
`model_selection_decision.csv`, `candidate_artifacts_manifest.csv`,
`protected_artifacts_fingerprint_before.txt`,
`protected_artifacts_fingerprint_after.txt`, `independent_verification.csv`,
`independent_verification.json`, `output_manifest.csv`.

**Note on `feature_contribution_explainer.py`**: imported read-only for
`label_for`/`sigmoid` only. The actual per-feature contribution
reconstruction used throughout this diagnostic (Phase B, Phase G) is a
local copy inside `run_role_contextualization_diagnostic.py` with one bug
fix (§10). `models/feature_contribution_explainer.py` itself was never
written to — confirmed by direct re-read after the run, byte-identical to
what this diagnostic read at the start. It is not on the task's literal
protected-file list (which names `models/deployment_scorer.py` explicitly
but not this file), so it is not part of the before/after hash manifest;
its unchanged status is confirmed separately here rather than via that
manifest.

## 12. Confirmation

No production artifact, API route, HTML file, slide, or `Data/processed/
contribution_model_repair/` file was changed. No git operation was run.
`protected_artifacts_fingerprint_before.txt` and `..._after.txt` are
byte-identical (`independent_verification.json`'s
`protected_artifacts_unchanged` check). All new work is confined to
`Data/processed/role_contextualization_diagnostic/`,
`models/run_role_contextualization_diagnostic.py`, and
`scripts/verify_role_contextualization_diagnostic.py`.
