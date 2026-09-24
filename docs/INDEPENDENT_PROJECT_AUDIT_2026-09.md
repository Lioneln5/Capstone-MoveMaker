# MoveMaker — independent technical audit (September 2026)

Audit date: 2026-09-09. Auditor: independent, adversarial, no prior project context.
Scope: repository `Lioneln5/Capstone-MoveMaker` at `main` HEAD `fa0d4e3` (2026-08-30), the unmerged branch `origin/codex/engine-v2-safety-freeze` (HEAD `80cb0a0`, 2026-09-02), the local data lake, the frozen artifacts, and the hosted service at `movemaker-production.up.railway.app`.

Machine-readable evidence for every quantitative statement below is under `Data/processed/independent_project_audit_2026_09/` (see its `README.md`). Nothing in the repository was modified; all reconstruction ran in isolated scripts under that directory. The frozen `deployment_models/` inventory was hash-checked before and after the audit and is unchanged.

---

## 1. Executive verdict

MoveMaker is a carefully documented research programme whose *historical* evidence is largely reproducible, but whose *product* claims on the public default branch are no longer true and whose *deployed* V1 scoring specification is not decision-time valid.

Six findings drive the verdict.

1. **The repository has three different states.** The default branch (`main`), its README, model card and handoff describe a live 16-endpoint scoring demo. The hosted service actually runs code from an unmerged branch that disables scoring, freezes V1 as a "research prototype", retires the continuity endpoint, and records two open release blockers. None of that is visible on `main`. Leftover branch outputs sit untracked in the working tree without their source. (MM-01, critical)
2. **Twelve of the sixteen V1 artifacts use the user's own proposed wage and term as predictors of sporting and public-value outcomes.** Holding a real player fixed, moving the proposed term from 2 to 6 years raises the mean displayed "sustained contribution" probability from 0.53 to 0.71 and "continuous stay" from 0.45 to 0.75, and lowers "25% value downside" from 0.54 to 0.37. Tripling the proposed wage has similar effects. In training these fields were the *realized* terms, which encode the club's private intent; at scoring time they are hypothetical and user-controlled. The probabilities therefore partly restate the proposal. This alone invalidates the live interpretation of cards 1–3 on `main`. The branch reached the same conclusion independently and excluded these fields. (MM-02, critical)
3. **The post-2023 "genuinely unseen" verification for the survival endpoints is biased.** Rows whose horizon had not elapsed were included whenever the event had already happened. The 36-month cohort contains zero rows with 36 months of follow-up and is 100% events; the 24-month cohorts have inflated event rates. The verifier reports 29/29 passed. (MM-03, high)
4. **"Low contribution" is mostly departure.** Of 691 negative labels in the primary cohort, 433 left the club within two years (238 permanently); only 258 stayed and were under-used. Of the €1.82B headline "wages attached to low contribution", €0.77B belongs to players sold within 24 months. The euro card is labelled as wage exposure. (MM-04, high)
5. **Subgroup reliability is weaker than the pooled figures suggest.** Ligue 1 continuity AUC is 0.61 (24m) and 0.58 (36m) in the frozen V1 evidence and has negative skill in the branch's repaired candidate; the "80% reference range" on the wage-commitment card covers 66% of players without a known prior wage and 59–71% of Ligue 1 players. (MM-06, MM-07, high)
6. **What does reproduce is real.** Every headline Phase 1–5 metric, the €5.22B/€1.82B/54-vs-79 business illustration, the C6 non-inferiority result, the frozen artifact hashes, and the branch's Engine V2 numbers (including its own failed gates) were reconstructed from row-level files and match. The public-value downside signal (AUC 0.82–0.84 pooled, above 0.73 in every league) is the strongest and most stable component, though its deployed specification shares the proposal-input defect.

**Present development status.** The presentation-era product (V1) is frozen and not served. Engine V2 has produced no authorised output: its own gates blocked every candidate, and its authors correctly declined to open an immature, partially exposed 2024 holdout. The project is a well-instrumented research record with deterministic contract arithmetic and historical peer context that can be shown, and predictive endpoints that require a proposal-free refit and a later, player-disjoint cohort before they can be called validated.

---

## 2. Reconstructed project architecture

Derived from code, not documentation.

**Data layer (local, mostly gitignored).**
Transfermarkt current collection and supplemental datalake → `canonical_integration` (identities, valuations, transfers) and `canonical_performance` (player-team-season rows, FBref advanced companion). Capology HTML tables → `capology_contracts` (salary panel, 2,805 extension events). `build_contract_extension_integration.py` anchors one row per extension and attaches pre-signing windows (365-day appearances, prior completed seasons, at-signing valuation, salary-panel context) and post-signing outcomes (Year-1/2/3 opportunity share, 12/24/36-month valuation landmarks, first any/permanent outbound). Output: `extension_modeling_master.csv`, 2,805 × 392, SHA `de2886ea…`, matching the deployment manifest.

**Research layer (`models/run_*`, `scripts/verify_*`).** Five chronological phases on the extension master (opportunity, survival, value preservation, financial peer benchmarks, integration), each with rolling origins 2020–2023, a fixed feature ladder M0→M6 (or B0→B5), hyperparameters selected on the immediately preceding year, and player-clustered bootstrap intervals. A business-translation audit and a product-layer prototype sit on top. Post-production experiments (contribution refinement, C6 repair, role contextualisation) reuse the same origins. An older transfer-risk track (retention, market-value downside, under-utilisation, fee benchmark, capital-at-risk, landmark monitoring) and the original compatibility pipeline are dormant history.

**Production layer (V1, on `main`).** `scripts/build_deployment_models.py` refits the selected variants on signed years ≤2023 (train ≤2022, validate 2023) and serialises 16 artifacts. `canonical_feature_retriever.py` rebuilds the feature vector for an arbitrary player/club/date from the same tables plus append-only 2024–26 overlays. `deployment_scorer.py` scores all 16 endpoints, applies isotonic re-ordering across downside thresholds and a one-sided clamp so permanent-stay ≥ continuous-stay, and produces exact logit decompositions. `business_metric_engine.py` turns probabilities into six typed cards. `api/main.py` (FastAPI) exposes search, input defaults and `POST /profile`; `html/index.html` renders four headline cards.

**Branch layer (`codex/engine-v2-safety-freeze`).** Freezes V1 byte-for-byte, replaces `api/main.py` with a fail-closed service, adds `models/engine_v2/` contracts (validity boundary, feature policy, result schema, movement product scope), five research phases with runners and verifiers, a critical-issues register, and a data-usage audit. The hosted service's `/health` payload is byte-identical to this branch's code.

---

## 3. Dataset and coverage inventory

Full table: `dataset_and_activity_inventory.csv`, `cohort_target_and_overlap_audit.json`.

| Layer | What is actually there | Coverage facts established in this audit |
| --- | --- | --- |
| Capology extensions | 2,805 events, 2,096 players, 141 clubs, Big Five, Jul 2018–Jun 2025 | 2018: 243 … 2023: 398, 2024: 385, 2025: 180. Premier League 2022-23 page absent. 97.2% player linkage. 399 players have 2 events, 100 have 3+. |
| Capology salaries | 19,770 rows to 2023-24; overlay 9,103 rows 2024-25 to 2026-27 | Extension page season equals the signing season in 2,751 of 2,805 events. |
| Transfermarkt | appearances 1.89M, games 88,958, valuations 507,815 (+901K datalake), transfers 35,139 (+1.10M datalake) | Sporting data ends 2026-07-06 (65 days before this audit); valuations end 2026-02-27; outbound censor 2026-08-10. |
| FBref | 18,243 player-squad-seasons 2017-24; 2,854 + 2,839 rows 2024-26 | Advanced features present for 59% of events at pre1; absent for all three demo players today. |
| Injury sources | 15,603-row thesis file (used in two rejected experiments); 143,195-row datalake table (never integrated) | No injury field in any deployed or candidate model. |
| StatsBomb | 15 GB local; 4,235 event files; 642 matches extracted | Case studies only; no model input. Not listed in `DATA_PROVENANCE.md`. |

Cohort funnels reconstructed: Phase 1 primary 1,385 (1,188 players); Phase 2 2,358; Phase 3 2,162/1,944; Phase 4 2,265/2,358/2,265/2,222; Phase 5 overlap 839. All match the READMEs.

---

## 4. Active-versus-dormant inventory

| Status | Components |
| --- | --- |
| Active on `main` (would run if served) | 16 V1 artifacts, retriever, scorer, explainer, business engine, API, HTML, live overlays, C6 artifact. |
| Active on the hosted service | Static pages and `/health` only. Scoring routes return 503. |
| Frozen (branch policy) | Everything under `deployment_models/` (24 files, hashes verified identical on disk). |
| Development-only (branch) | Engine V2 compact sustained-contribution and value-downside candidates; Meaningful Retention; Temporary Displacement. |
| Research-only | Role-contextualisation candidates; contribution refinement; Engine V2 continuity, Permanent Separation, club-finance, lagged-behaviour, role-squad, manager-tactical blocks; the entire transfer-risk track. |
| Dormant / obsolete | Compatibility pipeline and its outputs (`outputs/`, 253 MB), `Data/app-html/index.html` (superseded UI), `html/MOVE MAKER … v2.dc.html` (design export), `git/` (53 MB copy of repository metadata), `.tmp/` (296 MB), legacy root matrices `model_dataset*.csv`, presentation decks. |
| Ambiguous | 14 untracked `engine_v2_*` directories plus `capology_club_financial_context*` and `data_usage_audit_*` in the working tree: row-level outputs of branch code that is not checked out. They look like results but have no source on `main`. |

---

## 5. Target audit

**Sustained meaningful contribution** (`≥25% same-club opportunity share in Year 1 and Year 2`, cohort requires ≥10 club games in the pre-365 and Year-2 windows and a contract covering day 730). Reconstructed rate 0.501. The negative class is heterogeneous: 63% of non-sustainers had an outbound move within 730 days (34% permanent). Eighty-seven sustainers also had an outbound event (loans out and back). Football reading: a striker sold for a large fee in month 14 is labelled the same as an injured squad player. The V2 branch's decomposition into Meaningful Retention, Temporary Displacement and Permanent Separation is the right correction and reproduces (AUC 0.759, 0.777, 0.628).

**Year-2 opportunity share** (continuous). Defensible as a broad band (R² 0.25). Not displayed.

**Any outbound / permanent outbound** (first transfer record leaving the extension club, loans count as any-outbound). Scheduled expiration is not a censor, which is correct. Sixty-five first events are typed `loan_return`, whose meaning as an outbound from the incumbent club is unclear (MM-23). The V1 label mixes squad-management loans with sales, which is why it ranks poorly and why the branch retired it.

**Public-value downside** (nearest valuation within 90 days of 12/24 months, at-signing valuation within 365 days, thresholds 10/25/50%). Clean, strict, and the best-behaved target. Public market value is an estimate, and the interface says so.

**Wage/term benchmarks.** Targets are the realized terms themselves; predictors deliberately exclude the current terms (verified in the feature manifest and artifacts). This is the only model family whose training and scoring semantics agree.

**Selection.** All targets are conditional on an extension having been signed on the observed terms. There is no population of players considered and not extended. `main` does not state this; the branch does (MM-08).

---

## 6. Feature and scoring-time availability audit

Deployed feature inventory: `deployed_artifact_feature_inventory.csv`.

- **Proposal fields as predictors (MM-02).** Twelve artifacts (all Phase 1–3 endpoints, including C6) consume `exact_duration_years`, `age_at_expiration`, `log_annual_gross_eur`, `log_fixed_wage_commitment_eur`, `salary_change_from_prior_pct`, `annual_wage_to_market_value`, `commitment_to_market_value`, and club-panel percentile/share computed *with the proposal inserted*. Training values were the signed terms. In the primary cohort the sustained rate climbs from 0.37 for 2–3-year deals to 0.69 for 5-year-plus deals, and from 0.25 to 0.72 across signed-wage quintiles: the signed contract is a strong proxy for the club's intent to use the player. At scoring time the same fields are whatever the user types.
- **Measured effect (T1, `t1_effect_of_proposed_years_wage_fixed.csv`).** Across ten real 2023 profiles spread over leagues, positions and value bands, scored through the live retriever and scorer:

| Proposed term | sustained contribution | stay 24m | permanent 24m | downside 25% 24m | downside 50% 24m |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 years | 0.526 | 0.450 | 0.634 | 0.536 | 0.331 |
| 4 years | 0.621 | 0.623 | 0.717 | 0.451 | 0.243 |
| 6 years | 0.706 | 0.752 | 0.787 | 0.373 | 0.177 |

  Wage ×0.5 → ×3 (term fixed at 3): sustained 0.516 → 0.683; downside 50% 0.328 → 0.153. Median per-profile range over the 30-cell grid: sustained 0.32, stay-24m 0.40, downside-25% 0.26.
- **Global scoring rates (MM-05).** Eleven artifacts (12 endpoints, including the contract-duration peer) carry goals/assists-per-90 with a single global coefficient. The project's own role diagnostic showed sign-stable *positive* hazard coefficients (more goals → more departure risk) and that Bellingham's score was genuinely lowered by his scoring rate. The live explanation panel for Haaland and Mbappé today lists "Goals per 90" as raising departure risk. Only the sustained-contribution artifact was repaired.
- **Live availability.** Sporting data end 2026-07-06 (warned); demo market values are 271–458 days old (age shown only in the input field; a caller-supplied stale value bypasses the retriever's staleness warning); advanced FBref features are unavailable for all three demo players; the OOS re-score imputed 5.1 features per downside score on average. Missing-indicator columns therefore fire for reasons that differ from training (MM-11).
- **League encoding.** The 2026-08-16 bug (competition code vs league display name) is genuinely fixed; artifacts' categories are display names and the retriever maps codes.
- **Incumbency.** The API scores an arbitrary player–club pairing with only a warning (`incumbency_verified=false`); the HTML hides club choice; the branch makes it a hard refusal (MM-12).

---

## 7. Validation and leakage audit

**Implemented as described?** Rolling origins (train ≤ Y-2, validate Y-1, evaluate Y) with preprocessing and hyperparameters fit inside each origin: yes, verified in all five runners. Player-clustered bootstrap: yes. Survival horizons evaluated only where every subject has full follow-up (`horizon_is_complete`): yes for the historical evaluation. Feature timing checks exclude `post_*` fields: yes.

**Not player-disjoint (MM-09).** 13% (2020) to 37% (2023) of evaluation-year events belong to players already in earlier years. Splitting the frozen predictions by seen/unseen player: Phase 1 AUC 0.793 vs 0.787; Phase 2 0.703 vs 0.703; Phase 3 0.840 vs 0.808; Phase 4 commitment MAE 0.576 vs 0.725, coverage 0.872 vs 0.777. The effect is small for the classifiers and material for the wage benchmark.

**Evaluation-set reuse (MM-13).** The same four evaluation years selected Phase 1–5 decisions, the business proxy, C6 among six candidates, role-aware candidates across 13 endpoints, and the branch's Phases 2–4. No untouched V1 holdout exists. The branch documents that its 2024 role/continuity labels were exposed in aggregate and refuses to open them; that is the correct call.

**Post-2023 verifier (MM-03).** `verify_deployment_out_of_sample.py` uses `event OR followup ≥ horizon` as eligibility. Reconstructed (`oos_verifier_maturity_bias.csv`, `t2_oos_live_rescoring_metrics.json`):

| Endpoint | Verifier cohort | Fully followed | Included only because event occurred | Verifier | Corrected |
| --- | ---: | ---: | ---: | --- | --- |
| any_outbound_24m | 323 | 244 | 79 | AUC 0.679, rate 0.60 | AUC 0.689, rate 0.48 (n=243) |
| any_outbound_36m | 196 | 0 | 196 | Brier 0.094, one class | no valid rows |
| permanent_outbound_24m | 291 | 244 | 47 | AUC 0.652 | AUC 0.697 (n=243) |

Phase 1 (n=80, AUC 0.793), Phase 3 12m (n=381, AUC 0.794) and Phase 4 rows are not affected by this bug but are small or single-year.

**Temporal leakage inside features.** None found. Salary-panel context uses the same page season as signing (which contains the extension's own new wage in most rows: a mild look-ahead into the negotiated figure, but that figure is precisely what the user supplies at scoring time, so semantics match). At-signing valuation can be up to 5,724 days old in Phase 1/2/4 cohorts (strict 365-day rule only in Phase 3) (MM-18).

**Duplicate-player leakage inside the survival person-period expansion.** Person-periods are per event; the same player's second extension is a separate subject. Clustering is by player in the bootstrap only.

---

## 8. Model-by-model assessment

Reconstructed from `model_predictions.csv`, `survival_predictions.csv`, `value_predictions.csv`, `benchmark_predictions.csv` (local, gitignored). Full table: `metric_reconstruction.csv`; classifications: `model_metric_status_table.csv`.

| Endpoint (deployed variant) | Reproduced pooled evidence | Assessment |
| --- | --- | --- |
| Sustained contribution, V1 C6 (logistic, 15 features) | Frozen M5 AUC 0.788, Brier 0.187, ECE 0.046; C6 vs M5 +0.002 Brier (CI includes 0). Live re-score 2024-25: AUC 0.793, mean pred 0.526 vs observed 0.475, ECE 0.094 (n=80). By league AUC 0.72–0.87; Ligue 1 0.79. | Positive, imprecise signal historically. As deployed: proposal-driven (MM-02) and departure-conflated (MM-04). Sporting-only features alone (M2/M3) give AUC 0.77–0.78, so most of the signal is real. |
| Year-2 opportunity share (Ridge M5) | MAE 0.24, R² 0.25, Spearman 0.51. | Broad band only; not displayed; same defects. |
| Any outbound 24m / 36m (discrete hazard M5) | Pooled AUC 0.703 / 0.682, Brier 0.219 / 0.203; Ligue 1 AUC 0.609 / 0.584, skill +0.012 / +0.004; direct classifier 0.714 beat the deployed survival form. | Weak, league-unreliable, proposal-driven, mixes loans and sales. The branch retired it; that is supported. |
| Permanent outbound 24m (hazard M5) | AUC 0.694; Ligue 1 0.585; clamped to stay-24m in 9% of OOS profiles (40% in the project's live sample). | Same as above; two survival models are mutually inconsistent. |
| Downside 25% / 10% / 50% at 24m (logistic M5) | AUC 0.820 / 0.833 / 0.799; Brier 0.172 / 0.165 / 0.165; ECE 0.03–0.05; every league AUC ≥ 0.73; contract-economics increment ≈ 0 (M3 AUC 0.817). OOS n=41 only. | Strongest component. The proposal fields add nothing in the historical test yet drive the live number (downside-50% halves when wage triples). A proposal-free refit is defensible with limitation. |
| Downside 25% / 10% at 12m | AUC 0.780 / 0.786; OOS 12m AUC 0.794, ECE 0.045 (n=381). | Same. |
| Value log-ratio 12m / 24m (Ridge) | Spearman 0.61 / 0.66. | Not displayed. |
| Annual wage / fixed commitment / wage-to-value peers (Ridge B2) | MAE 0.52 / 0.69 / 0.51 log units; Spearman 0.89 / 0.83 / 0.83; coverage 0.81 / 0.80 / 0.81 pooled, 0.67 / 0.66 / 0.67 without prior wage, Ligue 1 0.71 / 0.76 / 0.70. | Proposal-free, strong ranking, but the interval is a single global width; V2's sealed test failed Ligue 1 coverage (0.59–0.61; Bundesliga 0.70). |
| Contract-duration peer (GBM B5, 20 features) | MAE 0.83 years, Spearman 0.66, coverage 0.82. | Weak; uses global scoring rates; shown only as a delta. |
| Integrated composite | ΔAUC −0.005 [−0.039, +0.028]; +0.010 [−0.010, +0.030]. | Rejected; correctly not exposed. |
| Low-contribution exposure proxy | €5.22B / €1.82B (34.8%); 54 vs 79 hits at capacity 182; MAE gain €761,630: all reproduced exactly. A probability-only screen finds 164 hits at the same capacity. | The comparison baseline (largest contracts) is weak; the euro amount counts wages of sold players; the probability is proposal-driven. |
| Engine V2 candidates (branch) | Compact sustained AUC 0.754; compact downside 0.817; continuity core+age 0.644 with Ligue 1 skill −0.0066; Meaningful Retention 0.759; Temporary Displacement 0.777; Permanent Separation 0.628 with negative Bundesliga/Ligue 1 skill; wage sealed test logMAE 0.506 vs 0.812–0.836, coverage 0.770, Ligue 1 0.59–0.61. All reproduced from leftover row-level files. | Development-only; none authorised; the branch's own gates hold. |

---

## 9. Calibration, uncertainty, subgroup and out-of-distribution assessment

- **Calibration.** Frozen ECE 0.03–0.06 on the sporting/value endpoints; risk bands monotone. On the newest cohort the contribution model over-predicts by 5 points (n=80) and the 24-month downside models show ECE 0.13–0.15 (n=41): small samples, but the direction is consistent (MM-10). The survival models' calibration cannot be assessed on post-2023 data because of MM-03.
- **Uncertainty.** V1 shows no per-profile uncertainty on probabilities; the only intervals are the Phase-4 empirical residual widths, which are marginal (MM-07). The branch adds 250-fit bootstrap model-fit intervals (mean width 0.12–0.16) and explicitly labels them as not prediction intervals; correct.
- **Subgroups.** Ligue 1 is the recurring weak league for continuity and for wage-benchmark coverage; La Liga is weakest for contribution (AUC 0.72); the ≤21 and 34+ bands are thin (n=203/46 in Phase 1). V1 gates were pooled only; V2 gates are per advertised league and correctly block.
- **Out-of-distribution.** The data-density warning fires correctly: today's demo players sit at the 99.9–100th percentile of the training value distribution with 1–3 comparable extensions, and the interface says every estimate is an extrapolation. This is honest. It does not change the fact that the numbers still render as precise percentages.
- **Refusal behaviour.** Out-of-Big-Five league: hard refusal (verified). Missing DOB or unknown player: all modules unavailable (verified by the project's own checks). Arbitrary club: warning only (MM-12). Missing secondary features: imputed by trained medians with missing indicators (documented; but see MM-11 and MM-15).

---

## 10. API / frontend / artifact consistency audit

- **Artifacts vs code.** All 16 artifacts load with the pinned scikit-learn 1.6.1, their hashes match the manifest and the branch's freeze inventory, and the scorer reproduces the explainer's decomposition exactly (project checks, re-confirmed on today's demo run).
- **API vs frontend (main).** Consistent. The HTML never lets a user choose a club; the API does (MM-12). The `/profile` decision date is the server's date (non-reproducible).
- **Frontend copy vs evidence.** Card 1 says the score is "not a talent or tactical-fit rating" but not that it is conditioned on the proposal or that most negatives are departures. Card 2 presents "continuous-stay probability" with a green/amber/red bar at fixed thresholds for a model with Ligue 1 AUC 0.61. Card 4 says "historical 80% reference range" unconditionally. The "Model basis" text does disclose that proposed terms are inputs; the effect size is not disclosed.
- **Hosted service vs both.** `/health` reports `research_prototype`, `scoring_enabled: false`; the branch's HTML shows a "Research prototype — live scoring is paused" banner and replaces card 2 with a scope explanation. `main`'s README links this URL as a "live demo" and its screenshot shows the retired card.
- **Verification outputs.** Two verifiers write into the frozen artifact directory; the branch's freeze check would fail after any rerun (MM-17).

---

## 11. Football-logic review

- **Signed terms are outcomes of the club's decision, not causes of the player's outcome.** A five-year, top-of-wage-structure deal is offered to a player the club intends to build around. Using that as a predictor of "will he play" measures intent, not ability or fitness. The historical association is real (rate 0.37 → 0.69 by term band) and exactly why it must not be a user input to a sporting forecast. The branch's feature contract states this correctly.
- **Departure is not failure.** A permanent sale within two years ends the wage obligation and often returns a fee; a loan may be developmental. Bundling these with under-use into "low contribution", then multiplying by scheduled wages, produces a euro figure that is neither expected wage waste nor expected loss. The project already says "not expected loss"; it should also say "includes players who were sold".
- **Goals per 90 is not comparable across roles**, and a global coefficient that says scoring raises departure hazard is picking up market demand for forwards. The project found this and fixed one of twelve endpoints.
- **Continuity mixes mechanisms.** Loan-out of a 20-year-old and sale of a 29-year-old are different club decisions with different wage consequences; a single "any outbound" probability has little decision value, which the AUCs confirm.
- **Public-value downside is the football-coherent signal.** Age at signing, age at expiry and current value dominate the coefficients in the expected direction; the strong AUC survives removal of contract terms. This is the one predictive component whose logic a sporting director would recognise.
- **Wage benchmarks are the honest financial context.** They do not use the proposal, they rank well, and their weakness (interval width for players without a public prior wage, and in Ligue 1) is a coverage problem, not a logic problem.
- **What is not observed** (options, clauses, bonuses, injuries at usable coverage, manager change, squad competition) is stated repeatedly in the documentation; the branch tested manager, formation, squad and club-finance context and found no stable gain, which is a useful negative result.

---

## 12. Reproducibility assessment

- **Reproduced in this audit:** every Phase 1–4 headline metric from row-level prediction files; Phase 5 rejection numbers; the business illustration to the euro; C6 selection statistics; the branch's V2 metrics from leftover row-level files; artifact hashes; the retriever's live path (547 post-2023 events and 300 perturbation cells scored without exception in 128 s after a 42 s warm-up).
- **Not reproducible from Git alone:** the row-level predictions, the modeling master, canonical tables, raw sources and live overlays are all gitignored; the repository is, as it says, an "auditable publication baseline", not a rebuild bundle. The 2,425-event frozen snapshot for Phases 1–5 exists locally only.
- **Verifier independence is partial (MM-17).** Verifiers recompute metrics and hashes but import the runners' target construction; they would not catch a wrong target definition, and one of them has a wrong eligibility rule (MM-03).
- **Environment.** Python 3.13.5, scikit-learn 1.6.1 (pinned), pandas 2.2.3; local FastAPI is newer than pinned. Random seeds are fixed; GBM seeds are derived deterministically.
- **The handoff documents (`CODEX.md`, `CLAUDE.md`) are gitignored.** A clone does not contain the operating rules the project treats as authoritative.

---

## 13. Contradictions between claims and implementation

Table: `contradictions_claims_vs_implementation.csv`. The most consequential:

| Claim | Reality |
| --- | --- |
| README (main): live demo; 16 deployed, separately validated endpoints | Service refuses to score; the branch that runs there calls V1 a frozen research prototype |
| Model card (main): "Deployed decision signals" | Nothing is deployed; the branch's model card says so |
| Interface: probabilities describe the player's contribution/continuity/value risk | They move 10–30 points with the user's own proposal |
| Interface: "low-contribution wage exposure" | 42% of the historical exposed wages belong to players sold within 24 months |
| Out-of-sample verification: 29/29 passed, 36m "Brier only" | 36m cohort has no row with 36 months of follow-up |
| Card 4: "historical 80% reference range" | 66% for players without a known prior wage; 59–71% Ligue 1 |
| Scope page: production package independently checked on post-2023 events | Survival portion biased; other portions rest on n=80 and n=41 |
| Data provenance: four source families | StatsBomb (15 GB, processed) omitted |

---

## 14. Issue ledger

Full ledger with evidence, files, consequence, classification, repair and repairability: `issue_ledger.csv` / `issue_ledger.json`. Summary:

| ID | Sev. | Status | Title | Effect |
| --- | --- | --- | --- | --- |
| MM-01 | critical | confirmed | Main, hosted service and unmerged branch describe three different products | invalidates main's status claims |
| MM-02 | critical | confirmed | Proposed wage/term are predictors in 12 of 16 artifacts; probabilities move 10–30 pts with the proposal | invalidates live interpretation of cards 1–3 |
| MM-03 | high | confirmed | OOS verifier admits immature survival rows only when the event occurred | constrains; 36m result meaningless |
| MM-04 | high | confirmed | "Low contribution" and euro exposure conflate sales/loans with under-use | constrains; label misleading |
| MM-05 | high | confirmed | Global goals/assists-per-90 in 11 artifacts with counterintuitive live explanations | constrains |
| MM-06 | high | confirmed | Continuity weak in Ligue 1 and internally inconsistent (clamp in 9–40% of profiles) | invalidates for Ligue 1 |
| MM-07 | high | confirmed | "80% reference range" covers 66–71% in identifiable subgroups | constrains |
| MM-08 | high | confirmed | Conditioning on having been extended not stated on main | constrains |
| MM-09 | medium | confirmed | Evaluation not player-disjoint; material for the wage benchmark | constrains |
| MM-10 | medium | probable | Calibration drift on 2024-25 cohort (small n) | constrains |
| MM-11 | medium | confirmed | Stale/absent live inputs vs training semantics | constrains |
| MM-12 | medium | confirmed | Arbitrary club pairing scored with a warning at API level | constrains |
| MM-13 | medium | confirmed | Same four evaluation years reused for every selection; no untouched V1 holdout | constrains |
| MM-14 | medium | confirmed | Display-time re-ordering and clamps are load-bearing | debt / constrains |
| MM-15 | medium | confirmed | Maximal feature set deployed even where blocks add nothing | debt |
| MM-16 | medium | confirmed | C6 is a non-inferiority repair selected on reporting folds, not an improvement | debt |
| MM-17 | low | confirmed | Verifiers share target code and write into the frozen directory | debt |
| MM-18 | low | confirmed | Data-quality residues (positions, valuation gaps, missing PL page) | debt |
| MM-19 | low | confirmed | Dormant assets that read as results | debt |
| MM-20 | low | confirmed | Documentation gaps (handoff gitignored, provenance, model card) | debt |
| MM-21 | low | confirmed | API hardening and environment drift | debt |
| MM-22 | medium | confirmed | Engine V2 has no authorised output; its gates blocked every candidate | status |
| MM-23 | low | unresolved | 65 `loan_return` first-outbound events of unclear meaning | debt |

---

## 15. Recommended order of work

1. **Reconcile the repository with reality (MM-01, MM-20).** Merge or rebase `codex/engine-v2-safety-freeze` into `main` through a reviewed PR, or at minimum update README, model card and handoff on `main` to the frozen/paused state; commit the handoff; record the deployed commit; remove or commit the orphaned `engine_v2_*` outputs.
2. **Fix the post-2023 verifier (MM-03)** so survival rows require full follow-up, and re-publish the corrected numbers (any_outbound_24m AUC 0.69, permanent 0.70, 36m pending).
3. **Refit the sporting and value endpoints without commercial-scenario features (MM-02, MM-05, MM-15).** The branch's Phase 2 shows compact, proposal-free specifications retain most of the signal (sustained 0.75, downside 0.82). Keep the proposal only in deterministic and peer cards. Re-run parity, explainability and refusal checks.
4. **Decompose the contribution/continuity targets (MM-04, MM-06, MM-23)** along the branch's Meaningful Retention / Temporary Displacement / Permanent Separation lines; cap wage exposure at the departure date; relabel the euro card; review the 65 loan-return cases.
5. **Make intervals and support conditional (MM-07, MM-09).** Report coverage by prior-wage availability and league; return `limited` where coverage is below the declared level; publish player-disjoint metrics alongside pooled.
6. **Harden the scoring contract (MM-11, MM-12, MM-14, MM-21).** Hard-refuse unverified incumbency; propagate market-value and sporting-data staleness into profile warnings; fit ordered thresholds and the any/permanent pair jointly; pin `decision_date`; tighten CORS.
7. **Reserve a genuinely later cohort (MM-13, MM-10, MM-22).** Freeze candidates now; open the 2025-26 signings only when Year-2 and 24-month outcomes have matured (2028), player-disjoint, with league counts declared in advance.
8. **State the conditioning (MM-08)** in every user-facing surface: outcomes among players who were extended on the terms they signed.

Do not: lower gates, add manual adjustments, or re-tune on the 2024 labels.

---

## What can MoveMaker responsibly claim today?

MoveMaker can claim to be a well-documented, reproducible research programme on Big-Five incumbent-club contract extensions, built on 2,805 Capology extension events joined to Transfermarkt and FBref history, with chronological evaluation and saved verification.

It can claim, as historical findings conditional on an extension having been signed: that public-value downside over 12–24 months is predictable from age, current value and recent involvement (AUC 0.78–0.84, stable across leagues); that sustained same-club involvement through Year 2 is moderately predictable from recent involvement and profile (AUC about 0.77–0.79); that realized wage commitments rank strongly against a profile-plus-prior-wage benchmark (Spearman 0.83–0.89) with an interval that is well calibrated in aggregate but not for players without a public prior wage or in Ligue 1; and that neither a composite score nor a generic "continuity" probability earned its place.

It can offer, today, deterministic contract arithmetic (fixed commitment, commitment-to-value, age at expiry, wage change), matched club salary context, and historical peer context with explicit coverage caveats.

It cannot claim a live, validated decision profile. The hosted service is paused for good reasons that the public branch does not yet state. The V1 sporting and value probabilities are partly a function of the user's own proposal, their negative class is mostly departures, one league is unsupported for continuity, and the only post-2023 check for the survival models is biased. Engine V2 has candidates with real development-stage signal and no authorised output.

The honest one-line status is: **frozen prototype with reproducible historical evidence; predictive endpoints require a proposal-free refit and a later player-disjoint test before any personalised probability is shown again.**
