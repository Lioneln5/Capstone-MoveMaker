# C6 sustained_meaningful_contribution deployment verification report

Deployed at (UTC): 2026-08-17T02:12:56.770948+00:00
Variant: `C6_C3_minus_pre365_goals_assists`
Feature count: 15 (excludes pre365_goals_per90 and pre365_assists_per90)
Training rows: 1304
Selection basis: **compact_non_inferiority** (not statistical superiority)
Artifact sha256: `adf4e464fe213bec23e8eaf612ebb8959a0fd5aee9bd7ce725325a94b06614db`

## Checks

- PASS `check6_deployed_artifact_excludes_goals_and_assists_per90`: observed=[] expected=[]
- PASS `check6_manifest_documents_exclusion`: observed=['pre365_goals_per90', 'pre365_assists_per90'] expected=['pre365_goals_per90', 'pre365_assists_per90']
- PASS `check4_selection_basis_documented_as_compact_non_inferiority`: observed='compact_non_inferiority' expected='compact_non_inferiority'
- PASS `manifest_other_15_endpoints_untouched`: observed=16 expected=16
- PASS `check7_all_16_endpoints_scored_without_exception`: observed=16 expected=16
- PASS `check7_all_statuses_are_valid_declared_states`: observed=['supported_candidate', 'supported_candidate_with_imputed_inputs'] expected='supported_candidate(_with_imputed_inputs) or unavailable'
- PASS `check8_Haaland_current_live_probability_in_unit_interval`: observed=0.808672 expected='[0, 1]'
- PASS `check8_Mbappe_current_live_probability_in_unit_interval`: observed=0.814661 expected='[0, 1]'
- PASS `check8_Bellingham_current_live_probability_in_unit_interval`: observed=0.772695 expected='[0, 1]'
- PASS `check8_mbappe_2022_matches_research_candidate_probability`: observed=0.850056 expected=0.850056
- PASS `check9_Haaland_explainability_reconstructs_displayed_probability`: observed=0.8087 expected=0.8087
- PASS `check9_Haaland_no_goals_assists_contribution`: observed=[] expected=[]
- PASS `check9_Haaland_other_gauges_unaffected`: observed=True expected=True
- PASS `check9_Mbappe_explainability_reconstructs_displayed_probability`: observed=0.8147 expected=0.8147
- PASS `check9_Mbappe_no_goals_assists_contribution`: observed=[] expected=[]
- PASS `check9_Mbappe_other_gauges_unaffected`: observed=True expected=True
- PASS `check9_Bellingham_explainability_reconstructs_displayed_probability`: observed=0.7727 expected=0.7727
- PASS `check9_Bellingham_no_goals_assists_contribution`: observed=[] expected=[]
- PASS `check9_Bellingham_other_gauges_unaffected`: observed=True expected=True
- PASS `check9_Mbappe_2022_PSG_historical_explainability_reconstructs_displayed_probability`: observed=0.8501 expected=0.8501
- PASS `check9_Mbappe_2022_PSG_historical_no_goals_assists_contribution`: observed=[] expected=[]
- PASS `check9_Mbappe_2022_PSG_historical_other_gauges_unaffected`: observed=True expected=True
- PASS `check5_deployed_artifact_matches_experiment_saved_artifact`: observed=0.0 expected='< 1e-6'
- PASS `artifact_sha256_matches_manifest`: observed='adf4e464fe213bec23e8eaf612ebb8959a0fd5aee9bd7ce725325a94b06614db' expected='adf4e464fe213bec23e8eaf612ebb8959a0fd5aee9bd7ce725325a94b06614db'
- PASS `deployment_log_present`: observed=True expected=True

**All checks passed: True**

## Demo scores (production DeploymentScorer path)

- Haaland: 0.8086721598281377
- Mbappe: 0.8146606032387413
- Bellingham: 0.7726952632854973
- Mbappe_2022_PSG_historical: 0.8500560090339199
