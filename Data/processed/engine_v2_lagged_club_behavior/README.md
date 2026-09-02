# Engine V2 lagged club-behavior profiles

**Status:** research feature build; not deployed and not evaluated on the final holdout.

This table converts dated transfer and earlier-extension histories into club
behavior available at each extension decision. Transfer evidence is restricted
to dates strictly before signing. Earlier extension outcomes are usable only
after the full 730-day horizon has matured.

## Coverage

- canonical extension anchors: **2,805**;
- relevant canonical transfer events scanned: **46,931**;
- 2020–2023 role-conditioned profile coverage: **97.4%**;
- 2020–2023 rows with at least one mature prior same-club extension: **71.7%**.

Zero historical events is a real, explicit profile with a support count of zero;
it is not treated as missing. Role-conditioned fields are unavailable only when
the extension player's broad position or age band is unavailable.

## Mechanisms represented

1. General club churn and the balance between temporary and permanent outbound movement.
2. The same behavior conditional on the extension player's broad position and age band.
3. Outcomes of earlier same-club extensions whose two-year results were already knowable.

Rates use fixed weak smoothing priors and always retain their support counts.
No global future league rate, future transfer, or immature extension outcome is
used. The `loan_return` source type remains visible rather than being silently
rewritten as an ordinary loan.

## Evidence

- `lagged_club_behavior_features.csv`
- `asof_evidence_audit.csv`
- `feature_dictionary.csv`
- `movement_target_contract.json`
- `build_checks.csv`
- `source_manifest.csv`
