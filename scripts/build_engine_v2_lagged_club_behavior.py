"""Build strictly pre-decision club movement and extension-behavior profiles.

The output is research-only.  Every transfer must precede the extension date,
and a previous extension can contribute an outcome only after its full
730-day horizon has matured.  No 2024+ target is read for model evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "models")]

from models.engine_v2.movement_target_contract import (  # noqa: E402
    CONTRACT_VERSION,
    HORIZON_DAYS,
    add_movement_targets,
    contract_payload,
)
import run_engine_v2_feature_specification as phase2  # noqa: E402


DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "engine_v2_lagged_club_behavior"
TRANSFER_PATH = ROOT / "Data" / "processed" / "canonical_integration" / "canonical_transfer_history.csv"
PLAYER_PATH = ROOT / "Data" / "processed" / "canonical_integration" / "canonical_player_dimension.csv"
EXTENSION_PATH = ROOT / "Data" / "processed" / "capology_contracts" / "canonical_extension_events.csv"
FREEZE = ROOT / "docs" / "releases" / "engine_v1_freeze_2026-08-30.json"
TRANSFER_LOOKBACK_DAYS = 1095
GENERAL_WINDOW_DAYS = 730

GENERAL_FEATURES = [
    "club_outbound_events_730d_log1p",
    "club_outbound_unique_players_730d_log1p",
    "club_transaction_churn_730d_log1p",
    "club_inbound_outbound_ratio_730d_log",
    "club_temporary_outbound_share_1095d",
    "club_loan_return_within_temporary_share_1095d",
]
ROLE_FEATURES = [
    "club_position_outbound_typed_count_1095d_log1p",
    "club_position_temporary_share_1095d",
    "club_ageband_outbound_typed_count_1095d_log1p",
    "club_ageband_permanent_share_1095d",
]
EXTENSION_FEATURES = [
    "club_mature_prior_extension_count_log1p",
    "club_prior_extension_temporary_share",
    "club_prior_extension_permanent_first_share",
    "club_prior_extension_strict_stay_share",
    "club_position_mature_prior_extension_count_log1p",
    "club_position_prior_extension_temporary_share",
]
MODEL_FEATURES = GENERAL_FEATURES + ROLE_FEATURES + EXTENSION_FEATURES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def age_band(age: float | int | None) -> str | None:
    if age is None or pd.isna(age):
        return None
    age = float(age)
    if age <= 21:
        return "age_le_21"
    if age <= 25:
        return "age_22_25"
    if age <= 29:
        return "age_26_29"
    if age <= 33:
        return "age_30_33"
    return "age_34_plus"


def broad_position(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    aliases = {
        "Attack": "Attack",
        "Forward": "Attack",
        "Midfield": "Midfield",
        "Defender": "Defender",
        "Defence": "Defender",
        "Goalkeeper": "Goalkeeper",
    }
    if text in aliases:
        return aliases[text]
    for token, normalized in aliases.items():
        if text.startswith(f"{token} -"):
            return normalized
    return None


def load_extension_anchors() -> pd.DataFrame:
    columns = [
        "capology_extension_event_id", "canonical_player_id", "canonical_club_id",
        "canonical_position", "age_at_signing", "signed_date", "league",
    ]
    anchors = pd.read_csv(EXTENSION_PATH, usecols=columns, low_memory=False)
    anchors["signed_date"] = pd.to_datetime(anchors["signed_date"], errors="raise")
    anchors["canonical_club_id"] = pd.to_numeric(anchors["canonical_club_id"], errors="raise").astype(int)
    anchors["canonical_position"] = anchors["canonical_position"].map(broad_position)
    anchors["age_band_at_signing"] = anchors["age_at_signing"].map(age_band)
    return anchors.sort_values(["signed_date", "capology_extension_event_id"]).reset_index(drop=True)


def load_transfer_evidence(anchors: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "canonical_transfer_event_id", "canonical_player_id", "transfer_date",
        "from_club_id", "to_club_id", "canonical_transfer_type", "event_conflict",
        "event_resolution_status",
    ]
    transfers = pd.read_csv(TRANSFER_PATH, usecols=columns, low_memory=False)
    transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
    clubs = set(anchors["canonical_club_id"].astype(int))
    earliest = anchors["signed_date"].min() - pd.Timedelta(days=TRANSFER_LOOKBACK_DAYS)
    latest = anchors["signed_date"].max()
    conflict = transfers["event_conflict"].fillna(False).astype(bool)
    relevant = (
        transfers["transfer_date"].between(earliest, latest, inclusive="left")
        & (transfers["from_club_id"].isin(clubs) | transfers["to_club_id"].isin(clubs))
        & ~conflict
    )
    transfers = transfers.loc[relevant].copy()
    players = pd.read_csv(
        PLAYER_PATH,
        usecols=["canonical_player_id", "canonical_date_of_birth", "canonical_sub_position", "canonical_position"],
        low_memory=False,
    )
    players["canonical_date_of_birth"] = pd.to_datetime(players["canonical_date_of_birth"], errors="coerce")
    players["movement_position"] = players["canonical_sub_position"].map(broad_position)
    fallback = players["canonical_position"].map(broad_position)
    players["movement_position"] = players["movement_position"].fillna(fallback)
    transfers = transfers.merge(
        players[["canonical_player_id", "canonical_date_of_birth", "movement_position"]],
        on="canonical_player_id", how="left", validate="many_to_one",
    )
    transfer_age = (transfers["transfer_date"] - transfers["canonical_date_of_birth"]).dt.days / 365.2425
    transfers["movement_age_band"] = transfer_age.map(age_band)
    transfers["movement_class"] = transfers["canonical_transfer_type"].map(
        {"transfer": "permanent", "loan": "temporary", "loan_return": "temporary"}
    )
    return transfers.sort_values(["transfer_date", "canonical_transfer_event_id"]).reset_index(drop=True)


def load_mature_extension_history() -> pd.DataFrame:
    history = phase2.load_endpoint_frames()["any_outbound_24m"].copy()
    history = add_movement_targets(history)
    history["signed_date"] = pd.to_datetime(history["signed_date"], errors="raise")
    history["outcome_available_date"] = history["signed_date"] + pd.Timedelta(days=HORIZON_DAYS)
    history["canonical_club_id"] = pd.to_numeric(history["canonical_club_id_x"], errors="raise").astype(int)
    history["canonical_position"] = history["canonical_position"].map(broad_position)
    return history[[
        "capology_extension_event_id", "canonical_club_id", "canonical_position", "signed_date",
        "outcome_available_date", "movement_state_3", "target_strict_meaningful_stay_24m",
    ]].copy()


def smoothed_binary(successes: int, trials: int) -> float:
    return float((successes + 1) / (trials + 2))


def smoothed_state(count: int, trials: int) -> float:
    return float((count + 1) / (trials + 3))


def profile_anchor(
    anchor: pd.Series,
    outbound: pd.DataFrame,
    inbound: pd.DataFrame,
    prior_extensions: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object]]:
    decision = anchor["signed_date"]
    start_730 = decision - pd.Timedelta(days=GENERAL_WINDOW_DAYS)
    start_1095 = decision - pd.Timedelta(days=TRANSFER_LOOKBACK_DAYS)
    out_730 = outbound.loc[outbound["transfer_date"].ge(start_730) & outbound["transfer_date"].lt(decision)]
    in_730 = inbound.loc[inbound["transfer_date"].ge(start_730) & inbound["transfer_date"].lt(decision)]
    out_1095 = outbound.loc[outbound["transfer_date"].ge(start_1095) & outbound["transfer_date"].lt(decision)]
    typed = out_1095.loc[out_1095["movement_class"].notna()]
    temporary = typed["movement_class"].eq("temporary")
    permanent = typed["movement_class"].eq("permanent")
    temp_rows = typed.loc[temporary]
    loan_returns = temp_rows["canonical_transfer_type"].eq("loan_return")

    out_count = len(out_730)
    in_count = len(in_730)
    typed_count = len(typed)
    temp_count = int(temporary.sum())
    permanent_count = int(permanent.sum())

    position = anchor["canonical_position"]
    position_typed = typed.loc[typed["movement_position"].eq(position)] if position else typed.iloc[0:0]
    position_temp = int(position_typed["movement_class"].eq("temporary").sum())
    age_group = anchor["age_band_at_signing"]
    age_typed = typed.loc[typed["movement_age_band"].eq(age_group)] if age_group else typed.iloc[0:0]
    age_permanent = int(age_typed["movement_class"].eq("permanent").sum())

    mature = prior_extensions.loc[prior_extensions["outcome_available_date"].le(decision)]
    resolved = mature.loc[mature["movement_state_3"].notna()]
    resolved_count = len(resolved)
    prior_temp = int(resolved["movement_state_3"].eq("temporary_first_24m").sum())
    prior_permanent = int(resolved["movement_state_3"].eq("permanent_first_24m").sum())
    prior_strict = mature.loc[mature["target_strict_meaningful_stay_24m"].notna()]
    prior_strict_success = int(pd.to_numeric(prior_strict["target_strict_meaningful_stay_24m"]).sum())
    position_resolved = resolved.loc[resolved["canonical_position"].eq(position)] if position else resolved.iloc[0:0]
    position_prior_temp = int(position_resolved["movement_state_3"].eq("temporary_first_24m").sum())

    record = {
        "capology_extension_event_id": anchor["capology_extension_event_id"],
        "canonical_player_id": int(anchor["canonical_player_id"]) if pd.notna(anchor["canonical_player_id"]) else np.nan,
        "canonical_club_id": int(anchor["canonical_club_id"]),
        "signed_date": decision.date().isoformat(),
        "signed_year": int(decision.year),
        "league": anchor["league"],
        "canonical_position": position,
        "age_band_at_signing": age_group,
        "club_outbound_events_730d": out_count,
        "club_outbound_events_730d_log1p": math.log1p(out_count),
        "club_outbound_unique_players_730d": int(out_730["canonical_player_id"].nunique()),
        "club_outbound_unique_players_730d_log1p": math.log1p(out_730["canonical_player_id"].nunique()),
        "club_inbound_events_730d": in_count,
        "club_transaction_churn_730d": in_count + out_count,
        "club_transaction_churn_730d_log1p": math.log1p(in_count + out_count),
        "club_inbound_outbound_ratio_730d_log": math.log((in_count + 1) / (out_count + 1)),
        "club_outbound_typed_events_1095d": typed_count,
        "club_temporary_outbound_events_1095d": temp_count,
        "club_permanent_outbound_events_1095d": permanent_count,
        "club_temporary_outbound_share_1095d": smoothed_binary(temp_count, typed_count),
        "club_permanent_outbound_share_1095d": smoothed_binary(permanent_count, typed_count),
        "club_loan_return_within_temporary_share_1095d": smoothed_binary(int(loan_returns.sum()), len(temp_rows)),
        "club_position_outbound_typed_count_1095d": len(position_typed) if position else np.nan,
        "club_position_outbound_typed_count_1095d_log1p": math.log1p(len(position_typed)) if position else np.nan,
        "club_position_temporary_share_1095d": smoothed_binary(position_temp, len(position_typed)) if position else np.nan,
        "club_ageband_outbound_typed_count_1095d": len(age_typed) if age_group else np.nan,
        "club_ageband_outbound_typed_count_1095d_log1p": math.log1p(len(age_typed)) if age_group else np.nan,
        "club_ageband_permanent_share_1095d": smoothed_binary(age_permanent, len(age_typed)) if age_group else np.nan,
        "club_mature_prior_extension_count": resolved_count,
        "club_mature_prior_extension_count_log1p": math.log1p(resolved_count),
        "club_prior_extension_no_outbound_share": smoothed_state(int(resolved["movement_state_3"].eq("no_outbound_24m").sum()), resolved_count),
        "club_prior_extension_temporary_share": smoothed_state(prior_temp, resolved_count),
        "club_prior_extension_permanent_first_share": smoothed_state(prior_permanent, resolved_count),
        "club_prior_extension_strict_stay_trials": len(prior_strict),
        "club_prior_extension_strict_stay_share": smoothed_binary(prior_strict_success, len(prior_strict)),
        "club_position_mature_prior_extension_count": len(position_resolved) if position else np.nan,
        "club_position_mature_prior_extension_count_log1p": math.log1p(len(position_resolved)) if position else np.nan,
        "club_position_prior_extension_temporary_share": smoothed_state(position_prior_temp, len(position_resolved)) if position else np.nan,
    }
    used_transfer = pd.concat([out_1095, in_730], ignore_index=True)
    audit = {
        "capology_extension_event_id": anchor["capology_extension_event_id"],
        "signed_date": decision.date().isoformat(),
        "earliest_allowed_transfer_date": start_1095.date().isoformat(),
        "latest_transfer_date_used": used_transfer["transfer_date"].max().date().isoformat() if len(used_transfer) else None,
        "transfer_events_used": int(used_transfer["canonical_transfer_event_id"].nunique()),
        "latest_mature_extension_signed_date": mature["signed_date"].max().date().isoformat() if len(mature) else None,
        "latest_mature_extension_outcome_available_date": mature["outcome_available_date"].max().date().isoformat() if len(mature) else None,
        "mature_extension_events_used": int(len(mature)),
        "role_profile_available": bool(position and age_group),
    }
    return record, audit


def build_profiles() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anchors = load_extension_anchors()
    transfers = load_transfer_evidence(anchors)
    history = load_mature_extension_history()
    outbound_groups = {int(key): value for key, value in transfers.groupby("from_club_id")}
    inbound_groups = {int(key): value for key, value in transfers.groupby("to_club_id")}
    extension_groups = {int(key): value for key, value in history.groupby("canonical_club_id")}
    empty_transfer = transfers.iloc[0:0]
    empty_history = history.iloc[0:0]
    profiles, audits = [], []
    for _, anchor in anchors.iterrows():
        club = int(anchor["canonical_club_id"])
        record, audit = profile_anchor(
            anchor,
            outbound_groups.get(club, empty_transfer),
            inbound_groups.get(club, empty_transfer),
            extension_groups.get(club, empty_history),
        )
        profiles.append(record)
        audits.append(audit)
    counts = {
        "anchor_rows": len(anchors),
        "relevant_transfer_rows": len(transfers),
        "extension_history_rows": len(history),
    }
    return pd.DataFrame(profiles), pd.DataFrame(audits), counts


def feature_dictionary() -> pd.DataFrame:
    rows = [
        ("club_outbound_events_730d_log1p", "general_movement", "log(1 + outbound events in prior 730 days)", "strictly pre-signing canonical transfers"),
        ("club_outbound_unique_players_730d_log1p", "general_movement", "log(1 + distinct outbound players in prior 730 days)", "strictly pre-signing canonical transfers"),
        ("club_transaction_churn_730d_log1p", "general_movement", "log(1 + inbound + outbound events in prior 730 days)", "strictly pre-signing canonical transfers"),
        ("club_inbound_outbound_ratio_730d_log", "general_movement", "log((inbound+1)/(outbound+1)) in prior 730 days", "strictly pre-signing canonical transfers"),
        ("club_temporary_outbound_share_1095d", "general_movement", "Beta(1,1)-smoothed temporary share of typed outbound events", "loan and loan_return remain auditable raw types"),
        ("club_loan_return_within_temporary_share_1095d", "general_movement", "Beta(1,1)-smoothed loan-return share within temporary outbound events", "keeps loan-return mechanism visible"),
        ("club_position_outbound_typed_count_1095d_log1p", "role_conditioned", "log(1 + same-broad-position typed outbounds)", "unavailable when extension position is unknown"),
        ("club_position_temporary_share_1095d", "role_conditioned", "smoothed temporary share among same-position outbounds", "unavailable when extension position is unknown"),
        ("club_ageband_outbound_typed_count_1095d_log1p", "role_conditioned", "log(1 + same-age-band typed outbounds)", "age calculated at each historical move"),
        ("club_ageband_permanent_share_1095d", "role_conditioned", "smoothed permanent share among same-age-band outbounds", "unavailable when extension age is unknown"),
        ("club_mature_prior_extension_count_log1p", "mature_extension_history", "log(1 + resolved prior extensions with 730-day horizon complete)", "outcome_available_date must be <= current signing date"),
        ("club_prior_extension_temporary_share", "mature_extension_history", "Dirichlet(1,1,1)-smoothed temporary-first share", "future outcomes prohibited"),
        ("club_prior_extension_permanent_first_share", "mature_extension_history", "Dirichlet(1,1,1)-smoothed permanent-first share", "future outcomes prohibited"),
        ("club_prior_extension_strict_stay_share", "mature_extension_history", "Beta(1,1)-smoothed strict meaningful-stay share", "only mature, observable prior extensions"),
        ("club_position_mature_prior_extension_count_log1p", "mature_extension_history", "log(1 + same-position mature resolved extensions)", "unavailable when extension position is unknown"),
        ("club_position_prior_extension_temporary_share", "mature_extension_history", "smoothed same-position temporary-first share", "unavailable when extension position is unknown"),
    ]
    return pd.DataFrame(rows, columns=["feature", "block", "definition", "timing_or_availability_rule"])


def build_checks(profiles: pd.DataFrame, audits: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(check: str, passed: bool, observed: object, expected: object, note: str) -> None:
        rows.append({"check": check, "passed": bool(passed), "observed": observed, "expected": expected, "note": note})

    add("event_grain", profiles["capology_extension_event_id"].is_unique, int(profiles["capology_extension_event_id"].duplicated().sum()), 0, "One profile per canonical extension event.")
    add("anchor_count", len(profiles) == 2805, len(profiles), 2805, "Canonical extension universe is preserved.")
    transfer_dates = pd.to_datetime(audits["latest_transfer_date_used"], errors="coerce")
    signing_dates = pd.to_datetime(audits["signed_date"], errors="raise")
    add("transfers_strictly_predecision", bool((transfer_dates.dropna() < signing_dates.loc[transfer_dates.notna()]).all()), int((transfer_dates >= signing_dates).fillna(False).sum()), 0, "No same-day or future transfer enters a profile.")
    maturity_dates = pd.to_datetime(audits["latest_mature_extension_outcome_available_date"], errors="coerce")
    add("extension_outcomes_mature", bool((maturity_dates.dropna() <= signing_dates.loc[maturity_dates.notna()]).all()), int((maturity_dates > signing_dates).fillna(False).sum()), 0, "Every reused extension outcome was knowable at the decision date.")
    finite = np.isfinite(profiles[MODEL_FEATURES].to_numpy(float)[~np.isnan(profiles[MODEL_FEATURES].to_numpy(float))]).all()
    add("model_features_finite", bool(finite), bool(finite), True, "No infinite engineered value.")
    add("no_target_columns", not any(column.startswith("target_") or "movement_state" in column for column in profiles), [column for column in profiles if column.startswith("target_") or "movement_state" in column], [], "Feature table contains no future target.")
    add("general_profile_complete", profiles[GENERAL_FEATURES].notna().all(axis=None), int(profiles[GENERAL_FEATURES].isna().sum().sum()), 0, "No-history clubs receive explicit counts and priors, not missing values.")
    role_expected = profiles["canonical_position"].notna() & profiles["age_band_at_signing"].notna()
    role_observed = profiles[ROLE_FEATURES].notna().all(axis=1)
    add("role_availability_explicit", role_expected.equals(role_observed), int((role_expected != role_observed).sum()), 0, "Role block is unavailable only when position/age context is missing.")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    unchanged = all(sha256(ROOT / item["path"]) == item["sha256"] for item in freeze["files"])
    add("v1_frozen", unchanged, unchanged, True, "All frozen V1 files remain byte-identical.")
    return pd.DataFrame(rows)


def write_readme(output: Path, profiles: pd.DataFrame, counts: dict[str, int]) -> None:
    dev = profiles.loc[profiles["signed_year"].between(2020, 2023)]
    role_coverage = float(dev[ROLE_FEATURES].notna().all(axis=1).mean())
    mature_any = float(dev["club_mature_prior_extension_count"].gt(0).mean())
    text = f"""# Engine V2 lagged club-behavior profiles

**Status:** research feature build; not deployed and not evaluated on the final holdout.

This table converts dated transfer and earlier-extension histories into club
behavior available at each extension decision. Transfer evidence is restricted
to dates strictly before signing. Earlier extension outcomes are usable only
after the full {HORIZON_DAYS}-day horizon has matured.

## Coverage

- canonical extension anchors: **{len(profiles):,}**;
- relevant canonical transfer events scanned: **{counts['relevant_transfer_rows']:,}**;
- 2020–2023 role-conditioned profile coverage: **{role_coverage:.1%}**;
- 2020–2023 rows with at least one mature prior same-club extension: **{mature_any:.1%}**.

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
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    before = {item["path"]: sha256(ROOT / item["path"]) for item in json.loads(FREEZE.read_text())["files"]}
    profiles, audits, counts = build_profiles()
    dictionary = feature_dictionary()
    checks = build_checks(profiles, audits)

    profiles.to_csv(output / "lagged_club_behavior_features.csv", index=False)
    audits.to_csv(output / "asof_evidence_audit.csv", index=False)
    dictionary.to_csv(output / "feature_dictionary.csv", index=False)
    checks.to_csv(output / "build_checks.csv", index=False)
    write_json(output / "movement_target_contract.json", contract_payload())
    write_readme(output, profiles, counts)

    sources = []
    for path in [TRANSFER_PATH, PLAYER_PATH, EXTENSION_PATH, ROOT / "models" / "engine_v2" / "movement_target_contract.py"]:
        sources.append({"source": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(sources).to_csv(output / "source_manifest.csv", index=False)
    after = {path: sha256(ROOT / path) for path in before}
    summary = {
        **counts,
        "contract_version": CONTRACT_VERSION,
        "feature_rows": len(profiles),
        "model_features": MODEL_FEATURES,
        "development_years": [2020, 2021, 2022, 2023],
        "final_holdout_opened": False,
        "deployment_changed": False,
        "v1_unchanged": before == after,
        "checks_passed": int(checks["passed"].sum()),
        "checks_total": len(checks),
    }
    write_json(output / "run_summary.json", summary)
    manifest = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            manifest.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(output / "output_manifest.csv", index=False)
    if not checks["passed"].all():
        raise RuntimeError(checks.loc[~checks["passed"], ["check", "observed", "expected"]].to_dict("records"))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
