"""Independently verify the extension-centered contract integration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "contract_extension_integration"
CONTRACTS = ROOT / "Data" / "processed" / "capology_contracts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype("string").fillna("").str.strip().str.casefold().isin({"true", "1", "yes"})


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def main() -> None:
    required = {
        "extension_modeling_master.csv", "extension_seasonal_performance_long.csv",
        "extension_exact_sporting_windows.csv", "extension_valuation_landmarks.csv",
        "extension_departure_outcomes.csv", "join_coverage_summary.csv",
        "modeling_cohort_funnel.csv", "field_dictionary.csv", "build_checks.csv",
        "source_manifest.csv", "output_manifest.csv", "run_summary.json", "README.md",
    }
    missing = sorted(name for name in required if not (OUTPUT / name).exists())
    if missing:
        raise RuntimeError(f"Missing integration artifacts: {missing}")
    checks: list[dict] = []
    add(checks, "required_outputs", len(missing), 0, not missing, "All expected builder outputs exist.")

    source = pd.read_csv(CONTRACTS / "canonical_extension_events.csv", low_memory=False)
    master = pd.read_csv(OUTPUT / "extension_modeling_master.csv", low_memory=False)
    seasonal = pd.read_csv(OUTPUT / "extension_seasonal_performance_long.csv", low_memory=False)
    sporting = pd.read_csv(OUTPUT / "extension_exact_sporting_windows.csv", low_memory=False)
    valuations = pd.read_csv(OUTPUT / "extension_valuation_landmarks.csv", low_memory=False)
    departures = pd.read_csv(OUTPUT / "extension_departure_outcomes.csv", low_memory=False)
    funnel = pd.read_csv(OUTPUT / "modeling_cohort_funnel.csv")
    build_checks = pd.read_csv(OUTPUT / "build_checks.csv")

    source_ids = set(source["capology_extension_event_id"])
    master_ids = set(master["capology_extension_event_id"])
    add(checks, "anchor_row_count", len(master), len(source), len(master) == len(source), "Every clean extension is preserved once.")
    add(checks, "anchor_id_set", len(master_ids), len(source_ids), master_ids == source_ids, "Output anchor IDs exactly match source IDs.")
    add(checks, "anchor_id_unique", int(master["capology_extension_event_id"].duplicated().sum()), 0, not master["capology_extension_event_id"].duplicated().any(), "No join multiplication.")

    signed = pd.to_datetime(source["signed_date"], errors="coerce")
    expected_pre = np.where(
        (signed.dt.month >= 7) | ((signed.dt.month == 6) & (signed.dt.day == 30)),
        signed.dt.year - 1,
        signed.dt.year - 2,
    )
    expected_post = np.where(
        (signed.dt.month <= 6) | ((signed.dt.month == 7) & (signed.dt.day == 1)),
        signed.dt.year,
        signed.dt.year + 1,
    )
    pre_ok = np.array_equal(pd.to_numeric(source["pre_completed_season_start_year"]).to_numpy(), expected_pre)
    post_ok = np.array_equal(pd.to_numeric(source["first_full_post_season_start_year"]).to_numpy(), expected_post)
    add(checks, "pre_season_boundary", int(pre_ok), 1, pre_ok, "January-May excludes the unfinished domestic season.")
    add(checks, "post_full_season_boundary", int(post_ok), 1, post_ok, "First full post season follows the July 1 convention.")

    seasonal_key_ok = not seasonal.duplicated(["capology_extension_event_id", "window"]).any()
    sporting_key_ok = not sporting.duplicated(["capology_extension_event_id", "window"]).any()
    valuation_key_ok = not valuations.duplicated(["capology_extension_event_id", "landmark"]).any()
    add(checks, "seasonal_long_grain", len(seasonal), len(source) * 6, len(seasonal) == len(source) * 6 and seasonal_key_ok, "Six unique seasonal slots per event.")
    add(checks, "sporting_long_grain", len(sporting), len(source) * 4, len(sporting) == len(source) * 4 and sporting_key_ok, "Four unique exact-date windows per event.")
    add(checks, "valuation_long_grain", len(valuations), len(source) * 4, len(valuations) == len(source) * 4 and valuation_key_ok, "Four unique valuation landmarks per event.")
    add(checks, "departure_grain", len(departures), len(source), len(departures) == len(source) and not departures["capology_extension_event_id"].duplicated().any(), "One departure record per event.")
    funnel_ok = funnel["records"].diff().fillna(0).le(0).all()
    add(checks, "modeling_funnel_monotone", int(funnel_ok), 1, funnel_ok, "Cumulative funnel counts never increase.")

    role_ok = seasonal.loc[seasonal["window"].str.startswith("pre"), "information_role"].eq("predictor").all()
    role_ok &= seasonal.loc[seasonal["window"].str.startswith("post"), "information_role"].eq("outcome").all()
    role_ok &= sporting.loc[sporting["window"].eq("pre365"), "information_role"].eq("predictor").all()
    role_ok &= sporting.loc[sporting["window"].str.startswith("post"), "information_role"].eq("outcome").all()
    add(checks, "information_roles", int(role_ok), 1, role_ok, "Pre and post blocks are labeled consistently.")

    source_timing = source[["capology_extension_event_id", "signed_date"]].copy()
    source_timing["signed_date"] = pd.to_datetime(source_timing["signed_date"], errors="coerce")
    pre_season = seasonal.loc[seasonal["window"].str.startswith("pre")].merge(source_timing, on="capology_extension_event_id", how="left", validate="many_to_one")
    season_end = pd.to_datetime(pre_season["fbref_season_end_date"], errors="coerce")
    leakage = season_end.notna() & season_end.gt(pre_season["signed_date"])
    add(checks, "pre_season_date_leakage", int(leakage.sum()), 0, not leakage.any(), "No pre-signing seasonal record ends after signing.")

    minutes_cols = ["same_club_all_competition_minutes", "same_club_same_big5_competition_minutes"]
    nonnegative = all(pd.to_numeric(sporting[c], errors="coerce").fillna(0).ge(0).all() for c in minutes_cols)
    share_cols = ["same_club_all_competition_opportunity_share", "same_club_same_big5_competition_opportunity_share"]
    shares_ok = all(pd.to_numeric(sporting[c], errors="coerce").dropna().between(0, 1.05).all() for c in share_cols)
    eligibility_expected = as_bool(sporting["window_fully_observable"]) & pd.to_numeric(sporting["same_club_all_competition_games"], errors="coerce").ge(10)
    eligibility_ok = eligibility_expected.eq(as_bool(sporting["window_evidence_eligible_10_club_games"])).all()
    add(checks, "sporting_nonnegative", int(nonnegative), 1, nonnegative, "Minutes are nonnegative.")
    add(checks, "opportunity_share_bounds", int(shares_ok), 1, shares_ok, "Shares are capped at the documented 1.05 tolerance.")
    add(checks, "sporting_eligibility_rule", int(eligibility_ok), 1, eligibility_ok, "Eligibility requires observability and at least 10 club games.")

    wide_bool = lambda c: as_bool(master[c])
    observability = (~wide_bool("post_y3_window_fully_observable") | wide_bool("post_y2_window_fully_observable")).all()
    observability &= (~wide_bool("post_y2_window_fully_observable") | wide_bool("post_y1_window_fully_observable")).all()
    add(checks, "sporting_horizon_observability", int(observability), 1, observability, "Longer observable windows imply shorter observable windows.")

    valuation_eligible = as_bool(valuations["strict_timing_eligible"])
    valuation_rule = (~valuation_eligible | (as_bool(valuations["horizon_fully_observable"]) & pd.to_numeric(valuations["gap_days"], errors="coerce").abs().le(np.where(valuations["landmark"].eq("at_signing"), 365, 90)))).all()
    add(checks, "valuation_timing_rule", int(valuation_rule), 1, valuation_rule, "Eligible landmarks satisfy horizon and timing tolerances.")

    joined_departures = departures.merge(source_timing, on="capology_extension_event_id", how="left", validate="one_to_one")
    any_date = pd.to_datetime(joined_departures["first_any_outbound_date"], errors="coerce")
    perm_date = pd.to_datetime(joined_departures["first_permanent_outbound_date"], errors="coerce")
    departure_dates_ok = (~any_date.notna() | any_date.gt(joined_departures["signed_date"])).all() and (~perm_date.notna() | perm_date.gt(joined_departures["signed_date"])).all()
    add(checks, "departure_dates_after_signing", int(departure_dates_ok), 1, departure_dates_ok, "All selected departures occur strictly after signing.")
    r365 = pd.to_numeric(master["uninterrupted_same_club_spell_365d"], errors="coerce")
    r730 = pd.to_numeric(master["uninterrupted_same_club_spell_730d"], errors="coerce")
    r1095 = pd.to_numeric(master["uninterrupted_same_club_spell_1095d"], errors="coerce")
    retention_ok = (~(r1095.eq(1) & ~r730.eq(1))).all() and (~(r730.eq(1) & ~r365.eq(1))).all()
    add(checks, "retention_monotone", int(retention_ok), 1, retention_ok, "Longer uninterrupted spells imply shorter spells.")

    source_manifest = pd.read_csv(OUTPUT / "source_manifest.csv")
    source_hash_failures = []
    for row in source_manifest.itertuples(index=False):
        path = ROOT / row.source_file
        if not path.exists() or sha256(path) != row.sha256:
            source_hash_failures.append(row.source_file)
    add(checks, "source_hashes", len(source_hash_failures), 0, not source_hash_failures, f"Failures: {source_hash_failures or 'none'}")
    output_manifest = pd.read_csv(OUTPUT / "output_manifest.csv")
    output_failures = []
    for row in output_manifest.itertuples(index=False):
        if row.output_file in {"independent_verification.csv", "independent_verification.json"}:
            continue
        path = OUTPUT / row.output_file
        if not path.exists() or sha256(path) != row.sha256:
            output_failures.append(row.output_file)
    add(checks, "builder_output_hashes", len(output_failures), 0, not output_failures, f"Failures: {output_failures or 'none'}")
    build_ok = as_bool(build_checks["passed"]).all()
    add(checks, "builder_checks", int(build_ok), 1, build_ok, "All builder checks passed.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "extension_events_verified": len(master),
        "master_columns": len(master.columns),
    }
    (OUTPUT / "independent_verification.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file() and path.name != "output_manifest.csv")
    manifest = pd.DataFrame([{"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files])
    manifest.to_csv(OUTPUT / "output_manifest.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Independent contract-integration verification failed: {failed}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
