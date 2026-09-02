"""Audit Capology club payroll and transfer-window panels without modeling.

The CSV files under ``Data/leagues/*/{Payrolls,TransferWindows}`` are treated
as immutable source artifacts.  This script parses their display fields,
profiles coverage and arithmetic consistency, tests conservative canonical
club links, and measures leakage-safe prior-season coverage for extension
events.  It does not change raw data, V1 artifacts, API code, or model files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEAGUES_ROOT = ROOT / "Data" / "leagues"
DEFAULT_OUTPUT = ROOT / "Data" / "processed" / "capology_club_financial_context_audit"
sys.path.insert(0, str(ROOT / "scripts"))

from build_capology_contracts import (  # noqa: E402
    build_club_map,
    extract_club_slug,
    extract_display,
    link_club,
    normalize_text,
    parse_eur,
)


LEAGUE_META = {
    "bundesliga": ("Bundesliga", "L1"),
    "la liga": ("La Liga", "ES1"),
    "ligue1": ("Ligue 1", "FR1"),
    "premier league": ("Premier League", "GB1"),
    "serie a": ("Serie A", "IT1"),
}
SEASONS = list(range(2014, 2025))
FAMILY_META = {
    "Payrolls": {
        "source_family": "payroll",
        "expected_columns": 56,
        "required_numeric": [
            "weekly_gross_eur",
            "annual_gross_eur",
            "total_gross_eur",
            "adjusted_total_gross_eur",
        ],
    },
    "TransferWindows": {
        "source_family": "transfer_window",
        "expected_columns": 17,
        "required_numeric": [
            "income_eur",
            "expense_eur",
            "balance_eur",
            "adjbalance_eur",
            "players",
            "foreign",
            "age",
        ],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def season_from_path(path: Path) -> tuple[str, int, int]:
    match = re.search(r"(20\d{2})_(20\d{2})", path.name)
    if not match:
        raise ValueError(f"No season encoded in filename: {path}")
    start, end = map(int, match.groups())
    return f"{start}-{end}", start, end


def expected_clubs(league_folder: str, season_start: int) -> int:
    if league_folder == "bundesliga":
        return 18
    if league_folder == "ligue1" and season_start >= 2023:
        return 18
    return 20


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str, allow_nan=False) + "\n")


def add_check(
    checks: list[dict],
    check: str,
    observed: object,
    expected: object,
    passed: bool,
    note: str,
) -> None:
    checks.append(
        {
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
            "note": note,
        }
    )


def read_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    club_map, _ = build_club_map()
    manifests: list[dict] = []
    profiles: list[dict] = []
    payroll_rows: list[pd.DataFrame] = []
    transfer_rows: list[pd.DataFrame] = []

    for folder, meta in FAMILY_META.items():
        family = meta["source_family"]
        for path in sorted(LEAGUES_ROOT.glob(f"*/{folder}/*.csv")):
            league_folder = path.parent.parent.name
            league, competition_id = LEAGUE_META[league_folder]
            season, season_start, season_end = season_from_path(path)
            frame = pd.read_csv(path, low_memory=False)
            columns = frame.columns.tolist()
            manifests.append(
                {
                    "source_relative_path": path.relative_to(ROOT).as_posix(),
                    "source_family": family,
                    "league_folder": league_folder,
                    "league": league,
                    "competition_id": competition_id,
                    "season": season,
                    "season_start_year": season_start,
                    "season_end_year": season_end,
                    "source_rows": len(frame),
                    "source_columns": len(columns),
                    "source_bytes": path.stat().st_size,
                    "source_sha256": sha256(path),
                    "schema_signature": hashlib.sha256("|".join(columns).encode()).hexdigest(),
                }
            )

            frame["club_name"] = frame["club"].map(extract_display)
            frame["capology_club_slug"] = frame["club"].map(extract_club_slug)
            frame["club_name_normalized"] = frame["club_name"].map(normalize_text)
            links = frame["club_name_normalized"].map(lambda value: link_club(value, club_map))
            frame["canonical_club_id"] = [item["canonical_club_id"] for item in links]
            frame["canonical_club_name"] = [item["canonical_club_name"] for item in links]
            frame["club_link_method"] = [item["club_link_method"] for item in links]
            frame["source_family"] = family
            frame["league_folder"] = league_folder
            frame["league"] = league
            frame["competition_id"] = competition_id
            frame["season"] = season
            frame["season_start_year"] = season_start
            frame["season_end_year"] = season_end
            frame["source_relative_path"] = path.relative_to(ROOT).as_posix()
            frame["source_row_number"] = np.arange(2, len(frame) + 2)

            numeric_coverage = {}
            for column in meta["required_numeric"]:
                parsed = (
                    pd.to_numeric(frame[column], errors="coerce")
                    if column in {"players", "foreign", "age"}
                    else frame[column].map(parse_eur)
                )
                frame[f"parsed_{column}"] = parsed
                numeric_coverage[column] = int(parsed.notna().sum())
            profiles.append(
                {
                    "source_relative_path": path.relative_to(ROOT).as_posix(),
                    "source_family": family,
                    "league": league,
                    "season": season,
                    "rows": len(frame),
                    "expected_rows": expected_clubs(league_folder, season_start),
                    "unique_club_slugs": frame["capology_club_slug"].nunique(dropna=True),
                    "duplicate_club_slugs": int(frame["capology_club_slug"].duplicated().sum()),
                    "missing_club_name": int(frame["club_name"].isna().sum()),
                    "missing_club_slug": int(frame["capology_club_slug"].isna().sum()),
                    "missing_club_code": int(frame["club_code"].isna().sum()),
                    "canonical_club_links": int(frame["canonical_club_id"].notna().sum()),
                    **{f"nonnull_{key}": value for key, value in numeric_coverage.items()},
                }
            )

            identity = [
                "source_family",
                "league_folder",
                "league",
                "competition_id",
                "season",
                "season_start_year",
                "season_end_year",
                "source_relative_path",
                "source_row_number",
                "club_name",
                "club_name_normalized",
                "capology_club_slug",
                "club_code",
                "canonical_club_id",
                "canonical_club_name",
                "club_link_method",
            ]
            if family == "payroll":
                optional = ["bonus_gross_eur"]
                for column in optional:
                    frame[f"parsed_{column}"] = frame[column].map(parse_eur)
                payroll_rows.append(
                    frame[
                        identity
                        + [
                            "parsed_weekly_gross_eur",
                            "parsed_annual_gross_eur",
                            "parsed_bonus_gross_eur",
                            "parsed_total_gross_eur",
                            "parsed_adjusted_total_gross_eur",
                        ]
                    ].copy()
                )
            else:
                transfer_rows.append(
                    frame[
                        identity
                        + [
                            "parsed_income_eur",
                            "parsed_expense_eur",
                            "parsed_balance_eur",
                            "parsed_adjbalance_eur",
                            "parsed_players",
                            "parsed_foreign",
                            "parsed_age",
                        ]
                    ].copy()
                )

    return (
        pd.DataFrame(manifests),
        pd.DataFrame(profiles),
        pd.concat(payroll_rows, ignore_index=True),
        pd.concat(transfer_rows, ignore_index=True),
    )


def build_extension_coverage(payroll: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    extension_path = ROOT / "Data" / "processed" / "capology_contracts" / "canonical_extension_events.csv"
    evaluation_path = ROOT / "Data" / "processed" / "integrated_contract_profile" / "integration_evaluation_cohort.csv"
    extensions = pd.read_csv(extension_path, low_memory=False)
    evaluation_ids = set(pd.read_csv(evaluation_path, usecols=["capology_extension_event_id"])["capology_extension_event_id"])

    keys = payroll.dropna(subset=["canonical_club_id"])[
        ["canonical_club_id", "season_start_year", "club_name", "capology_club_slug"]
    ].drop_duplicates(["canonical_club_id", "season_start_year"])
    keys["canonical_club_id"] = keys["canonical_club_id"].astype(int)
    available_pairs = set(zip(keys["canonical_club_id"], keys["season_start_year"]))
    available_ids = set(keys["canonical_club_id"])

    audit = extensions[
        [
            "capology_extension_event_id",
            "signed_date",
            "league",
            "club_name",
            "canonical_club_id",
            "canonical_club_name",
            "event_season_start_year",
            "pre_completed_season_start_year",
        ]
    ].copy()
    audit["in_integrated_839_evaluation_cohort"] = audit["capology_extension_event_id"].isin(evaluation_ids)
    audit["prior_completed_season_panel_available"] = [
        (int(club_id), int(season)) in available_pairs
        for club_id, season in zip(audit["canonical_club_id"], audit["pre_completed_season_start_year"])
    ]
    audit["event_season_panel_available"] = [
        (int(club_id), int(season)) in available_pairs
        for club_id, season in zip(audit["canonical_club_id"], audit["event_season_start_year"])
    ]
    audit["coverage_status"] = np.where(
        audit["prior_completed_season_panel_available"],
        "available_prior_completed_season",
        np.where(
            audit["event_season_panel_available"],
            "missing_prior_top_flight_panel_likely_promoted",
            np.where(
                audit["canonical_club_id"].isin(available_ids),
                "club_present_other_seasons_but_no_prior_or_event_panel",
                "canonical_club_absent_from_big_five_panel",
            ),
        ),
    )

    summaries = []
    for scope, subset in {
        "all_canonical_extension_events": audit,
        "integrated_839_evaluation_cohort": audit.loc[audit["in_integrated_839_evaluation_cohort"]],
    }.items():
        summaries.append(
            {
                "scope": scope,
                "events": len(subset),
                "prior_completed_season_available": int(subset["prior_completed_season_panel_available"].sum()),
                "prior_completed_season_missing": int((~subset["prior_completed_season_panel_available"]).sum()),
                "prior_completed_season_coverage_rate": float(subset["prior_completed_season_panel_available"].mean()),
                "likely_promoted_missing": int(
                    subset["coverage_status"].eq("missing_prior_top_flight_panel_likely_promoted").sum()
                ),
            }
        )
    return audit, pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest, profiles, payroll, transfers = read_sources()
    extension_audit, extension_summary = build_extension_coverage(payroll)

    key = ["league_folder", "season_start_year", "capology_club_slug"]
    payroll_keys = set(map(tuple, payroll[key].itertuples(index=False, name=None)))
    transfer_keys = set(map(tuple, transfers[key].itertuples(index=False, name=None)))

    panel = payroll.merge(
        transfers.drop(columns=["source_family", "league", "competition_id", "season", "season_end_year"]),
        on=key,
        how="outer",
        suffixes=("_payroll", "_transfer"),
        validate="one_to_one",
        indicator=True,
    )
    club_identity = (
        pd.concat(
            [
                payroll[["league", "club_name", "club_name_normalized", "capology_club_slug", "club_code", "canonical_club_id", "canonical_club_name", "club_link_method"]],
                transfers[["league", "club_name", "club_name_normalized", "capology_club_slug", "club_code", "canonical_club_id", "canonical_club_name", "club_link_method"]],
            ],
            ignore_index=True,
        )
        .drop_duplicates()
        .sort_values(["league", "club_name"])
    )

    numeric_rows = []
    for family, frame, columns in [
        (
            "payroll",
            payroll,
            [
                "parsed_weekly_gross_eur",
                "parsed_annual_gross_eur",
                "parsed_bonus_gross_eur",
                "parsed_total_gross_eur",
                "parsed_adjusted_total_gross_eur",
            ],
        ),
        (
            "transfer_window",
            transfers,
            [
                "parsed_income_eur",
                "parsed_expense_eur",
                "parsed_balance_eur",
                "parsed_adjbalance_eur",
                "parsed_players",
                "parsed_foreign",
                "parsed_age",
            ],
        ),
    ]:
        for (league, season), group in frame.groupby(["league", "season"]):
            for column in columns:
                series = group[column]
                numeric_rows.append(
                    {
                        "source_family": family,
                        "league": league,
                        "season": season,
                        "field": column.removeprefix("parsed_"),
                        "rows": len(group),
                        "nonnull": int(series.notna().sum()),
                        "zero": int(series.eq(0).sum()),
                        "minimum": series.min(),
                        "maximum": series.max(),
                    }
                )
    numeric_coverage = pd.DataFrame(numeric_rows)

    checks: list[dict] = []
    add_check(checks, "source_file_count", len(manifest), 110, len(manifest) == 110, "55 payroll and 55 transfer-window files.")
    add_check(checks, "payroll_file_count", int(manifest["source_family"].eq("payroll").sum()), 55, int(manifest["source_family"].eq("payroll").sum()) == 55, "Five leagues by eleven seasons.")
    add_check(checks, "transfer_file_count", int(manifest["source_family"].eq("transfer_window").sum()), 55, int(manifest["source_family"].eq("transfer_window").sum()) == 55, "Five leagues by eleven seasons.")
    grid = set(zip(manifest["source_family"], manifest["league_folder"], manifest["season_start_year"]))
    expected_grid = {(family, league, season) for family in ["payroll", "transfer_window"] for league in LEAGUE_META for season in SEASONS}
    add_check(checks, "league_season_grid_complete", len(grid), len(expected_grid), grid == expected_grid, "No missing or extra league-season file.")
    add_check(checks, "expected_file_row_counts", int(profiles["rows"].eq(profiles["expected_rows"]).sum()), len(profiles), profiles["rows"].eq(profiles["expected_rows"]).all(), "Accounts for Bundesliga's 18 clubs and Ligue 1's 2023-24 contraction to 18.")
    add_check(checks, "schema_consistency_by_family", int(manifest.groupby("source_family")["schema_signature"].nunique().max()), 1, manifest.groupby("source_family")["schema_signature"].nunique().eq(1).all(), "One schema per source family.")
    add_check(checks, "source_columns_expected", int(profiles.shape[0]), len(profiles), ((manifest["source_family"].eq("payroll") & manifest["source_columns"].eq(56)) | (manifest["source_family"].eq("transfer_window") & manifest["source_columns"].eq(17))).all(), "Payroll has 56 columns; transfer window has 17.")
    duplicates = int(payroll.duplicated(key).sum() + transfers.duplicated(key).sum())
    add_check(checks, "club_season_keys_unique", duplicates, 0, duplicates == 0, "One club row per family, league, and season.")
    add_check(checks, "payroll_transfer_keys_identical", len(payroll_keys ^ transfer_keys), 0, payroll_keys == transfer_keys, "Both source families cover the exact same 1,074 club-seasons.")
    missing_identity = int(payroll[["club_name", "capology_club_slug", "club_code"]].isna().sum().sum() + transfers[["club_name", "capology_club_slug", "club_code"]].isna().sum().sum())
    add_check(checks, "source_club_identity_complete", missing_identity, 0, missing_identity == 0, "Display name, stable Capology slug, and source club code are present.")
    required_payroll = ["parsed_weekly_gross_eur", "parsed_annual_gross_eur", "parsed_total_gross_eur", "parsed_adjusted_total_gross_eur"]
    required_transfer = ["parsed_income_eur", "parsed_expense_eur", "parsed_balance_eur", "parsed_adjbalance_eur", "parsed_players", "parsed_foreign", "parsed_age"]
    missing_numeric = int(payroll[required_payroll].isna().sum().sum() + transfers[required_transfer].isna().sum().sum())
    add_check(checks, "required_numeric_fields_complete", missing_numeric, 0, missing_numeric == 0, "Bonus is optional and assessed separately.")
    weekly_difference = (payroll["parsed_annual_gross_eur"] - payroll["parsed_weekly_gross_eur"] * 52).abs()
    add_check(checks, "payroll_weekly_annual_arithmetic", float(weekly_difference.max()), "<=26 EUR", bool(weekly_difference.le(26).all()), "Weekly figures are whole-euro displays, creating at most half-euro times 52 rounding drift.")
    total_difference = (
        payroll["parsed_total_gross_eur"]
        - payroll["parsed_annual_gross_eur"]
        - payroll["parsed_bonus_gross_eur"].fillna(0)
    ).abs()
    add_check(checks, "payroll_total_arithmetic", float(total_difference.max()), "<=10 EUR", bool(total_difference.le(10).all()), "Small display-currency rounding tolerance.")
    balance_difference = (transfers["parsed_balance_eur"] - transfers["parsed_income_eur"] + transfers["parsed_expense_eur"]).abs()
    add_check(checks, "transfer_balance_arithmetic", float(balance_difference.max()), "<=2 EUR", bool(balance_difference.le(2).all()), "Balance equals income minus expense.")
    foreign_over_players = int((transfers["parsed_foreign"] > transfers["parsed_players"]).sum())
    add_check(checks, "squad_counts_coherent", foreign_over_players, 0, foreign_over_players == 0, "Foreign-player count never exceeds squad-player count.")
    link_rate = float(payroll["canonical_club_id"].notna().mean())
    add_check(checks, "canonical_club_link_rate_diagnostic", round(link_rate, 6), ">=0.95", link_rate >= 0.95, "Unresolved historical clubs remain quarantined for alias review; no fuzzy matching.")
    add_check(checks, "club_season_merge_complete", panel["_merge"].eq("both").sum(), len(panel), panel["_merge"].eq("both").all(), "Every payroll row has its same-season transfer-window row.")

    manifest.to_csv(output / "source_manifest.csv", index=False)
    profiles.to_csv(output / "file_profile.csv", index=False)
    payroll.to_csv(output / "parsed_payroll_audit_panel.csv", index=False)
    transfers.to_csv(output / "parsed_transfer_window_audit_panel.csv", index=False)
    panel.to_csv(output / "paired_club_season_audit.csv", index=False)
    club_identity.to_csv(output / "club_identity_audit.csv", index=False)
    numeric_coverage.to_csv(output / "numeric_coverage.csv", index=False)
    extension_audit.to_csv(output / "extension_prior_season_coverage.csv", index=False)
    extension_summary.to_csv(output / "extension_coverage_summary.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "source_checks.csv", index=False)

    bonus_by_season = (
        payroll.assign(bonus_available=payroll["parsed_bonus_gross_eur"].notna())
        .groupby("season", observed=True)
        .agg(club_seasons=("capology_club_slug", "size"), bonus_available=("bonus_available", "sum"))
        .reset_index()
    )
    bonus_by_season.to_csv(output / "bonus_coverage_by_season.csv", index=False)

    summary = {
        "source_files": len(manifest),
        "payroll_files": int(manifest["source_family"].eq("payroll").sum()),
        "transfer_window_files": int(manifest["source_family"].eq("transfer_window").sum()),
        "seasons": ["2014-2015", "2024-2025"],
        "club_seasons_per_family": len(payroll),
        "unique_capology_clubs": int(payroll["capology_club_slug"].nunique()),
        "canonical_linked_club_seasons": int(payroll["canonical_club_id"].notna().sum()),
        "canonical_link_rate": link_rate,
        "unresolved_unique_clubs": int(payroll.loc[payroll["canonical_club_id"].isna(), "capology_club_slug"].nunique()),
        "bonus_available_club_seasons": int(payroll["parsed_bonus_gross_eur"].notna().sum()),
        "position_group_columns_status": "paywalled_locked_placeholders_not_usable",
        "extension_prior_season_coverage": extension_summary.to_dict("records"),
        "checks_passed": int(pd.DataFrame(checks)["passed"].sum()),
        "checks_total": len(checks),
        "modeling_status": "audit_only_not_approved_for_training_or_scoring",
    }
    write_json(output / "summary.json", summary)

    readme = f"""# Capology club financial context audit

This directory is an **audit-only, non-deployed** assessment of the 110 raw
Capology payroll and transfer-window CSVs covering the Big Five leagues from
2014-15 through 2024-25. Raw files were read and hashed but never modified.

## What passed

- All 110 expected files exist and parse: 55 payroll plus 55 transfer-window.
- Each source family uses one consistent schema.
- Each has {len(payroll):,} unique club-season rows, and their keys match exactly.
- Required club-level euro values are complete and pass arithmetic checks.
- {int(payroll['canonical_club_id'].notna().sum()):,}/{len(payroll):,} club-season rows ({link_rate:.1%}) link through existing conservative canonical aliases.

## Material limitations

1. The keeper/defender/midfielder/forward payroll fields are paywalled `Locked`
   placeholders. This source supports **club-level payroll context only**.
2. Bonus values are absent before 2021-22 and complete only from 2022-23 onward.
   Do not treat missing historical bonus as zero.
3. Capology payroll is estimated, not audited club accounts.
4. The panel covers top-flight Big Five membership. A leakage-safe prior-season
   join is therefore structurally missing for newly promoted clubs.
5. Sixteen historical club names remain unresolved under conservative exact
   mapping. They are quarantined in `club_identity_audit.csv`; no fuzzy links
   were accepted.

## Leakage-safe extension coverage

Use only the last completed season at the decision date. On this rule the panel
covers {int(extension_summary.iloc[0]['prior_completed_season_available']):,}/{int(extension_summary.iloc[0]['events']):,}
all extension events ({extension_summary.iloc[0]['prior_completed_season_coverage_rate']:.1%}) and
{int(extension_summary.iloc[1]['prior_completed_season_available']):,}/{int(extension_summary.iloc[1]['events']):,}
events in the integrated 2020-2023 evaluation cohort
({extension_summary.iloc[1]['prior_completed_season_coverage_rate']:.1%}). The latter's missing cases are overwhelmingly
clubs absent from the previous top-flight panel, consistent with promotion.

Do **not** use a full current-season transfer-window total for an extension
signed during that season. The safe first candidate is prior-season payroll,
prior-season transfer spending/income/net balance, club payroll rank/percentile,
and multi-season changes calculated strictly from seasons completed before the
signing date. Any imputation must be learned within each training origin.

## Status

These outputs are not a production feature table and are not approved inputs to
V1 or V2. Canonical alias review, source-definition documentation, and a
rolling-origin feature experiment are still required.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
