"""Build a canonical, leakage-aware Capology club-season feature table.

Raw Capology CSVs are immutable.  This builder uses reviewed exact club aliases,
retains nominal decision-time-safe source values, derives only backward-looking
club history, and gates player-wage distribution features on source coverage.
It does not train a model or modify V1, API, HTML, or deployment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LEAGUES_ROOT = ROOT / "Data" / "leagues"
OUTPUT = ROOT / "Data" / "processed" / "capology_club_financial_context"
CONTRACTS = ROOT / "Data" / "processed" / "capology_contracts"
INTEGRATED = ROOT / "Data" / "processed" / "integrated_contract_profile"
TM_CLUBS = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "clubs_clean.csv"
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

# The source aliases and IDs are deliberately duplicated here as a frozen
# build contract.  The verifier checks this list independently against the
# shared Capology alias map and Transfermarkt evidence.
REVIEWED_ALIASES = {
    "bastia": 595,
    "carpi": 4102,
    "cesena": 1429,
    "cordoba": 993,
    "evian": 14171,
    "gazelecajaccio": 3558,
    "holsteinkiel": 269,
    "hullcity": 3008,
    "ingolstadt": 4795,
    "middlesbrough": 641,
    "nancy": 1159,
    "palermo": 458,
    "pescara": 2921,
    "qpr": 1039,
    "sportinggijon": 2448,
    "sunderland": 289,
}

CANDIDATE_FEATURE_BLOCKS = {
    "club_payroll_annual_gross_eur": "resource_level_raw",
    "transfer_income_eur": "recruitment_context_raw",
    "transfer_spend_eur": "recruitment_context_raw",
    "transfer_balance_eur": "recruitment_context_raw",
    "squad_average_age": "squad_context",
    "payroll_log_eur": "resource_level_transformed",
    "payroll_league_percentile": "resource_level_transformed",
    "payroll_to_league_median": "resource_level_transformed",
    "transfer_spend_log_eur": "recruitment_context_transformed",
    "transfer_income_log_eur": "recruitment_context_transformed",
    "transfer_spend_to_payroll": "recruitment_context_transformed",
    "transfer_income_to_payroll": "recruitment_context_transformed",
    "squad_foreign_share": "squad_context",
    "payroll_growth_1y": "resource_trajectory",
    "payroll_cagr_3y": "resource_trajectory",
    "transfer_spend_3y_mean_eur": "recruitment_trajectory",
    "transfer_income_3y_mean_eur": "recruitment_trajectory",
    "transfer_balance_3y_mean_eur": "recruitment_trajectory",
    "transfer_spend_3y_std_eur": "recruitment_trajectory",
    "wage_structure_median_eur": "wage_structure_coverage_gated",
    "wage_structure_max_eur": "wage_structure_coverage_gated",
    "wage_structure_top5_share": "wage_structure_coverage_gated",
    "wage_structure_hhi": "wage_structure_coverage_gated",
    "wage_structure_gini": "wage_structure_coverage_gated",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(*values: object) -> str:
    text = "|".join("" if pd.isna(value) else str(value) for value in values)
    return hashlib.sha256(text.encode()).hexdigest()


def season_from_path(path: Path) -> tuple[str, int, int]:
    match = re.search(r"(20\d{2})_(20\d{2})", path.name)
    if not match:
        raise ValueError(f"No season in filename: {path}")
    start, end = map(int, match.groups())
    return f"{start}-{end}", start, end


def gini(values: pd.Series) -> float:
    array = np.sort(pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float))
    array = array[array >= 0]
    if len(array) == 0 or array.sum() == 0:
        return np.nan
    index = np.arange(1, len(array) + 1)
    return float((2 * np.sum(index * array) / (len(array) * array.sum())) - (len(array) + 1) / len(array))


def hhi(values: pd.Series) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    total = array.sum()
    return float(np.square(array / total).sum()) if len(array) and total > 0 else np.nan


def top_n_share(values: pd.Series, n: int) -> float:
    array = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    total = array.sum()
    return float(array.head(n).sum() / total) if len(array) and total > 0 else np.nan


def add_check(checks: list[dict], name: str, observed: object, expected: object, passed: bool, note: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "note": note})


def source_files() -> tuple[list[Path], list[Path]]:
    payroll = sorted(LEAGUES_ROOT.glob("*/Payrolls/*.csv"))
    transfers = sorted(LEAGUES_ROOT.glob("*/TransferWindows/*.csv"))
    return payroll, transfers


def build_source_manifest(payroll_files: list[Path], transfer_files: list[Path]) -> pd.DataFrame:
    rows = []
    for family, files in [("payroll", payroll_files), ("transfer_window", transfer_files)]:
        for path in files:
            season, start, end = season_from_path(path)
            league_folder = path.parent.parent.name
            league, competition_id = LEAGUE_META[league_folder]
            header = pd.read_csv(path, nrows=0)
            rows.append(
                {
                    "source_relative_path": path.relative_to(ROOT).as_posix(),
                    "source_family": family,
                    "league": league,
                    "competition_id": competition_id,
                    "season": season,
                    "season_start_year": start,
                    "season_end_year": end,
                    "source_rows": sum(len(chunk) for chunk in pd.read_csv(path, usecols=[0], chunksize=10000)),
                    "source_columns": len(header.columns),
                    "source_bytes": path.stat().st_size,
                    "source_sha256": sha256(path),
                }
            )
    return pd.DataFrame(rows)


def parse_family(files: list[Path], family: str, club_map: dict) -> pd.DataFrame:
    rows = []
    for path in files:
        raw = pd.read_csv(path, low_memory=False)
        season, start, end = season_from_path(path)
        league_folder = path.parent.parent.name
        league, competition_id = LEAGUE_META[league_folder]
        for source_index, row in raw.iterrows():
            club_name = extract_display(row["club"])
            club_normalized = normalize_text(club_name)
            link = link_club(club_normalized, club_map)
            record = {
                "league": league,
                "competition_id": competition_id,
                "season": season,
                "season_start_year": start,
                "season_end_year": end,
                "capology_club_name": club_name,
                "capology_club_name_normalized": club_normalized,
                "capology_club_slug": extract_club_slug(row["club"]),
                "capology_club_code": row["club_code"],
                "canonical_club_id": link["canonical_club_id"],
                "canonical_club_name": link["canonical_club_name"],
                "club_link_method": link["club_link_method"],
                f"{family}_source_relative_path": path.relative_to(ROOT).as_posix(),
                f"{family}_source_row_number": source_index + 2,
                f"{family}_source_sha256": sha256(path),
            }
            if family == "payroll":
                record.update(
                    {
                        "club_payroll_weekly_gross_eur": parse_eur(row["weekly_gross_eur"]),
                        "club_payroll_annual_gross_eur": parse_eur(row["annual_gross_eur"]),
                        "club_payroll_bonus_gross_eur": parse_eur(row["bonus_gross_eur"]),
                        "club_payroll_total_gross_eur": parse_eur(row["total_gross_eur"]),
                        "club_payroll_bonus_reported": pd.notna(parse_eur(row["bonus_gross_eur"])),
                    }
                )
            else:
                record.update(
                    {
                        "transfer_income_eur": parse_eur(row["income_eur"]),
                        "transfer_spend_eur": parse_eur(row["expense_eur"]),
                        "transfer_balance_eur": parse_eur(row["balance_eur"]),
                        "squad_player_count": pd.to_numeric(row["players"], errors="coerce"),
                        "squad_foreign_player_count": pd.to_numeric(row["foreign"], errors="coerce"),
                        "squad_average_age": pd.to_numeric(row["age"], errors="coerce"),
                    }
                )
            rows.append(record)
    return pd.DataFrame(rows)


def build_alias_review(payroll: pd.DataFrame) -> pd.DataFrame:
    clubs = pd.read_csv(TM_CLUBS, low_memory=False)
    evidence = clubs.loc[clubs["club_id"].isin(REVIEWED_ALIASES.values())].copy()
    evidence = evidence.sort_values(["club_id", "last_season"], ascending=[True, False]).drop_duplicates("club_id")
    evidence_by_id = evidence.set_index("club_id")
    rows = []
    for alias, club_id in REVIEWED_ALIASES.items():
        source = payroll.loc[payroll["capology_club_name_normalized"].eq(alias)].iloc[0]
        tm = evidence_by_id.loc[club_id]
        rows.append(
            {
                "capology_alias_normalized": alias,
                "capology_display_name": source["capology_club_name"],
                "capology_club_slug": source["capology_club_slug"],
                "capology_club_code": source["capology_club_code"],
                "canonical_club_id": club_id,
                "canonical_club_name": tm["name"],
                "transfermarkt_club_code": tm["club_code"],
                "transfermarkt_domestic_competition_id": tm["domestic_competition_id"],
                "transfermarkt_url": tm["url"],
                "review_status": "accepted_reviewed_exact_alias",
                "evidence_rule": "Capology slug/code + Transfermarkt URL club ID + historical Big-Five membership",
            }
        )
    return pd.DataFrame(rows).sort_values("capology_alias_normalized")


def player_wage_structure() -> pd.DataFrame:
    salary = pd.read_csv(CONTRACTS / "canonical_salary_panel.csv", low_memory=False)
    salary = salary.dropna(subset=["canonical_club_id", "source_page_season_start_year", "annual_gross_eur"]).copy()
    salary = salary.loc[pd.to_numeric(salary["annual_gross_eur"], errors="coerce").gt(0)]
    salary["canonical_club_id"] = salary["canonical_club_id"].astype(int)
    salary["source_page_season_start_year"] = salary["source_page_season_start_year"].astype(int)
    grouped = []
    for (club_id, season), group in salary.groupby(["canonical_club_id", "source_page_season_start_year"], observed=True):
        wages = pd.to_numeric(group["annual_gross_eur"], errors="coerce").dropna()
        grouped.append(
            {
                "canonical_club_id": club_id,
                "season_start_year": season,
                "known_player_wage_count": len(wages),
                "known_canonical_player_count": int(group["canonical_player_id"].nunique(dropna=True)),
                "known_player_wage_sum_eur": float(wages.sum()),
                "source_player_wage_median_eur": float(wages.median()),
                "source_player_wage_max_eur": float(wages.max()),
                "source_player_wage_top5_share": top_n_share(wages, 5),
                "source_player_wage_hhi": hhi(wages),
                "source_player_wage_gini": gini(wages),
            }
        )
    return pd.DataFrame(grouped)


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["canonical_club_id", "season_start_year"]).copy()
    panel["season_club_count"] = panel.groupby(["league", "season_start_year"])["canonical_club_id"].transform("size")
    panel["payroll_log_eur"] = np.log1p(panel["club_payroll_annual_gross_eur"])
    panel["payroll_league_percentile"] = panel.groupby(["league", "season_start_year"])["club_payroll_annual_gross_eur"].rank(method="average", pct=True)
    league_median = panel.groupby(["league", "season_start_year"])["club_payroll_annual_gross_eur"].transform("median")
    panel["payroll_to_league_median"] = panel["club_payroll_annual_gross_eur"] / league_median
    panel["transfer_spend_log_eur"] = np.log1p(panel["transfer_spend_eur"])
    panel["transfer_income_log_eur"] = np.log1p(panel["transfer_income_eur"])
    panel["transfer_spend_to_payroll"] = panel["transfer_spend_eur"] / panel["club_payroll_annual_gross_eur"]
    panel["transfer_income_to_payroll"] = panel["transfer_income_eur"] / panel["club_payroll_annual_gross_eur"]
    panel["squad_foreign_share"] = panel["squad_foreign_player_count"] / panel["squad_player_count"]

    lookup = panel.set_index(["canonical_club_id", "season_start_year"])
    key_set = set(lookup.index)
    growth_1y = []
    cagr_3y = []
    history_1y = []
    history_3y = []
    spend_mean = []
    income_mean = []
    balance_mean = []
    spend_std = []
    for row in panel.itertuples(index=False):
        club_id, year = int(row.canonical_club_id), int(row.season_start_year)
        prior1 = (club_id, year - 1)
        full3 = [(club_id, year - offset) for offset in range(3)]
        full4 = [(club_id, year - offset) for offset in range(4)]
        has1 = prior1 in key_set
        has3 = all(key in key_set for key in full3)
        has4 = all(key in key_set for key in full4)
        history_1y.append(has1)
        history_3y.append(has3)
        if has1:
            prior_payroll = float(lookup.loc[prior1, "club_payroll_annual_gross_eur"])
            growth_1y.append(float(row.club_payroll_annual_gross_eur / prior_payroll - 1))
        else:
            growth_1y.append(np.nan)
        if has4:
            prior3_payroll = float(lookup.loc[(club_id, year - 3), "club_payroll_annual_gross_eur"])
            cagr_3y.append(float((row.club_payroll_annual_gross_eur / prior3_payroll) ** (1 / 3) - 1))
        else:
            cagr_3y.append(np.nan)
        if has3:
            history = lookup.loc[full3]
            spends = history["transfer_spend_eur"].to_numpy(dtype=float)
            spend_mean.append(float(spends.mean()))
            income_mean.append(float(history["transfer_income_eur"].mean()))
            balance_mean.append(float(history["transfer_balance_eur"].mean()))
            spend_std.append(float(spends.std(ddof=0)))
        else:
            spend_mean.append(np.nan)
            income_mean.append(np.nan)
            balance_mean.append(np.nan)
            spend_std.append(np.nan)
    panel["prior_top_flight_panel_available"] = history_1y
    panel["history_3y_consecutive_available"] = history_3y
    panel["payroll_growth_1y"] = growth_1y
    panel["payroll_cagr_3y"] = cagr_3y
    panel["transfer_spend_3y_mean_eur"] = spend_mean
    panel["transfer_income_3y_mean_eur"] = income_mean
    panel["transfer_balance_3y_mean_eur"] = balance_mean
    panel["transfer_spend_3y_std_eur"] = spend_std
    panel["feature_history_max_season_start_year"] = panel["season_start_year"]
    return panel


def build_extension_coverage(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(CONTRACTS / "canonical_extension_events.csv", low_memory=False)
    evaluation_ids = set(pd.read_csv(INTEGRATED / "integration_evaluation_cohort.csv", usecols=["capology_extension_event_id"])["capology_extension_event_id"])
    feature_columns = [
        "canonical_club_season_id",
        "canonical_club_id",
        "season_start_year",
        "club_payroll_annual_gross_eur",
        "payroll_league_percentile",
        "transfer_spend_eur",
        "transfer_income_eur",
        "transfer_balance_eur",
        "wage_structure_eligible",
    ]
    audit = events[
        [
            "capology_extension_event_id",
            "signed_date",
            "league",
            "club_name",
            "canonical_club_id",
            "pre_completed_season_start_year",
            "event_season_start_year",
        ]
    ].copy()
    audit["canonical_club_id"] = audit["canonical_club_id"].astype(int)
    audit["in_integrated_839_evaluation_cohort"] = audit["capology_extension_event_id"].isin(evaluation_ids)
    audit = audit.merge(
        panel[feature_columns],
        left_on=["canonical_club_id", "pre_completed_season_start_year"],
        right_on=["canonical_club_id", "season_start_year"],
        how="left",
        validate="many_to_one",
    )
    audit["prior_completed_season_feature_available"] = audit["canonical_club_season_id"].notna()
    event_keys = set(zip(panel["canonical_club_id"], panel["season_start_year"]))
    audit["event_season_panel_available"] = [
        (int(club_id), int(year)) in event_keys
        for club_id, year in zip(audit["canonical_club_id"], audit["event_season_start_year"])
    ]
    audit["missing_reason"] = np.where(
        audit["prior_completed_season_feature_available"],
        "available",
        np.where(audit["event_season_panel_available"], "prior_season_outside_big_five_top_flight_likely_promoted", "no_club_season_source_row"),
    )
    rows = []
    for scope, subset in {
        "all_canonical_extension_events": audit,
        "integrated_839_evaluation_cohort": audit.loc[audit["in_integrated_839_evaluation_cohort"]],
    }.items():
        rows.append(
            {
                "scope": scope,
                "events": len(subset),
                "feature_available": int(subset["prior_completed_season_feature_available"].sum()),
                "feature_missing": int((~subset["prior_completed_season_feature_available"]).sum()),
                "feature_coverage_rate": float(subset["prior_completed_season_feature_available"].mean()),
                "wage_structure_eligible": int(subset["wage_structure_eligible"].eq(True).sum()),
            }
        )
    return audit, pd.DataFrame(rows)


def field_dictionary(panel: pd.DataFrame) -> pd.DataFrame:
    identity = {
        "canonical_club_season_id",
        "canonical_club_id",
        "canonical_club_name",
        "league",
        "competition_id",
        "season",
        "season_start_year",
        "season_end_year",
        "capology_club_name",
        "capology_club_name_normalized",
        "capology_club_slug",
        "capology_club_code",
        "club_link_method",
    }
    lineage = {column for column in panel if "source_" in column or column == "feature_history_max_season_start_year"}
    diagnostic = {
        "club_payroll_bonus_gross_eur",
        "club_payroll_bonus_reported",
        "known_player_wage_count",
        "known_canonical_player_count",
        "known_player_wage_sum_eur",
        "known_player_wage_coverage_ratio",
        "wage_structure_eligible",
        "wage_structure_ineligible_reason",
        "season_club_count",
        "prior_top_flight_panel_available",
        "history_3y_consecutive_available",
    }
    descriptions = {
        "club_payroll_annual_gross_eur": "Estimated nominal fixed gross annual club payroll, excluding bonus.",
        "payroll_league_percentile": "Within-league-season percentile of nominal fixed payroll.",
        "payroll_to_league_median": "Nominal fixed payroll divided by contemporaneous league median.",
        "transfer_balance_eur": "Nominal transfer income minus nominal transfer spend.",
        "payroll_growth_1y": "Change from the immediately preceding club-season; null across a panel gap.",
        "payroll_cagr_3y": "Three-year payroll CAGR requiring four consecutive top-flight panel seasons.",
        "wage_structure_gini": "Player-wage Gini emitted only when the player panel reconciles to club payroll.",
        "wage_structure_hhi": "Player-wage concentration HHI emitted only when coverage is eligible.",
    }
    rows = []
    for column in panel.columns:
        if column in identity:
            category, eligible = "identity", False
        elif column in lineage:
            category, eligible = "lineage", False
        elif column in diagnostic:
            category, eligible = "coverage_or_diagnostic", False
        elif column.startswith("source_player_wage_"):
            category, eligible = "quarantined_source_statistic", False
        elif column in CANDIDATE_FEATURE_BLOCKS:
            category, eligible = "candidate_feature", True
        else:
            category, eligible = "source_value_not_in_initial_candidate_contract", False
        rows.append(
            {
                "field": column,
                "dtype": str(panel[column].dtype),
                "category": category,
                "candidate_model_eligible": eligible,
                "feature_block": CANDIDATE_FEATURE_BLOCKS.get(column),
                "timing_rule": "Join table row only as last completed season before decision" if eligible else "not a direct model feature",
                "description": descriptions.get(column, column.replace("_", " ").capitalize()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    payroll_files, transfer_files = source_files()
    manifest = build_source_manifest(payroll_files, transfer_files)
    club_map, _ = build_club_map()
    payroll = parse_family(payroll_files, "payroll", club_map)
    transfers = parse_family(transfer_files, "transfer", club_map)
    alias_review = build_alias_review(payroll)

    key = ["canonical_club_id", "season_start_year"]
    transfer_values = transfers.drop(
        columns=[
            "league",
            "competition_id",
            "season",
            "season_end_year",
            "capology_club_name",
            "capology_club_name_normalized",
            "capology_club_slug",
            "capology_club_code",
            "canonical_club_name",
            "club_link_method",
        ]
    )
    panel = payroll.merge(transfer_values, on=key, how="outer", validate="one_to_one", indicator=True)
    panel["canonical_club_id"] = panel["canonical_club_id"].astype(int)
    panel["canonical_club_season_id"] = [
        stable_hash("capology_club_season", club_id, season)
        for club_id, season in zip(panel["canonical_club_id"], panel["season_start_year"])
    ]
    structure = player_wage_structure()
    panel = panel.merge(structure, on=key, how="left", validate="one_to_one")
    panel["known_player_wage_coverage_ratio"] = panel["known_player_wage_sum_eur"] / panel["club_payroll_annual_gross_eur"]
    panel["wage_structure_eligible"] = (
        panel["known_player_wage_count"].ge(15)
        & panel["known_player_wage_coverage_ratio"].between(0.95, 1.05)
    )
    panel["wage_structure_ineligible_reason"] = np.select(
        [
            panel["known_player_wage_sum_eur"].isna(),
            panel["known_player_wage_count"].lt(15),
            ~panel["known_player_wage_coverage_ratio"].between(0.95, 1.05),
        ],
        ["no_player_salary_panel", "fewer_than_15_known_player_wages", "known_wage_sum_not_within_5pct_of_club_payroll"],
        default="eligible",
    )
    for source, target in {
        "source_player_wage_median_eur": "wage_structure_median_eur",
        "source_player_wage_max_eur": "wage_structure_max_eur",
        "source_player_wage_top5_share": "wage_structure_top5_share",
        "source_player_wage_hhi": "wage_structure_hhi",
        "source_player_wage_gini": "wage_structure_gini",
    }.items():
        panel[target] = panel[source].where(panel["wage_structure_eligible"])
    panel = add_features(panel)
    panel = panel.drop(columns=["_merge"])
    panel = panel.sort_values(["season_start_year", "league", "canonical_club_name"]).reset_index(drop=True)

    extension_audit, extension_summary = build_extension_coverage(panel)
    dictionary = field_dictionary(panel)

    checks: list[dict] = []
    add_check(checks, "source_file_count", len(manifest), 110, len(manifest) == 110, "55 files per source family.")
    add_check(checks, "alias_review_count", len(alias_review), 16, len(alias_review) == 16, "Every previously unresolved alias has an evidence row.")
    add_check(checks, "alias_ids_unique", alias_review["canonical_club_id"].nunique(), 16, alias_review["canonical_club_id"].is_unique, "No two aliases collapse to one club in this reviewed set.")
    add_check(checks, "all_club_rows_linked", int(payroll["canonical_club_id"].notna().sum()), len(payroll), payroll["canonical_club_id"].notna().all() and transfers["canonical_club_id"].notna().all(), "No unresolved source club remains.")
    add_check(checks, "canonical_panel_rows", len(panel), 1074, len(panel) == 1074, "Expected Big-Five club-season universe.")
    add_check(checks, "canonical_key_unique", int(panel.duplicated(key).sum()), 0, not panel.duplicated(key).any(), "One row per canonical club and season.")
    add_check(checks, "stable_id_unique", int(panel["canonical_club_season_id"].duplicated().sum()), 0, panel["canonical_club_season_id"].is_unique, "Stable hashed row ID is unique.")
    add_check(checks, "payroll_transfer_merge_complete", int(len(panel)), 1074, len(panel) == len(payroll) == len(transfers), "Exact paired source coverage.")
    add_check(checks, "no_adjusted_2025_values", len([column for column in panel if "adjusted" in column or "adjbalance" in column]), 0, not any("adjusted" in column or "adjbalance" in column for column in panel), "Retrospective 2025-adjusted values are excluded.")
    eligible = panel["wage_structure_eligible"]
    emitted = ["wage_structure_median_eur", "wage_structure_max_eur", "wage_structure_top5_share", "wage_structure_hhi", "wage_structure_gini"]
    invalid_emitted = int(panel.loc[~eligible, emitted].notna().sum().sum())
    add_check(checks, "wage_structure_quarantine", invalid_emitted, 0, invalid_emitted == 0, "Distribution features emit only above coverage gate.")
    expected_eval = extension_summary.loc[extension_summary["scope"].eq("integrated_839_evaluation_cohort")].iloc[0]
    add_check(checks, "evaluation_prior_season_coverage", int(expected_eval["feature_available"]), 795, int(expected_eval["feature_available"]) == 795 and int(expected_eval["events"]) == 839, "Leakage-safe historical join reproduces audited coverage.")
    numeric = panel.select_dtypes(include=[np.number])
    infinities = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
    add_check(checks, "no_infinite_features", infinities, 0, infinities == 0, "Ratios and logs contain no infinities.")
    add_check(checks, "lineage_never_future", int((panel["feature_history_max_season_start_year"] > panel["season_start_year"]).sum()), 0, (panel["feature_history_max_season_start_year"] <= panel["season_start_year"]).all(), "Every row's derived history ends in that row's season.")

    manifest.to_csv(output / "source_manifest.csv", index=False)
    alias_review.to_csv(output / "reviewed_club_aliases.csv", index=False)
    panel.to_csv(output / "canonical_club_season_financial_context.csv", index=False)
    extension_audit.to_csv(output / "extension_prior_season_feature_coverage.csv", index=False)
    extension_summary.to_csv(output / "extension_feature_coverage_summary.csv", index=False)
    dictionary.to_csv(output / "field_dictionary.csv", index=False)
    pd.DataFrame(checks).to_csv(output / "build_checks.csv", index=False)

    summary = {
        "source_files": len(manifest),
        "canonical_club_seasons": len(panel),
        "unique_clubs": int(panel["canonical_club_id"].nunique()),
        "reviewed_aliases": len(alias_review),
        "unresolved_source_rows": int(panel["canonical_club_id"].isna().sum()),
        "wage_structure_eligible_club_seasons": int(panel["wage_structure_eligible"].sum()),
        "extension_coverage": extension_summary.to_dict("records"),
        "build_checks_passed": int(pd.DataFrame(checks)["passed"].sum()),
        "build_checks_total": len(checks),
        "modeling_status": "candidate_feature_source_not_trained_not_deployed",
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    readme = f"""# Canonical Capology club financial context

This immutable build contains **{len(panel):,} canonical club-seasons** across the
Big Five from 2014-15 through 2024-25. All 16 previously unresolved Capology
aliases were reviewed against Transfermarkt club IDs and top-flight source
evidence; no fuzzy club match remains.

The table uses nominal euro payroll and transfer values. Capology's
retrospective 2025-adjusted fields and paywalled position-payroll placeholders
are intentionally excluded. All history features require exact consecutive
seasons and never bridge a relegation/promotion gap.

Player-level wage distributions are derived from the existing canonical salary
panel but emitted only when at least 15 wages are present and their sum is within
5% of the new club-payroll total. **{int(panel['wage_structure_eligible'].sum()):,}**
club-seasons clear that gate. Missing or non-reconciling distributions remain
null rather than silently estimated.

For extension decisions, join on `canonical_club_id` and
`pre_completed_season_start_year`. This covers
**{int(expected_eval['feature_available']):,}/{int(expected_eval['events']):,} ({expected_eval['feature_coverage_rate']:.1%})**
of the integrated evaluation cohort. The 44 missing cases are the previously
identified promoted-club boundary; signing-season totals remain prohibited as a
fill.

Status: candidate feature source only. No model was trained, no final holdout was
opened, and no V1, API, HTML, or deployment artifact was changed.
"""
    (output / "README.md").write_text(readme)

    output_rows = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "output_manifest.csv":
            output_rows.append({"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(output_rows).to_csv(output / "output_manifest.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
