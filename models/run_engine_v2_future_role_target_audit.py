"""Football-validity audit for the future-role / meaningful-contribution target.

This is a target audit, not a model-selection experiment.  It never evaluates
the 2024+ final cohort, refits a model, or writes a deployment artifact.  It tests
the exact 2020-2023 development labels against alternative thresholds,
denominators, follow-up definitions, movement states, injury availability, and
club-window coverage rules.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

import run_contribution_model_repair as injury  # noqa: E402

MASTER = ROOT / "Data" / "processed" / "contract_extension_integration" / "extension_modeling_master.csv"
GAMES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "games_clean.csv"
APPEARANCES = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
INJURY_SOURCE = ROOT / "Data" / "injury" / "full_dataset_thesis - 1.csv"
OUTPUT = ROOT / "Data" / "processed" / "engine_v2_future_role_target_audit"

AUDIT_YEARS = (2020, 2023)
PRIMARY_THRESHOLD = 0.25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def as_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) in {"bool", "boolean"}:
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_frame() -> pd.DataFrame:
    frame = pd.read_csv(MASTER, low_memory=False)
    frame["signed_date"] = pd.to_datetime(frame["signed_date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["signed_year"] = frame["signed_date"].dt.year.astype("Int64")
    frame = frame.loc[frame["signed_year"].between(*AUDIT_YEARS)].copy()

    numeric_columns = [
        "pre365_same_club_all_competition_opportunity_share",
        "post_y1_same_club_all_competition_opportunity_share",
        "post_y2_same_club_all_competition_opportunity_share",
        "post_y1_same_club_all_competition_games",
        "post_y2_same_club_all_competition_games",
        "post_y1_same_club_all_competition_minutes",
        "post_y2_same_club_all_competition_minutes",
        "post_y1_same_club_same_big5_competition_games",
        "post_y2_same_club_same_big5_competition_games",
        "post_y1_same_club_same_big5_competition_minutes",
        "post_y2_same_club_same_big5_competition_minutes",
        "post_y1_same_club_same_big5_competition_opportunity_share",
        "post_y2_same_club_same_big5_competition_opportunity_share",
        "first_any_outbound_days",
        "first_permanent_outbound_days",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    linked = frame["canonical_player_id"].notna()
    contract_covers = frame["expiration_date"].ge(frame["signed_date"] + pd.Timedelta(days=730))
    pre_ok = as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
    y1_ok = as_bool(frame["post_y1_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(frame["post_y2_window_evidence_eligible_10_club_games"])
    pre = frame["pre365_same_club_all_competition_opportunity_share"]
    y1 = frame["post_y1_same_club_all_competition_opportunity_share"]
    y2 = frame["post_y2_same_club_all_competition_opportunity_share"]

    # Exact currently deployed/frozen cohort logic.  Its missing Y1 check is
    # intentionally reproduced so the audit can measure the defect.
    frame["current_model_cohort"] = linked & contract_covers & pre_ok & y2_ok & pre.notna() & y2.notna()
    frame["strict_symmetric_model_cohort"] = frame["current_model_cohort"] & y1_ok & y1.notna()
    # Outcome validity is separate from predictor availability: pre-extension
    # evidence is needed by the model, not to observe the two-year outcome.
    frame["strict_outcome_cohort"] = linked & contract_covers & y1_ok & y2_ok & y1.notna() & y2.notna()
    frame["current_target"] = (y1.ge(PRIMARY_THRESHOLD) & y2.ge(PRIMARY_THRESHOLD)).astype(int)
    frame["weighted_730_share"] = (
        frame["post_y1_same_club_all_competition_minutes"]
        + frame["post_y2_same_club_all_competition_minutes"]
    ) / (
        90
        * (
            frame["post_y1_same_club_all_competition_games"]
            + frame["post_y2_same_club_all_competition_games"]
        )
    )
    return frame


def cohort_flow(frame: pd.DataFrame) -> pd.DataFrame:
    linked = frame["canonical_player_id"].notna()
    covers = frame["expiration_date"].ge(frame["signed_date"] + pd.Timedelta(days=730))
    pre_ok = as_bool(frame["pre365_window_evidence_eligible_10_club_games"])
    y1_ok = as_bool(frame["post_y1_window_evidence_eligible_10_club_games"])
    y2_ok = as_bool(frame["post_y2_window_evidence_eligible_10_club_games"])
    rows = [
        ("all_2020_2023_extension_events", pd.Series(True, index=frame.index), "audit population; final 2024+ cohort excluded"),
        ("linked_player", linked, "canonical player identity resolved"),
        ("contract_covers_730_days", linked & covers, "scheduled expiration no earlier than day 730"),
        ("current_model_cohort", frame["current_model_cohort"], "current asymmetric pre + Y2 evidence rule"),
        ("strict_symmetric_model_cohort", frame["strict_symmetric_model_cohort"], "current rule plus explicit Y1 evidence and nonmissing Y1 share"),
        ("strict_outcome_cohort", frame["strict_outcome_cohort"], "Y1/Y2 outcome observable; does not require pre-extension predictor coverage"),
        ("y1_evidence_eligible", y1_ok, "at least 10 observed extension-club games in Year 1"),
        ("y2_evidence_eligible", y2_ok, "at least 10 observed extension-club games in Year 2"),
        ("pre_evidence_eligible", pre_ok, "at least 10 observed extension-club games before signing"),
    ]
    return pd.DataFrame(
        {"stage": stage, "records": int(mask.sum()), "share_of_audit_population": float(mask.mean()), "meaning": note}
        for stage, mask, note in rows
    )


def threshold_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    y1 = data["post_y1_same_club_all_competition_opportunity_share"]
    y2 = data["post_y2_same_club_all_competition_opportunity_share"]
    baseline = y1.ge(PRIMARY_THRESHOLD) & y2.ge(PRIMARY_THRESHOLD)
    rows = []
    for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        target = y1.ge(threshold) & y2.ge(threshold)
        rows.append({
            "threshold": threshold,
            "records": len(data),
            "positive_outcomes": int(target.sum()),
            "positive_rate": float(target.mean()),
            "labels_changed_vs_25pct": int(target.ne(baseline).sum()),
            "label_change_rate_vs_25pct": float(target.ne(baseline).mean()),
        })
    return pd.DataFrame(rows)


def definition_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    y1 = data["post_y1_same_club_all_competition_opportunity_share"]
    y2 = data["post_y2_same_club_all_competition_opportunity_share"]
    baseline = y1.ge(0.25) & y2.ge(0.25)
    definitions = {
        "both_years_at_least_25pct": baseline,
        "year2_only_at_least_25pct": y2.ge(0.25),
        "unweighted_mean_of_years_at_least_25pct": ((y1 + y2) / 2).ge(0.25),
        "game_weighted_730_day_share_at_least_25pct": data["weighted_730_share"].ge(0.25),
        "either_year_at_least_25pct": y1.ge(0.25) | y2.ge(0.25),
        "both_years_at_least_20pct": y1.ge(0.20) & y2.ge(0.20),
        "both_years_at_least_30pct": y1.ge(0.30) & y2.ge(0.30),
    }
    interpretations = {
        "both_years_at_least_25pct": "sustained annual floor; current definition",
        "year2_only_at_least_25pct": "end-horizon role only",
        "unweighted_mean_of_years_at_least_25pct": "two-year average; gives both years equal weight",
        "game_weighted_730_day_share_at_least_25pct": "cumulative two-year realized contribution",
        "either_year_at_least_25pct": "meaningful involvement at some point",
        "both_years_at_least_20pct": "lower sustained floor",
        "both_years_at_least_30pct": "higher sustained floor",
    }
    return pd.DataFrame([
        {
            "definition": name,
            "interpretation": interpretations[name],
            "records": len(data),
            "positive_outcomes": int(target.sum()),
            "positive_rate": float(target.mean()),
            "labels_changed_vs_current": int(target.ne(baseline).sum()),
            "label_change_rate_vs_current": float(target.ne(baseline).mean()),
        }
        for name, target in definitions.items()
    ])


def denominator_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_parts: dict[str, pd.Series] = {}
    for window in ["post_y1", "post_y2"]:
        all_share = data[f"{window}_same_club_all_competition_opportunity_share"]
        league_share = data[f"{window}_same_club_same_big5_competition_opportunity_share"]
        league_games = data[f"{window}_same_club_same_big5_competition_games"]
        eligible = all_share.notna() & league_share.notna() & league_games.ge(10)
        label_parts[window] = league_share.ge(0.25)
        rows.append({
            "comparison": f"{window}_all_competitions_vs_original_big5_league",
            "records_compared": int(eligible.sum()),
            "median_absolute_share_difference": float((all_share[eligible] - league_share[eligible]).abs().median()),
            "p90_absolute_share_difference": float((all_share[eligible] - league_share[eligible]).abs().quantile(0.90)),
            "25pct_status_flips": int(all_share[eligible].ge(0.25).ne(league_share[eligible].ge(0.25)).sum()),
            "25pct_status_flip_rate": float(all_share[eligible].ge(0.25).ne(league_share[eligible].ge(0.25)).mean()),
        })
    both = (
        data["post_y1_same_club_same_big5_competition_games"].ge(10)
        & data["post_y2_same_club_same_big5_competition_games"].ge(10)
        & data["post_y1_same_club_same_big5_competition_opportunity_share"].notna()
        & data["post_y2_same_club_same_big5_competition_opportunity_share"].notna()
    )
    league_target = label_parts["post_y1"] & label_parts["post_y2"]
    current = data["current_target"].eq(1)
    rows.append({
        "comparison": "two_year_label_all_competitions_vs_original_big5_league",
        "records_compared": int(both.sum()),
        "median_absolute_share_difference": np.nan,
        "p90_absolute_share_difference": np.nan,
        "25pct_status_flips": int(current[both].ne(league_target[both]).sum()),
        "25pct_status_flip_rate": float(current[both].ne(league_target[both]).mean()),
    })
    return pd.DataFrame(rows)


def coverage_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g1 = data["post_y1_same_club_all_competition_games"]
    g2 = data["post_y2_same_club_all_competition_games"]
    current = data["current_target"].eq(1)
    rows = []
    for minimum in [10, 20, 30, 34, 38, 40]:
        eligible = g1.ge(minimum) & g2.ge(minimum)
        rows.append({
            "minimum_observed_club_games_each_year": minimum,
            "eligible_records": int(eligible.sum()),
            "excluded_from_strict_model_cohort": int((~eligible).sum()),
            "excluded_rate": float((~eligible).mean()),
            "positive_rate_among_eligible": float(current[eligible].mean()),
        })
    cases = data.loc[g1.lt(30) | g2.lt(30), [
        "capology_extension_event_id", "signed_date", "player_name", "club_name", "league", "competition_id",
        "post_y1_same_club_all_competition_games", "post_y2_same_club_all_competition_games",
        "post_y1_same_club_same_big5_competition_games", "post_y2_same_club_same_big5_competition_games",
        "post_y1_same_club_all_competition_opportunity_share", "post_y2_same_club_all_competition_opportunity_share",
        "first_any_outbound_type", "first_any_outbound_days", "current_target",
    ]].copy()
    cases["coverage_warning"] = "fewer_than_30_observed_extension_club_games_in_at_least_one_365_day_window"
    cases["interpretation"] = "consistent with relegation/top-flight exit, partial competition capture, or another club-window coverage break; requires resolution, not automatic relabeling"
    return pd.DataFrame(rows), cases.sort_values(["club_name", "signed_date", "player_name"])


def movement_decomposition(data: pd.DataFrame) -> pd.DataFrame:
    days = data["first_any_outbound_days"]
    event_type = data["first_any_outbound_type"].fillna("").astype(str).str.lower()
    within = days.le(730)
    state = pd.Series("no_outbound_within_24m", index=data.index, dtype="object")
    state.loc[within & event_type.str.contains("loan")] = "temporary_first_within_24m"
    state.loc[within & event_type.str.contains("transfer")] = "permanent_first_within_24m"
    state.loc[within & ~event_type.str.contains("loan|transfer")] = "other_or_unresolved_first_within_24m"
    negative = data["current_target"].eq(0)
    rows = []
    for name, group_index in state.groupby(state).groups.items():
        mask = data.index.isin(group_index)
        rows.append({
            "movement_state": name,
            "records": int(mask.sum()),
            "share_of_cohort": float(mask.mean()),
            "not_sustained_records": int((mask & negative).sum()),
            "not_sustained_rate_within_state": float(negative[mask].mean()),
            "share_of_all_not_sustained_records": float((mask & negative).sum() / negative.sum()),
        })
    return pd.DataFrame(rows).sort_values("movement_state")


def subgroup_prevalence(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    specifications = {
        "league": data["league"].astype(str),
        "position": data["canonical_position"].astype(str),
        "signing_month": data["signed_date"].dt.month.astype(str),
    }
    warning = data["post_y1_same_club_all_competition_games"].lt(30) | data["post_y2_same_club_all_competition_games"].lt(30)
    for dimension, values in specifications.items():
        work = pd.DataFrame({"value": values, "target": data["current_target"], "coverage_warning": warning})
        summary = work.groupby("value", dropna=False).agg(
            records=("target", "size"),
            positive_rate=("target", "mean"),
            coverage_warning_records=("coverage_warning", "sum"),
            coverage_warning_rate=("coverage_warning", "mean"),
        ).reset_index()
        summary.insert(0, "dimension", dimension)
        frames.append(summary)
    return pd.concat(frames, ignore_index=True)


def build_game_dates() -> dict[int, np.ndarray]:
    games = pd.read_csv(
        GAMES,
        usecols=["date", "home_club_id", "away_club_id", "is_national_team_game"],
        low_memory=False,
    )
    games["date"] = pd.to_datetime(games["date"], errors="coerce")
    games = games.loc[games["date"].notna() & games["is_national_team_game"].ne(True)]
    by_club: dict[int, list[pd.Timestamp]] = {}
    for row in games.itertuples(index=False):
        for club_id in (row.home_club_id, row.away_club_id):
            if pd.notna(club_id):
                by_club.setdefault(int(club_id), []).append(row.date)
    return {club_id: np.sort(pd.Series(dates).unique()) for club_id, dates in by_club.items()}


def injury_audit(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_player, by_pair = injury.build_appearances_by_player_and_pair()
    spells, _, source_notes = injury.build_injury_linkage_corrected(by_pair)
    lookup = injury.build_spell_lookup(spells)
    source_max = spells["until_date"].max()
    game_dates = build_game_dates()

    clean_spells = spells.dropna(subset=["canonical_player_id", "canonical_club_id", "from_date", "until_date"]).copy()
    clean_spells["canonical_player_id"] = clean_spells["canonical_player_id"].astype(int)
    clean_spells["canonical_club_id"] = clean_spells["canonical_club_id"].astype(int)
    spells_by_pair = {
        (player_id, club_id): group[["from_date", "until_date"]].to_numpy()
        for (player_id, club_id), group in clean_spells.groupby(["canonical_player_id", "canonical_club_id"])
    }

    records = []
    for row in data.itertuples(index=False):
        pre_features = injury.corrected_injury_features_for(
            row.canonical_player_id, row.signed_date, lookup, by_player
        )
        eligible = (
            pre_features["injury_coverage_indicator_365"] == 1.0
            and row.signed_date + pd.Timedelta(days=730) <= source_max
            and injury.player_active_in_window(
                row.canonical_player_id,
                row.signed_date,
                row.signed_date + pd.Timedelta(days=730),
                by_player,
            )
        )
        if not eligible:
            continue

        player_id = int(row.canonical_player_id)
        club_id = int(row.canonical_club_id_x)
        player_spells = spells_by_pair.get((player_id, club_id), np.empty((0, 2), dtype="datetime64[ns]"))
        club_dates = game_dates.get(club_id, np.array([], dtype="datetime64[ns]"))
        values: dict[str, float] = {}
        total_injury_days = 0
        for label, start, end, minutes, observed_games in [
            ("y1", row.signed_date, row.signed_date + pd.Timedelta(days=365), row.post_y1_same_club_all_competition_minutes, row.post_y1_same_club_all_competition_games),
            ("y2", row.signed_date + pd.Timedelta(days=365), row.signed_date + pd.Timedelta(days=730), row.post_y2_same_club_all_competition_minutes, row.post_y2_same_club_all_competition_games),
        ]:
            # Calendar burden uses every validated spell for the player; the
            # available-match denominator only removes spells attributed to
            # the extension club so an injury at a loan destination cannot
            # erase a parent-club opportunity.
            injury_days = 0
            for spell in lookup.get(player_id, []):
                if pd.isna(spell["from"]) or pd.isna(spell["until"]):
                    continue
                overlap = min(spell["until"], end) - max(spell["from"], start)
                injury_days += max(0, overlap.days)
            total_injury_days += injury_days

            dates = club_dates[(club_dates > np.datetime64(start)) & (club_dates <= np.datetime64(end))]
            missed = np.zeros(len(dates), dtype=bool)
            for from_date, until_date in player_spells:
                missed |= (dates >= np.datetime64(from_date)) & (dates <= np.datetime64(until_date))
            injured_games = int(missed.sum())
            available_games = max(int(observed_games) - injured_games, 0)
            adjusted_share = min(float(minutes) / (90 * available_games), 1.05) if available_games > 0 else np.nan
            values[f"{label}_injury_days"] = injury_days
            values[f"{label}_injured_club_games"] = injured_games
            values[f"{label}_available_games"] = available_games
            values[f"{label}_availability_adjusted_share"] = adjusted_share

        adjusted_target = (
            values["y1_availability_adjusted_share"] >= 0.25
            and values["y2_availability_adjusted_share"] >= 0.25
        ) if pd.notna(values["y1_availability_adjusted_share"]) and pd.notna(values["y2_availability_adjusted_share"]) else np.nan
        records.append({
            "capology_extension_event_id": row.capology_extension_event_id,
            "current_target": int(row.current_target),
            "post_signing_injury_days_total": total_injury_days,
            **values,
            "availability_adjusted_target": adjusted_target,
        })

    detail = pd.DataFrame(records)
    detail["injury_burden_band"] = pd.cut(
        detail["post_signing_injury_days_total"],
        [-1, 0, 30, 90, 180, np.inf],
        labels=["0_days", "1_30_days", "31_90_days", "91_180_days", "181plus_days"],
    )
    decomposition = detail.groupby("injury_burden_band", observed=True).agg(
        records=("current_target", "size"),
        sustained_rate=("current_target", "mean"),
        mean_injury_days=("post_signing_injury_days_total", "mean"),
    ).reset_index()
    decomposition["not_sustained_rate"] = 1 - decomposition["sustained_rate"]
    total_negative = detail["current_target"].eq(0).sum()
    negative_by_band = detail.loc[detail["current_target"].eq(0), "injury_burden_band"].value_counts()
    decomposition["share_of_all_not_sustained_records"] = decomposition["injury_burden_band"].map(
        lambda band: float(negative_by_band.get(band, 0) / total_negative)
    )

    valid = detail["availability_adjusted_target"].notna()
    adjusted = detail.loc[valid, "availability_adjusted_target"].astype(bool)
    current = detail.loc[valid, "current_target"].astype(bool)
    adjustment = pd.DataFrame([{
        "injury_eligible_records": len(detail),
        "valid_adjusted_denominator_records": int(valid.sum()),
        "current_positive_rate": float(current.mean()),
        "availability_adjusted_positive_rate": float(adjusted.mean()),
        "labels_changed": int(current.ne(adjusted).sum()),
        "label_change_rate": float(current.ne(adjusted).mean()),
        "not_sustained_to_sustained": int((~current & adjusted).sum()),
        "sustained_to_not_sustained": int((current & ~adjusted).sum()),
        "records_with_at_least_one_injured_extension_club_game": int(
            (detail["y1_injured_club_games"] + detail["y2_injured_club_games"]).gt(0).sum()
        ),
        "scope_warning": "validated partial injury source only; suspensions, international duty, tactical non-selection and all other unavailability are not captured",
    }])
    return detail, decomposition, adjustment.merge(source_notes, how="cross")


def build_findings(
    frame: pd.DataFrame,
    data: pd.DataFrame,
    thresholds: pd.DataFrame,
    definitions: pd.DataFrame,
    denominators: pd.DataFrame,
    coverage: pd.DataFrame,
    movement: pd.DataFrame,
    injury_adjustment: pd.DataFrame,
) -> pd.DataFrame:
    asymmetry = int((frame["current_model_cohort"] & ~frame["strict_symmetric_model_cohort"]).sum())
    threshold_20 = int(thresholds.loc[thresholds["threshold"].eq(0.20), "labels_changed_vs_25pct"].iloc[0])
    threshold_30 = int(thresholds.loc[thresholds["threshold"].eq(0.30), "labels_changed_vs_25pct"].iloc[0])
    weighted_change = int(definitions.loc[
        definitions["definition"].eq("game_weighted_730_day_share_at_least_25pct"), "labels_changed_vs_current"
    ].iloc[0])
    domestic_flips = int(denominators.loc[
        denominators["comparison"].eq("two_year_label_all_competitions_vs_original_big5_league"), "25pct_status_flips"
    ].iloc[0])
    under30 = int(coverage.loc[coverage["minimum_observed_club_games_each_year"].eq(30), "excluded_from_strict_model_cohort"].iloc[0])
    temporary = movement.loc[movement["movement_state"].eq("temporary_first_within_24m")].iloc[0]
    all_outbound_negative_share = float(movement.loc[
        ~movement["movement_state"].eq("no_outbound_within_24m"), "share_of_all_not_sustained_records"
    ].sum())
    injury_row = injury_adjustment.iloc[0]
    over_one = int(
        data["post_y1_same_club_all_competition_opportunity_share"].gt(1).sum()
        + data["post_y2_same_club_all_competition_opportunity_share"].gt(1).sum()
    )
    rows = [
        ("FR-01", "Year-1/Year-2 evidence symmetry", "HIGH", "FAIL", f"{asymmetry} current 2020-2023 model rows pass the Y2 gate without passing the matching Y1 gate. All receive labels despite inadequate evidence; the 3 missing Y1 shares become negative by comparison semantics.", "Require explicit Y1 and Y2 evidence plus nonmissing shares before assigning any target."),
        ("FR-02", "25% threshold", "MEDIUM", "CONDITIONALLY_DEFENSIBLE", f"The threshold is a declared fringe/rotation boundary, not an externally validated football breakpoint. Moving it to 20% or 30% changes {threshold_20} or {threshold_30} of {len(data)} labels, respectively.", "Keep only as an explicit policy threshold; publish sensitivity and do not call 25% an empirical law."),
        ("FR-03", "Opportunity denominator", "MEDIUM", "CONDITIONALLY_DEFENSIBLE", f"All scheduled extension-club matches is coherent for realized club contribution. Restricting to the original Big-Five league changes {domestic_flips} adequately covered two-year labels.", "Retain all-competition capacity for realized contribution, but name it club-match capacity and repair lower-tier coverage."),
        ("FR-04", "Injury handling", "HIGH", "FAIL_AS_PURE_ROLE", f"The current denominator treats injury absence as zero contribution. On {int(injury_row['valid_adjusted_denominator_records'])} injury-auditable rows, removing same-club injury matches changes {int(injury_row['labels_changed'])} labels ({injury_row['label_change_rate']:.1%}).", "Distinguish realized contribution from role when available; do not market the current target as pure selection/role."),
        ("FR-05", "Loans and outbound movement", "HIGH", "FAIL_AS_PURE_ROLE", f"Temporary-first moves account for {int(temporary['not_sustained_records'])} negative labels ({temporary['share_of_all_not_sustained_records']:.1%} of all negatives); all outbound-first states account for {all_outbound_negative_share:.1%}.", "Use Meaningful Retention for the joint business outcome, Temporary Displacement separately, and conditional-on-stay involvement for a pure role question."),
        ("FR-06", "Relegation / club-season coverage", "CRITICAL", "FAIL", f"The 10-game gate admits {under30} of {len(data)} rows with fewer than 30 observed extension-club matches in at least one nominal 365-day year, a pattern consistent with relegation/top-flight exit or another coverage break.", "Do not label these rows until lower-tier schedules are integrated or a stricter completeness/refusal rule is frozen."),
        ("FR-07", "Year-1/Year-2 construction", "MEDIUM", "CONDITIONALLY_DEFENSIBLE", f"The windows are exact 0-365 and 366-730 day contract anniversaries, not football seasons. A game-weighted two-year definition changes {weighted_change} of {len(data)} labels because it answers cumulative contribution rather than a sustained annual floor.", "Keep the two annual floors only for a sustained outcome and describe the windows exactly; do not paraphrase as total two-year minutes."),
        ("FR-08", "Share upper bound", "LOW", "FAIL_IMPLEMENTATION", f"{over_one} Year-1/Year-2 shares exceed 100% because 90 minutes per match is used while extra time is allowed and values are capped at 1.05.", "Cap normalized shares at 1.00 or use exact match-minute capacity before future continuous-role modeling."),
        ("FR-09", "Overall target verdict", "CRITICAL", "NOT_DEPLOYABLE_AS_FUTURE_ROLE", "The current label mixes realized use, availability, loan/permanent movement, and incomplete post-relegation schedules. Its predictive performance cannot make that composite a pure role measure.", "Retain research lineage only. Rebuild a strict realized same-club contribution target; use Meaningful Retention and Temporary Displacement as separately named product questions."),
    ]
    return pd.DataFrame(rows, columns=["finding_id", "component", "severity", "verdict", "evidence", "required_action"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = [MASTER, GAMES, APPEARANCES, INJURY_SOURCE]
    source_before = {path: sha256(path) for path in sources}

    frame = load_frame()
    data = frame.loc[frame["strict_symmetric_model_cohort"]].copy()
    flow = cohort_flow(frame)
    asymmetry = frame.loc[
        frame["current_model_cohort"] & ~frame["strict_symmetric_model_cohort"],
        [
            "capology_extension_event_id", "signed_date", "player_name", "club_name", "league",
            "post_y1_same_club_all_competition_games", "post_y1_same_club_all_competition_opportunity_share",
            "post_y2_same_club_all_competition_games", "post_y2_same_club_all_competition_opportunity_share",
            "current_target",
        ],
    ].copy()
    asymmetry["defect"] = "current cohort does not require Year-1 evidence before assigning a both-years target"
    thresholds = threshold_sensitivity(data)
    definitions = definition_sensitivity(data)
    denominators = denominator_sensitivity(data)
    coverage, coverage_cases = coverage_sensitivity(data)
    movement = movement_decomposition(data)
    subgroups = subgroup_prevalence(data)
    injury_detail, injury_decomposition, injury_adjustment = injury_audit(data)
    findings = build_findings(frame, data, thresholds, definitions, denominators, coverage, movement, injury_adjustment)

    checks = []
    def check(name: str, actual: Any, expected: Any, passed: bool, note: str) -> None:
        checks.append({"check": name, "actual": actual, "expected": expected, "passed": bool(passed), "note": note})

    check("final_holdout_excluded", int(frame["signed_year"].max()), 2023, int(frame["signed_year"].max()) == 2023, "No 2024+ label is read by the audit population.")
    check("strict_model_cohort_reference", len(data), 922, len(data) == 922, "Symmetric 2020-2023 model cohort after removing the Y1 evidence defect.")
    check("current_cohort_reference", int(frame["current_model_cohort"].sum()), 929, int(frame["current_model_cohort"].sum()) == 929, "Reproduces frozen rolling-origin evaluation rows.")
    check("year1_asymmetry_exposed", len(asymmetry), 7, len(asymmetry) == 7, "Seven evaluated rows lack the symmetric Year-1 evidence gate.")
    check("primary_label_reference", int(data["current_target"].sum()), 453, int(data["current_target"].sum()) == 453, "Strict 25% target recomputed from raw shares.")
    check("threshold_monotonic", int(thresholds["positive_rate"].is_monotonic_decreasing), 1, thresholds["positive_rate"].is_monotonic_decreasing, "Higher thresholds cannot increase positive outcomes.")
    check("movement_partition_complete", int(movement["records"].sum()), len(data), int(movement["records"].sum()) == len(data), "Every strict row belongs to one first-movement state.")
    check("coverage_10_matches_strict", int(coverage.iloc[0]["eligible_records"]), len(data), int(coverage.iloc[0]["eligible_records"]) == len(data), "Strict cohort still reproduces the current 10-game rule before sensitivity testing.")
    check("injury_subset_nontrivial", int(len(injury_detail)), ">=500", len(injury_detail) >= 500, "Partial validated injury source supports a material diagnostic subset.")
    check("injury_adjustment_one_direction", int(injury_adjustment.iloc[0]["sustained_to_not_sustained"]), 0, int(injury_adjustment.iloc[0]["sustained_to_not_sustained"]) == 0, "Removing unavailable matches cannot reduce an opportunity share.")
    check("critical_verdict_present", int((findings["verdict"] == "NOT_DEPLOYABLE_AS_FUTURE_ROLE").sum()), 1, int((findings["verdict"] == "NOT_DEPLOYABLE_AS_FUTURE_ROLE").sum()) == 1, "Audit ends in one explicit product decision.")
    source_after = {path: sha256(path) for path in sources}
    check("sources_byte_identical", int(source_before == source_after), 1, source_before == source_after, "All four large input sources are read-only and unchanged.")
    checks_frame = pd.DataFrame(checks)

    write_csv(flow, OUTPUT / "cohort_flow.csv")
    write_csv(asymmetry, OUTPUT / "year1_evidence_asymmetry.csv")
    write_csv(thresholds, OUTPUT / "threshold_sensitivity.csv")
    write_csv(definitions, OUTPUT / "definition_sensitivity.csv")
    write_csv(denominators, OUTPUT / "denominator_sensitivity.csv")
    write_csv(coverage, OUTPUT / "club_window_coverage_sensitivity.csv")
    write_csv(coverage_cases, OUTPUT / "club_window_coverage_break_cases.csv")
    write_csv(movement, OUTPUT / "movement_decomposition.csv")
    write_csv(subgroups, OUTPUT / "subgroup_prevalence.csv")
    write_csv(injury_detail, OUTPUT / "injury_case_audit.csv")
    write_csv(injury_decomposition, OUTPUT / "injury_burden_decomposition.csv")
    write_csv(injury_adjustment, OUTPUT / "injury_available_denominator_summary.csv")
    write_csv(findings, OUTPUT / "audit_findings.csv")
    write_csv(checks_frame, OUTPUT / "build_checks.csv")
    write_csv(pd.DataFrame([
        {"source_file": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": source_after[path]}
        for path in sources
    ]), OUTPUT / "source_manifest.csv")

    summary = {
        "audit_population_years": [2020, 2023],
        "final_holdout_evaluated": False,
        "current_model_cohort_rows": int(frame["current_model_cohort"].sum()),
        "strict_symmetric_model_cohort_rows": len(data),
        "year1_evidence_defect_rows": len(asymmetry),
        "current_25pct_positive_rows": int(data["current_target"].sum()),
        "current_25pct_positive_rate": float(data["current_target"].mean()),
        "under_30_game_coverage_break_rows": int(len(coverage_cases)),
        "game_weighted_definition_changed_rows": int(definitions.loc[
            definitions["definition"].eq("game_weighted_730_day_share_at_least_25pct"), "labels_changed_vs_current"
        ].iloc[0]),
        "temporary_move_share_of_negative_labels": float(movement.loc[
            movement["movement_state"].eq("temporary_first_within_24m"), "share_of_all_not_sustained_records"
        ].iloc[0]),
        "all_outbound_share_of_negative_labels": float(movement.loc[
            ~movement["movement_state"].eq("no_outbound_within_24m"), "share_of_all_not_sustained_records"
        ].sum()),
        "injury_auditable_rows": int(len(injury_detail)),
        "injury_adjusted_label_changes": int(injury_adjustment.iloc[0]["labels_changed"]),
        "overall_verdict": "not_deployable_as_future_role; rebuild as precisely named realized same-club contribution and keep movement outcomes separate",
        "checks_passed": int(checks_frame["passed"].sum()),
        "checks_total": len(checks_frame),
        "all_checks_passed": bool(checks_frame["passed"].all()),
    }
    (OUTPUT / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    readme = f"""# Engine V2 future-role / meaningful-contribution target audit

This audit calculates and emits results only for 2020-2023 development
outcomes. It does not evaluate the 2024+ final cohort, fit a model, or change
any deployment artifact.

## Verdict

The existing `target_sustained_meaningful_contribution` is **not defensible as
a pure future-role outcome**. It is a composite of realized same-club use,
availability, loans/permanent departures, and—on {len(coverage_cases)} rows—an
annual club-match denominator with fewer than 30 observed matches in at least
one year. It can remain lineage evidence, but it must not be promoted under the
name “future role.”

A repaired **realized same-club contribution** outcome is defensible after:

1. requiring symmetric Year-1 and Year-2 evidence;
2. resolving lower-tier/relegated-club schedules or refusing incomplete years;
3. making the 25% floor an explicit policy threshold rather than an empirical
   law; and
4. capping shares at 100% (or using exact match-minute capacity).

Meaningful Retention should remain the joint primary extension outcome,
Temporary Displacement should remain separate, and a pure “role when
available” outcome should remain unavailable until availability coverage is
complete enough to support it.

## What the numbers mean

- Current frozen evaluation logic selects {int(frame['current_model_cohort'].sum()):,}
  rows; the symmetric correction leaves {len(data):,}. The {len(asymmetry)}
  removed rows were labeled without the required Year-1 evidence.
- At 25%, {int(data['current_target'].sum()):,} of {len(data):,} strict rows
  meet the floor in both contract-anniversary years.
- A cumulative, game-weighted two-year definition changes
  {summary['game_weighted_definition_changed_rows']} labels. That is not a
  better version of the same question; it is a different question.
- Temporary-first movement explains
  {summary['temporary_move_share_of_negative_labels']:.1%} of all negative
  labels, and all outbound-first states explain
  {summary['all_outbound_share_of_negative_labels']:.1%}. The current label is
  therefore not a clean selection-role measure.
- On {summary['injury_auditable_rows']} rows with independently constrained
  injury follow-up, an availability-adjusted club-match denominator changes
  {summary['injury_adjusted_label_changes']} labels. This is a sensitivity
  test, not a replacement target: suspensions and other unavailability remain
  missing.

See `audit_findings.csv` for the decision ledger and the other CSVs for each
mechanical decomposition.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")

    output_files = sorted(
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name not in {"output_manifest.csv", "independent_verification.csv"}
    )
    write_csv(pd.DataFrame([
        {"output_file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_files
    ]), OUTPUT / "output_manifest.csv")

    if not checks_frame["passed"].all():
        failed = checks_frame.loc[~checks_frame["passed"], "check"].tolist()
        raise RuntimeError(f"Future-role target audit failed checks: {failed}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
