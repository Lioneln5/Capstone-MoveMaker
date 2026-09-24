"""Live-request integrity audit.

Every other verification script in this project (scripts/verify_deployment_
models.py, scripts/verify_deployment_out_of_sample.py, and the retriever's
own _run_parity_check) tests HISTORICAL REPLAY exclusively -- every
decision_date used is a real Capology signed_date, always safely inside
every source table's own coverage window. That is a materially different
question from "does a genuinely live, present-day request behave
correctly," and nothing tested that until it broke in front of the user.

The incident (2026-08-12 session, Jude Bellingham): canonical_feature_
retriever.py's prior-wage and club-salary-panel lookups required an EXACT
season match computed from decision_date. The salary panel's own coverage
stops at the 2023-24 season; a live decision_date of "today" computes a
target season well past that. The lookup silently returned "unavailable"
for every player on every live request, not because the data didn't exist,
but because nobody had told it to fall back. It was found only because the
user personally knew Bellingham's salary was in the source data and pushed
back on a number that didn't look right -- not because any automated check
caught it. A second, structurally identical bug (the pre-365-day sporting
window silently running past the games/appearances data's own cutoff, with
no warning) was found by auditing every other date-relative lookup for the
same pattern once the first one was understood, before this script existed.

This script is the permanent guard against that whole class of bug
recurring silently: it specifically constructs LIVE, present-day requests
(decision_date = today) against a real sample of currently-rostered
players, and asserts that known-stale data sources produce an explicit
warning and a graceful fallback value -- never a silent "unavailable" for
data that actually exists at an earlier date, and never a silently wrong
number with no warning at all.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from canonical_feature_retriever import (  # noqa: E402
    CanonicalFeatureRetriever,
    current_season_start_year,
)
from incumbent_extension_profile import IncumbentExtensionProfileBuilder  # noqa: E402
from business_metric_engine import (  # noqa: E402
    PRIMARY_VALUE_HORIZON_WARNING,
    PROVISIONAL_24M_VALUE_WARNING,
)

OUTPUT = ROOT / "Data" / "processed" / "deployment_models"
SALARY_PANEL_PATH = ROOT / "Data" / "processed" / "capology_contracts" / "canonical_salary_panel.csv"
LIVE_SALARY_PATH = ROOT / "Data" / "processed" / "live_input_refresh" / "canonical_salary_overlay.csv"
APPEARANCE_PATH = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "appearances_clean.csv"
GAMES_PATH = ROOT / "Data" / "processed" / "transfermarkt_clean" / "tables" / "games_clean.csv"

TODAY = pd.Timestamp(date.today())


def add(checks: list[dict], name: str, observed: object, expected: object, passed: bool, notes: str) -> None:
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed), "notes": notes})


def load_salary_panel_sample(n: int, seed: int) -> pd.DataFrame:
    """Real players known to have at least one NON-NULL salary-panel wage --
    exactly the population that would have been silently broken by the
    original exact-season-match bug on a live request. Filtering out
    null-wage rows matters: Capology sometimes lists a player on a club's
    roster page without disclosing a figure (a real, correctly-preserved
    missing value, not a bug) -- sampling those would test the wrong thing,
    since "unavailable" is the CORRECT answer for a player with no reported
    wage at all, at any season, for that club."""
    panel = pd.read_csv(SALARY_PANEL_PATH, low_memory=False)
    panel["canonical_club_id"] = pd.to_numeric(panel["canonical_club_id"], errors="coerce")
    panel = panel.dropna(subset=["canonical_club_id", "player_name_normalized", "annual_gross_eur"])
    # One row per player: their most recent known season/club with a
    # reported wage, the most realistic "who currently plays where, and we
    # know what they earned at some point" proxy available from this source.
    latest = panel.sort_values("source_page_season_start_year").groupby("player_name_normalized").tail(1)
    return latest.sample(n=min(n, len(latest)), random_state=seed)


def load_canonical_player_lookup() -> pd.DataFrame:
    players = pd.read_csv(
        ROOT / "Data" / "processed" / "canonical_integration" / "canonical_player_dimension.csv",
        usecols=["canonical_player_id", "canonical_name_normalized", "canonical_date_of_birth"], low_memory=False,
    )
    players["canonical_player_id"] = pd.to_numeric(players["canonical_player_id"], errors="coerce")
    return players.dropna(subset=["canonical_player_id"])


def data_source_cutoffs() -> dict[str, Any]:
    panel = pd.read_csv(SALARY_PANEL_PATH, usecols=["source_page_season_start_year"], low_memory=False)
    salary_seasons = [pd.to_numeric(panel["source_page_season_start_year"], errors="coerce")]
    if LIVE_SALARY_PATH.exists():
        live = pd.read_csv(LIVE_SALARY_PATH, usecols=["source_page_season_start_year"], low_memory=False)
        salary_seasons.append(pd.to_numeric(live["source_page_season_start_year"], errors="coerce"))
    apps = pd.read_csv(APPEARANCE_PATH, usecols=["date"], low_memory=False)
    games = pd.read_csv(GAMES_PATH, usecols=["date"], low_memory=False)
    return {
        "salary_panel_max_season_start_year": int(pd.concat(salary_seasons, ignore_index=True).max()),
        "sporting_data_max_date": max(
            pd.to_datetime(apps["date"], errors="coerce").max(),
            pd.to_datetime(games["date"], errors="coerce").max(),
        ).date().isoformat(),
    }


def main() -> None:
    checks: list[dict] = []
    cutoffs = data_source_cutoffs()
    print(f"Data source cutoffs at this run: {cutoffs}")
    print(f"Today: {TODAY.date().isoformat()}")

    add(checks, "salary_panel_cutoff_detected", cutoffs["salary_panel_max_season_start_year"], ">=2023",
        cutoffs["salary_panel_max_season_start_year"] >= 2023,
        "Cutoff includes the frozen canonical panel and any append-only live salary overlay. Warning expectations below "
        "are evaluated per player and season, so refreshing the overlay cannot create a false audit failure.")

    players_lookup = load_canonical_player_lookup()
    sample = load_salary_panel_sample(n=60, seed=7)
    retriever = CanonicalFeatureRetriever()

    live_results, historical_results, live_expected_staleness, exceptions = [], [], [], []
    for row in sample.itertuples(index=False):
        player_rows = players_lookup.loc[players_lookup["canonical_name_normalized"].eq(row.player_name_normalized)]
        if player_rows.empty:
            continue
        player_id = int(player_rows.iloc[0]["canonical_player_id"])
        club_id = int(row.canonical_club_id)
        panel_season = int(row.source_page_season_start_year)
        # Historical request: decision_date inside the season THIS row's own
        # panel data covers -- must NOT trigger a staleness warning.
        # Must land in the season whose "prior season" IS panel_season, i.e.
        # decision_season == panel_season + 1. current_season_start_year()
        # only returns panel_season+1 for a month >= 7 (Aug-Dec) date in that
        # calendar year -- a March date resolves to the PRECEDING season
        # instead (off-by-one that briefly produced a false-positive
        # staleness warning here; the retriever was right, this date wasn't).
        historical_date = pd.Timestamp(year=panel_season + 1, month=9, day=1)
        # Live request: decision_date = today -- SHOULD trigger a staleness
        # warning if (and only if) the panel doesn't cover the live season.
        try:
            live = retriever.get_extension_features(
                canonical_player_id=player_id, canonical_club_id=club_id, competition_id="GB1",
                decision_date=TODAY, proposed_annual_fixed_wage_eur=1_000_000.0, proposed_contract_years=2.0,
                current_public_market_value_eur=1_000_000.0, player_name_normalized=row.player_name_normalized,
            )
            hist = retriever.get_extension_features(
                canonical_player_id=player_id, canonical_club_id=club_id, competition_id="GB1",
                decision_date=historical_date, proposed_annual_fixed_wage_eur=1_000_000.0, proposed_contract_years=2.0,
                current_public_market_value_eur=1_000_000.0, player_name_normalized=row.player_name_normalized,
            )
        except Exception as exc:  # noqa: BLE001
            exceptions.append(f"{row.player_name_normalized}: {exc!r}")
            continue
        if not live.in_scope or not hist.in_scope:
            continue  # incumbency/league refusal on this synthetic pairing; not what this script tests
        live_results.append(live)
        historical_results.append(hist)
        target_prior_season = current_season_start_year(TODAY) - 1
        _, actual_wage_season = retriever._prior_wage(  # noqa: SLF001 -- verifier intentionally checks source selection
            row.player_name_normalized, club_id, target_prior_season
        )
        live_expected_staleness.append(
            actual_wage_season is not None and actual_wage_season != target_prior_season
        )

    add(checks, "sample_scored_without_exception", len(exceptions), 0, not exceptions,
        f"{len(live_results)}/{len(sample)} usable pairs. Failures: {exceptions[:5] or 'none'}")

    # --- The core regression guard: live requests must not silently lose
    # data the panel actually has, and must warn when using an older season.
    silently_missing = sum(1 for r in live_results if "log_prior_season_annual_gross_eur" in r.unavailable)
    add(checks, "live_request_no_silent_prior_wage_loss", silently_missing, 0, silently_missing == 0,
        f"{silently_missing}/{len(live_results)} sampled players -- each KNOWN to have a salary-panel row -- came back "
        "'unavailable' on a live request instead of falling back to their most recent known season. This is exactly the bug class this script exists to catch.")

    live_warned = sum(1 for r in live_results if any("prior-season wage is from the" in w for w in r.warnings))
    expected_live_warnings = sum(live_expected_staleness)
    warning_parity = all(
        any("prior-season wage is from the" in w for w in result.warnings) == expected
        for result, expected in zip(live_results, live_expected_staleness)
    )
    add(checks, "live_request_staleness_warning_present", live_warned, expected_live_warnings, warning_parity,
        "Every live request using a fallback older than its immediate prior season must warn, while requests covered by "
        "the live salary overlay must not be mislabeled stale.")

    hist_false_positive = sum(1 for r in historical_results if any("prior-season wage is from the" in w for w in r.warnings))
    add(checks, "historical_request_no_false_positive_staleness_warning", hist_false_positive, 0, hist_false_positive == 0,
        "A decision_date inside the panel's own covered season must NOT trigger a staleness warning -- guards against the fix over-firing.")

    # --- Same pattern for the pre-365-day sporting-window coverage gap.
    live_pre365_warned = sum(1 for r in live_results if any("pre-365-day sporting window extends" in w for w in r.warnings))
    hist_pre365_false_positive = sum(1 for r in historical_results if any("pre-365-day sporting window extends" in w for w in r.warnings))
    sporting_gap_days = (TODAY - pd.Timestamp(cutoffs["sporting_data_max_date"])).days
    if sporting_gap_days > 0:
        add(checks, "live_request_pre365_coverage_gap_warning_present", live_pre365_warned, ">0", live_pre365_warned > 0,
            f"Sporting data cutoff is {cutoffs['sporting_data_max_date']}, {sporting_gap_days} day(s) behind today -- "
            "live requests should be warning about the resulting coverage gap.")
    add(checks, "historical_request_no_false_positive_pre365_warning", hist_pre365_false_positive, 0, hist_pre365_false_positive == 0,
        "A decision_date well inside the sporting data's coverage must not trigger the coverage-gap warning.")

    # --- Tier 1 metrics added 2026-08-12 (permanent-relationship
    # probability, contract-duration peer, wage-to-market-value peer) draw
    # on the SAME salary-panel features as the bug above -- verify the fix
    # actually propagates to them through the full engine, on the same real
    # sample, rather than trusting the structural "shared feature dict"
    # argument on faith.
    print("Checking Tier 1 metrics (permanent-relationship, duration peer, wage/value peer) through the full engine...")
    builder = IncumbentExtensionProfileBuilder()
    engine_exceptions: list[str] = []
    permanent_supported = stay24_supported = both_supported = ordering_violations = 0
    preservation12_supported = preservation24_supported = tier2_ordering_violations = 0
    horizon_policy_violations = 0
    duration_supported = wage_value_supported = engine_scored = 0
    for row in sample.itertuples(index=False):
        player_rows = players_lookup.loc[players_lookup["canonical_name_normalized"].eq(row.player_name_normalized)]
        if player_rows.empty:
            continue
        player_id = int(player_rows.iloc[0]["canonical_player_id"])
        club_id = int(row.canonical_club_id)
        try:
            profile = builder.build_profile(
                canonical_player_id=player_id, canonical_club_id=club_id, competition_id="GB1",
                decision_date=TODAY.date().isoformat(), proposed_annual_fixed_wage_eur=1_000_000.0,
                proposed_contract_years=2.0, current_public_market_value_eur=1_000_000.0,
                player_name_normalized=row.player_name_normalized,
            )
        except ValueError as exc:
            if "age_at_decision must be between" in str(exc):
                # Expected, not a bug: some sampled salary-panel rows are old
                # enough (panel stops at 2023-24) that the player's age AS OF
                # TODAY legitimately falls outside the engine's guarded
                # range -- the correct behavior is to refuse, exactly as
                # happened here. Not part of what this check measures.
                continue
            engine_exceptions.append(f"{row.player_name_normalized}: {exc!r}")
            continue
        except Exception as exc:  # noqa: BLE001
            engine_exceptions.append(f"{row.player_name_normalized}: {exc!r}")
            continue
        if profile["status"] != "scored":
            continue
        engine_scored += 1
        cont, offer = profile["cards"]["continuity"], profile["cards"]["offer_context"]
        permanent = next(m for m in cont if m["metric_key"] == "permanent_relationship_probability_24m")
        stay_horizon = next(m for m in cont if m["metric_key"] == "contract_continuity_horizon")
        duration = next(m for m in offer if m["metric_key"] == "contract_duration_peer_delta_years")
        wage_value = next(m for m in offer if m["metric_key"] == "wage_to_market_value_peer_comparison")
        permanent_supported += permanent["status"] != "unavailable"
        stay24_supported += stay_horizon["status"] != "unavailable"
        if permanent["status"] != "unavailable" and stay_horizon["status"] != "unavailable":
            both_supported += 1
            stay_24_value = stay_horizon["components"]["continuous_stay_probability_24m"]
            if stay_24_value is not None and permanent["value"] < stay_24_value - 1e-9:
                ordering_violations += 1
        duration_supported += duration["status"] != "unavailable"
        wage_value_supported += wage_value["status"] != "unavailable"

        # --- Tier 2: 12 months is the primary product horizon. The 24-month
        # estimates remain available only as explicitly provisional detail.
        # Check both mathematical ordering and the evidence-policy contract so
        # a later UI/refactor cannot silently promote the underpowered horizon.
        asset = profile["cards"]["asset_value"]
        preservation_12m = next(m for m in asset if m["metric_key"] == "value_preservation_probability_12m")
        preservation_24m = next(m for m in asset if m["metric_key"] == "value_preservation_probability_24m")
        downside_12m = next(m for m in asset if m["metric_key"] == "public_value_downside_25pct_scenario_12m")
        downside_24m = next(m for m in asset if m["metric_key"] == "public_value_downside_25pct_scenario")
        downside_50 = next((m for m in asset if m["metric_key"] == "public_value_downside_50pct_scenario"), None)
        preservation12_supported += preservation_12m["status"] != "unavailable"
        preservation24_supported += preservation_24m["status"] != "unavailable"

        if downside_12m["status"] != "unavailable":
            primary_components = downside_12m.get("components", {})
            if (
                downside_12m.get("status") != "primary_horizon_candidate"
                or downside_12m.get("warning") != PRIMARY_VALUE_HORIZON_WARNING
                or primary_components.get("horizon_months") != 12
                or primary_components.get("evidence_policy") != "primary_horizon"
            ):
                horizon_policy_violations += 1
        if downside_24m["status"] != "unavailable":
            provisional_components = downside_24m.get("components", {})
            if (
                downside_24m.get("status") != "provisional_pending_later_validation"
                or downside_24m.get("warning") != PROVISIONAL_24M_VALUE_WARNING
                or provisional_components.get("horizon_months") != 24
                or provisional_components.get("evidence_policy") != "provisional_pending_later_validation"
                or provisional_components.get("later_player_disjoint_rows") != 26
                or provisional_components.get("smallest_big_five_league_rows") != 3
            ):
                horizon_policy_violations += 1

        if preservation_12m["status"] != "unavailable" and downside_12m["status"] != "unavailable":
            downside_10_12m = 1 - preservation_12m["value"]
            p25_12m = downside_12m["value"]["probability"]
            if p25_12m > downside_10_12m + 1e-9:
                tier2_ordering_violations += 1
        if preservation_24m["status"] != "unavailable" and downside_24m["status"] != "unavailable":
            downside_10_24m = 1 - preservation_24m["value"]
            p25 = downside_24m["value"]["probability"]
            if p25 > downside_10_24m + 1e-9:
                tier2_ordering_violations += 1
            if downside_50 is not None and downside_50["status"] != "unavailable":
                p50 = downside_50["value"]["probability"]
                if p50 > p25 + 1e-9:
                    tier2_ordering_violations += 1

    add(checks, "tier1_engine_scored_without_exception", len(engine_exceptions), 0, not engine_exceptions,
        f"{engine_scored} profiles scored through the full engine. Failures: {engine_exceptions[:5] or 'none'}")
    add(checks, "tier1_permanent_relationship_availability_matches_continuity", permanent_supported, stay24_supported,
        permanent_supported == stay24_supported,
        f"permanent_relationship supported for {permanent_supported}/{engine_scored}; continuous_stay(24m) supported for {stay24_supported}/{engine_scored} -- "
        "both draw on the same survival-model feature set and should have matching availability.")
    add(checks, "tier1_permanent_relationship_ordering_never_violated", ordering_violations, 0, ordering_violations == 0,
        f"Checked {both_supported} profiles where both were available; permanent-relationship probability must never be lower than continuous-stay.")
    duration_floor = max(1, int(engine_scored * 0.9))
    add(checks, "tier1_duration_peer_supported_rate", duration_supported, f">={duration_floor}", duration_supported >= duration_floor,
        f"{duration_supported}/{engine_scored} live profiles got a supported contract-duration peer estimate.")
    add(checks, "tier1_wage_value_peer_supported_rate", wage_value_supported, f">={duration_floor}", wage_value_supported >= duration_floor,
        f"{wage_value_supported}/{engine_scored} live profiles got a supported wage-to-value peer estimate.")
    add(checks, "tier2_primary_12m_value_preservation_supported_rate", preservation12_supported, f">={duration_floor}", preservation12_supported >= duration_floor,
        f"{preservation12_supported}/{engine_scored} live profiles got a supported 12-month value-preservation probability.")
    add(checks, "tier2_provisional_24m_value_preservation_supported_rate", preservation24_supported, f">={duration_floor}", preservation24_supported >= duration_floor,
        f"{preservation24_supported}/{engine_scored} live profiles got a supported but explicitly provisional 24-month value-preservation probability.")
    add(checks, "tier2_public_value_horizon_policy_never_violated", horizon_policy_violations, 0, horizon_policy_violations == 0,
        "Every available 12-month estimate must carry the primary-horizon evidence contract, while every available 24-month estimate "
        "must carry the exact provisional warning and the 26-row / 3-row-smallest-league bottleneck disclosure.")
    add(checks, "tier2_downside_threshold_ordering_never_violated", tier2_ordering_violations, 0, tier2_ordering_violations == 0,
        f"Checked {preservation12_supported} primary 12-month and {preservation24_supported} provisional 24-month profiles for non-increasing severity order "
        "after the scorer's PAVA correction -- this is exactly the class of bug the permanent-relationship check caught minutes earlier, "
        "checked here for the newly-deployed 10% threshold specifically.")

    result = pd.DataFrame(checks)
    result.to_csv(OUTPUT / "live_request_integrity.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    payload = {
        "checks_passed": int(result["passed"].sum()),
        "checks_total": len(result),
        "all_checks_passed": bool(result["passed"].all()),
        "run_date": TODAY.date().isoformat(),
        "data_source_cutoffs": cutoffs,
        "sample_size": len(sample),
        "usable_pairs": len(live_results),
    }
    (OUTPUT / "live_request_integrity.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    print(result.to_string(index=False))
    print(f"\n{payload['checks_passed']}/{payload['checks_total']} checks passed.")
    if not payload["all_checks_passed"]:
        failed = result.loc[~result["passed"], "check"].tolist()
        raise RuntimeError(f"Live-request integrity audit failed: {failed}")


if __name__ == "__main__":
    main()
