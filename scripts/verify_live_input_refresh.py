"""Independent structural checks for the append-only live-input overlays."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Data" / "processed" / "live_input_refresh"


def main() -> None:
    performance = pd.read_csv(OUTPUT / "canonical_performance_overlay.csv", low_memory=False)
    salary = pd.read_csv(OUTPUT / "canonical_salary_overlay.csv", low_memory=False)
    summary = json.loads((OUTPUT / "run_summary.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def add(name: str, passed: bool, observed: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "observed": observed})

    performance_key = ["canonical_player_id", "canonical_club_id", "canonical_competition_id", "season_start_year"]
    salary_key = ["player_name_normalized", "canonical_club_id", "source_page_season_start_year"]
    add("frozen research artifacts explicitly preserved", summary.get("frozen_research_artifacts_modified") is False, summary.get("frozen_research_artifacts_modified"))
    add("performance seasons are 2024 and 2025", set(performance["season_start_year"]) == {2024, 2025}, sorted(performance["season_start_year"].unique().tolist()))
    add("performance keys unique", not performance.duplicated(performance_key).any(), int(performance.duplicated(performance_key).sum()))
    add("performance minutes nonnegative", not pd.to_numeric(performance["canonical_minutes_played"], errors="coerce").lt(0).any(), int(pd.to_numeric(performance["canonical_minutes_played"], errors="coerce").lt(0).sum()))
    add("salary seasons are 2024 through 2026", set(salary["source_page_season_start_year"]) == {2024, 2025, 2026}, sorted(salary["source_page_season_start_year"].unique().tolist()))
    add("salary keys unique", not salary.duplicated(salary_key).any(), int(salary.duplicated(salary_key).sum()))
    add("reported zero wages remain missing", not pd.to_numeric(salary["annual_gross_eur"], errors="coerce").eq(0).any(), int(pd.to_numeric(salary["annual_gross_eur"], errors="coerce").eq(0).sum()))

    for player_id, label in [(418560, "Haaland"), (342229, "Mbappe"), (581678, "Bellingham")]:
        player_perf = performance.loc[performance["canonical_player_id"].eq(player_id)]
        player_salary = salary.loc[salary["canonical_player_id"].eq(player_id)]
        add(f"{label} has both new performance seasons", set(player_perf["season_start_year"]) == {2024, 2025}, sorted(player_perf["season_start_year"].unique().tolist()))
        add(f"{label} has a 2025-26 wage", player_salary["source_page_season_start_year"].eq(2025).any(), sorted(player_salary["source_page_season_start_year"].unique().tolist()))

    frame = pd.DataFrame(checks)
    frame.to_csv(OUTPUT / "independent_verification.csv", index=False, encoding="utf-8-sig", lineterminator="\n")
    result = {"all_checks_passed": bool(frame["passed"].all()), "checks": len(frame), "failures": int((~frame["passed"]).sum())}
    (OUTPUT / "independent_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
