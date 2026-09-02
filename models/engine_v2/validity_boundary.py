"""Machine-readable validity boundary for Engine V2 research scoring.

This module deliberately does not load a model.  It answers the prior
question: *is this request inside the population and information boundary
that a future model is allowed to score?*

The historical cohort contains completed incumbent-club extension events,
not a documented set of every player a club considered and declined to
extend.  Passing this contract therefore means only "eligible for research
scenario scoring conditional on an extension".  It never means "extend this
player", "do not extend this player", or "the extension causes the outcome".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Literal

BIG_FIVE = frozenset({"GB1", "ES1", "IT1", "L1", "FR1"})
BROAD_POSITIONS = frozenset({"Goalkeeper", "Defender", "Midfield", "Attack"})
MODULES = ("future_role", "club_continuity", "public_value_downside", "wage_benchmark")
INCUMBENCY_EVIDENCE_TYPES = frozenset({
    "dated_club_appearance",
    "dated_club_roster",
    "dated_club_contract",
    "dated_club_salary_panel",
})

SUPPORTED_CLAIM = "outcome_scenario_conditional_on_extension"
PROHIBITED_CLAIMS = (
    "automated_extend_or_do_not_extend_recommendation",
    "causal_effect_of_extending_the_player",
    "probability_the_extension_will_succeed",
    "new_club_transfer_compatibility",
    "guaranteed_financial_loss_or_savings",
)


@dataclass(frozen=True)
class EngineV2Request:
    """Evidence supplied before any future V2 model is allowed to run.

    Dates are explicit because a current value, current club, or salary row
    without an as-of date cannot establish a decision-time-valid fact.
    """

    use_case: str
    canonical_player_id: int | None
    canonical_club_id: int | None
    competition_id: str | None
    decision_date: date
    player_identity_verified: bool
    club_identity_verified: bool
    date_of_birth: date | None
    broad_position: str | None
    incumbency_verified: bool
    incumbency_evidence_type: str | None
    incumbency_evidence_date: date | None
    sporting_data_cutoff: date | None
    pre365_club_games_observed: int | None
    current_market_value_eur: float | None
    market_value_as_of_date: date | None
    proposed_annual_fixed_wage_eur: float | None
    proposed_contract_years: float | None
    salary_panel_valid_through: date | None


@dataclass(frozen=True)
class ModuleDecision:
    module: str
    eligible: bool
    refusal_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundaryDecision:
    status: Literal["refused", "eligible_for_research_scoring"]
    supported_claim: str
    prohibited_claims: tuple[str, ...]
    global_refusal_codes: tuple[str, ...]
    modules: dict[str, ModuleDecision] = field(default_factory=dict)

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["modules"] = {key: asdict(value) for key, value in self.modules.items()}
        return payload


def _age_on(dob: date, as_of: date) -> float:
    return (as_of - dob).days / 365.25


def _positive(value: float | None) -> bool:
    return value is not None and value > 0


def evaluate_request(request: EngineV2Request, *, today: date | None = None) -> BoundaryDecision:
    """Apply hard population gates and module-specific information gates.

    A global refusal means no module may run.  A module refusal means other
    independent modules may remain eligible, but the refused module must not
    emit a score, fallback percentile, or silently imputed result.
    """

    today = today or date.today()
    global_codes: list[str] = []

    if request.use_case != "incumbent_club_extension":
        global_codes.append("UNSUPPORTED_USE_CASE")
    if not request.player_identity_verified or not request.canonical_player_id or request.canonical_player_id <= 0:
        global_codes.append("PLAYER_IDENTITY_UNVERIFIED")
    if not request.club_identity_verified or not request.canonical_club_id or request.canonical_club_id <= 0:
        global_codes.append("CLUB_IDENTITY_UNVERIFIED")
    if request.competition_id not in BIG_FIVE:
        global_codes.append("LEAGUE_OUTSIDE_BIG_FIVE")
    if request.decision_date > today:
        global_codes.append("FUTURE_DECISION_DATE")
    if request.date_of_birth is None:
        global_codes.append("DATE_OF_BIRTH_UNAVAILABLE")
    else:
        age = _age_on(request.date_of_birth, request.decision_date)
        if age < 16 or age > 46.1:
            global_codes.append("AGE_OUTSIDE_HISTORICAL_SUPPORT")
    if request.broad_position not in BROAD_POSITIONS:
        global_codes.append("BROAD_POSITION_UNRESOLVED")
    if not request.incumbency_verified:
        global_codes.append("INCUMBENCY_UNVERIFIED")
    if request.incumbency_evidence_type not in INCUMBENCY_EVIDENCE_TYPES:
        global_codes.append("INCUMBENCY_EVIDENCE_UNSUPPORTED")
    if request.incumbency_evidence_date is None:
        global_codes.append("INCUMBENCY_EVIDENCE_UNDATED")
    elif request.incumbency_evidence_date > request.decision_date:
        global_codes.append("INCUMBENCY_EVIDENCE_AFTER_DECISION")
    elif (request.decision_date - request.incumbency_evidence_date).days > 365:
        global_codes.append("INCUMBENCY_EVIDENCE_STALE")
    if request.proposed_contract_years is None or not 2 <= request.proposed_contract_years <= 6:
        global_codes.append("PROPOSED_TERM_OUTSIDE_SUPPORTED_RANGE")

    modules: dict[str, ModuleDecision] = {}
    if global_codes:
        codes = tuple(dict.fromkeys(global_codes))
        modules = {name: ModuleDecision(name, False, codes) for name in MODULES}
        return BoundaryDecision(
            status="refused",
            supported_claim=SUPPORTED_CLAIM,
            prohibited_claims=PROHIBITED_CLAIMS,
            global_refusal_codes=codes,
            modules=modules,
        )

    sporting_codes: list[str] = []
    if request.sporting_data_cutoff is None:
        sporting_codes.append("SPORTING_DATA_CUTOFF_UNKNOWN")
    elif request.sporting_data_cutoff < request.decision_date:
        sporting_codes.append("SPORTING_DATA_DOES_NOT_REACH_DECISION_DATE")
    if request.pre365_club_games_observed is None or request.pre365_club_games_observed < 10:
        sporting_codes.append("INSUFFICIENT_PRE365_CLUB_GAME_EVIDENCE")

    value_codes = list(sporting_codes)
    if not _positive(request.current_market_value_eur):
        value_codes.append("CURRENT_MARKET_VALUE_UNAVAILABLE")
    if request.market_value_as_of_date is None:
        value_codes.append("MARKET_VALUE_EVIDENCE_UNDATED")
    elif request.market_value_as_of_date > request.decision_date:
        value_codes.append("MARKET_VALUE_AFTER_DECISION")
    elif (request.decision_date - request.market_value_as_of_date).days > 365:
        value_codes.append("MARKET_VALUE_EVIDENCE_STALE")

    wage_codes: list[str] = []
    if not _positive(request.proposed_annual_fixed_wage_eur):
        wage_codes.append("PROPOSED_FIXED_WAGE_INVALID")
    if request.salary_panel_valid_through is None:
        wage_codes.append("SALARY_PANEL_COVERAGE_UNKNOWN")
    elif request.salary_panel_valid_through < request.decision_date:
        wage_codes.append("SALARY_PANEL_DOES_NOT_REACH_DECISION_DATE")

    for name, codes in {
        "future_role": sporting_codes,
        "club_continuity": sporting_codes,
        "public_value_downside": value_codes,
        "wage_benchmark": wage_codes,
    }.items():
        unique = tuple(dict.fromkeys(codes))
        modules[name] = ModuleDecision(name, not unique, unique)

    return BoundaryDecision(
        status="eligible_for_research_scoring" if any(item.eligible for item in modules.values()) else "refused",
        supported_claim=SUPPORTED_CLAIM,
        prohibited_claims=PROHIBITED_CLAIMS,
        global_refusal_codes=(),
        modules=modules,
    )
