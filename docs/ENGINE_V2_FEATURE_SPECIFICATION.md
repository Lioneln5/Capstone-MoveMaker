# Engine V2 Phase 2: endpoint and feature specification

Phase 2 translates the Phase-1 validity boundary into a smaller, testable
model design. It audits V1's actual serialized feature lists, tests compact
decision-time-valid candidates on four rolling evaluation years, and declares
which candidates may proceed to Phase 3.

No model artifact was serialized, promoted, or deployed. Public scoring
remains paused.

## Product topology

V2 retains four separate modules rather than a universal score:

1. **Future role:** sustained meaningful same-club contribution in both Year
   1 and Year 2.
2. **Club continuity:** any recorded loan or permanent outbound movement within
   24 months.
3. **Public-value downside:** a decline of at least 25% in the public market-
   value estimate within 24 months.
4. **Historical wage benchmark:** compare a proposed fixed wage with a peer
   estimate without feeding the proposal into the peer model.

The wider V1 grid of 12/24/36-month horizons, multiple downside thresholds,
and auxiliary endpoints is not carried forward as public product output. A
single defensible headline definition per module is easier to validate,
calibrate, explain, and refuse when evidence is weak.

## Feature decision

The predictive candidates use only:

- age, broad position, league, and decision-time public value;
- same-club opportunity share and appearance rate in the preceding 365 days;
- and, for the slightly richer variant, minutes in the prior two completed
  seasons.

They exclude:

- raw goals and assists per 90;
- xG, progressive actions, and pass-completion fields;
- signed or proposed wage, contract duration, age at proposed expiration,
  wage commitment, salary rank/share, and wage-to-value ratios; and
- every post-extension field.

This is stricter than asking whether a feature improved a historical score.
Actual signed terms can encode a club's selection and negotiation process.
That can make them historically predictive while still making them invalid as
user-controlled inputs to a sporting or public-value probability. In V2,
proposed wage and term remain financial scenario facts only.

## What the V1 audit found

The audit reads all 16 frozen serialized V1 artifacts. Twelve used global
goals/assists-rate features. Thirteen violate at least one Phase-2 module
policy because they use raw scoring rates, the unsupported advanced-stat
block, commercial scenario fields, or a combination of those groups.

Those findings do not modify V1. They document why V1 remains frozen.

## Chronological candidate results

All candidates were evaluated at rolling 2020, 2021, 2022, and 2023 origins.
The unit is an extension event, and uncertainty is clustered by player.

| Endpoint | Selected Phase-2 specification | Brier improvement vs chronology-only baseline | 95% player-cluster interval | Origins won | Historical M5 comparison |
| --- | --- | ---: | ---: | ---: | ---: |
| Sustained meaningful contribution | six-field core involvement | +0.05967 | [+0.04651, +0.07259] | 4/4 | -0.00458 |
| Any outbound within 24 months | core plus prior-season minutes | +0.01800 | [+0.01127, +0.02465] | 4/4 | -0.01664 |
| At least 25% public-value downside within 24 months | six-field core involvement | +0.08032 | [+0.06943, +0.09148] | 4/4 | +0.00226 |

The clean contribution and continuity candidates give up some Brier accuracy
relative to the historical M5 ladder, especially continuity. That comparison
is retained as an explicit tradeoff, not an approval gate: M5 contains fields
whose training-time and user-scenario meanings are not equivalent. All three
valid candidates nevertheless improve materially over chronology-only history
and also add value beyond age, position, league, and public value alone.

## Reproduce and verify

```bash
python models/run_engine_v2_feature_specification.py
python scripts/verify_engine_v2_feature_specification.py
```

Evidence is stored under
`Data/processed/engine_v2_feature_specification/`. Phase 3 must now test
calibration, uncertainty, subgroup support, out-of-distribution refusal, and
cross-endpoint behavior before any artifact is eligible for serialization.

