# Engine V2 lagged club-behavior diagnostic

**Status:** movement target contract frozen; lagged club-behavior feature block
tested; no model promoted or deployed.

## What this run changed

The movement question is now encoded as an explicit versioned contract rather
than being inferred differently inside individual experiments. It keeps three
separate concepts:

1. first movement state: no outbound, temporary first, or permanent first;
2. eventual permanent relationship ending within 24 months, even if a loan
   occurred first; and
3. strict meaningful stay: no outbound plus at least 25% same-club opportunity
   share in both post-extension years.

Loan and loan-return records remain separate in the raw audit. They are grouped
only in the modeled temporary-first state. Seven unresolved first-movement
types remain unavailable and are never silently assigned.

The machine-readable contract is
`Data/processed/engine_v2_lagged_club_behavior/movement_target_contract.json`;
the implementation is `models/engine_v2/movement_target_contract.py`.

## Feature construction

For all 2,805 canonical extension anchors, the build created three historical
club profiles:

- **general movement behavior:** recent outbound volume, unique players moved,
  transaction churn, inbound/outbound balance, temporary-movement share, and
  loan-return share;
- **role-conditioned movement behavior:** the same club's movement of players
  in the extension player's broad position and age band; and
- **mature prior-extension behavior:** outcomes of previous same-club
  extensions only after their full 730-day horizon was already observable.

The build scans 46,931 relevant canonical transfer events. Every transfer date
is strictly before the decision date. A prior extension signed on date `d` can
contribute an outcome only at or after `d + 730 days`. Zero prior events remains
an explicit support count of zero; it is not converted into missing data or a
future-derived league average.

General profiles cover every development row. Role-conditioned profiles cover
97.4% of all 2020–2023 canonical anchors and every row in the linked modeling
cohort. Mature prior-extension history is structurally uneven: only 33.6% of
2020 anchors have any prior mature same-club extension, compared with 92.5% in
2023.

## Preregistered rolling-origin test

Four blocks were frozen before evaluation:

| Block | Added information |
| --- | --- |
| C1 | General club movement |
| C2 | Position- and age-conditioned movement |
| C3 | Mature prior-extension outcomes |
| C4 | All behavior features combined |

Each block was compared with the compact Phase-4 baseline retrained on exactly
the same train, validation, and evaluation rows. The primary statistic is
paired Brier-loss improvement with a player-cluster interval adjusted to 98.75%
for the four-block family. Only the established 2020–2023 development origins
were used.

## Results

### Three-state model

| Block | Brier improvement | Adjusted interval | Origin wins | Operational refusal | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| C1 general movement | +0.00159 | [−0.00204, +0.00545] | 3/4 | 6.3% | Inconclusive |
| C2 role-conditioned | **+0.00319** | [−0.00075, +0.00717] | 3/4 | 5.7% | Inconclusive |
| C3 mature extensions | **−0.01232** | [−0.01912, −0.00591] | 1/4 | 27.0% | Repeated harm |
| C4 combined | **−0.01126** | [−0.02044, −0.00227] | 2/4 | 28.3% | Repeated harm |

C2's point improvement is distributed across all three states: +0.00134 for
no outbound, +0.00173 for temporary first, and +0.00013 for permanent first.
It is not statistically secure after the predeclared family adjustment and is
not a promotion result.

### Binary endpoints

The only repeated positive pattern is temporary-first movement:

- C1 improves Brier by +0.00078 but wins only 2/4 origins;
- C2 improves Brier by +0.00088 and wins 4/4 origins, but its adjusted interval
  remains [−0.00118, +0.00292];
- C3 and C4 cause adjusted repeated harm for temporary-first movement; and
- C4 also causes adjusted repeated harm for permanent-first movement.

No block improves eventual permanent separation or strict meaningful stay with
adjusted evidence. The combined block worsens every binary endpoint on the
pooled estimate.

### League localization

For the three-state outcome, C2's Brier change is positive in Bundesliga
(+0.00483), La Liga (+0.00373), Ligue 1 (+0.00329), and Serie A (+0.00556), but
negative in the Premier League (−0.00141). This is encouraging mechanism
evidence for temporary movement, not universal cross-league validation.

## Football interpretation

The small C1/C2 pattern makes football sense. Clubs with recent temporary
movement volume—and especially clubs that frequently move players of the same
position or age band—show a slightly higher chance that an extended player is
loaned or otherwise temporarily displaced first. The standardized coefficient
signs for same-age-band movement count, same-position movement count, and
same-position temporary share point toward temporary-first movement in all
four origins.

That pattern does not solve permanent separation. A club's aggregate transfer
habit is not the same as its intent toward one player. Permanent exits depend
more directly on the player's role trajectory, squad competition, contract
scenario, market demand, and manager decision.

The mature-extension block fails for both statistical and substantive reasons:

- the extension panel begins in 2018, producing structural zeros in early
  origins and much richer histories later;
- 27% of cases fall outside the historically supported feature region;
- clubs change managers, sporting directors, competitive level, and policy, so
  a small set of old extension outcomes is not a stable institutional identity;
- previous extension recipients are themselves a selected mix of prospects,
  core players, and resale-protection decisions; and
- adding those unstable histories overwhelms the smaller, coherent temporary-
  movement signal.

This is not evidence that club policy never matters. It is evidence that the
current short, accumulating same-club extension panel cannot represent it as a
standalone historical rate.

## Decision and next step

- Keep the movement target contract.
- Retain C1 and C2 only as bounded research features for a later combined test;
  do not expose or deploy them independently.
- Reject the current C3 and C4 specifications. Reconsider prior-extension rates
  only with a longer historical panel, time decay, minimum-support rules, and a
  hierarchical design learned strictly inside each origin.
- Build player role trajectory and player-relative squad competition next.
  Those profiles are closer to the actual football mechanisms behind temporary
  displacement, permanent exit, and meaningful retention.

The 2024+ final holdout remains sealed. No model binary, API, site, or frozen V1
artifact changed.

## Verification and evidence

The independent verifier passes 24/24 checks. It reconstructs transfer counts
and mature-extension support directly from source records, recalculates binary
and multiclass losses, repeats both builds in clean directories, and confirms
all 24 frozen V1 hashes.

- `Data/processed/engine_v2_lagged_club_behavior/`
- `Data/processed/engine_v2_lagged_club_behavior_experiment/`
- `Data/processed/engine_v2_lagged_club_behavior_experiment/independent_verification.json`
