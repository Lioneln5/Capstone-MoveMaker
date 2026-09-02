# Engine V2 player-role and squad-competition profiles

**Status:** research feature build; no model promoted or deployed.

The build converts exact dated lineup selections, appearances, club games,
historical valuations, and incoming transfers into decision-time profiles for
all 2,805 canonical extensions.

## 2020–2023 availability

- current role structure: **92.1%**;
- split-window role trajectory: **85.6%**;
- on-field positional competition: **92.0%**;
- positional value and incoming competition: **88.2%**.

A player's zero start, selection, or minutes rate is accepted only after the
club's lineup/appearance game coverage passes the 80% gate. Missing source
coverage becomes unavailable. Active positional competitors require at least
three prior-year lineup selections. Valuations are on or before signing and no
more than 365 days old.

## Source rows retained for the anchor universe

- club games: 33,996
- lineup rows: 682,489
- appearance rows: 488,651
- incoming transfers: 22,094
- valuation rows: 283,896

The 2024+ outcome holdout is not read by this build.
