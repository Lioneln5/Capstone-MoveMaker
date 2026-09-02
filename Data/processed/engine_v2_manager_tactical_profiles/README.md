# Engine V2 manager and tactical profiles

**Status:** research feature build; no model promoted or deployed.

The build converts exact dated club games, manager labels, reported formation
shapes, player lineups, and appearances into decision-time profiles for all
2,805 canonical extensions.

## 2020–2023 availability

- manager stability: **95.1%**;
- formation/tactical stability: **87.9%**;
- player role under the current manager: **87.9%**;
- player role change around the latest manager transition: **78.3%**.

Manager tenure means the trailing uninterrupted manager-coded match spell, not
an official appointment date. Formation modifiers are reduced to base numeric
shape. Zero player involvement is accepted only after lineup and appearance
coverage passes the 80% gate.

## Source rows retained

- club games: 33,996
- lineup rows: 682,489
- appearance rows: 488,651

The 2024+ outcome holdout is not read by this build.
