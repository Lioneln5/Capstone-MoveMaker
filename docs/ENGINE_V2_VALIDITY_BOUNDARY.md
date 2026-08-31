# Engine V2 Phase 1: validity boundary

Phase 1 answers a question that must precede model refinement: **what is a
MoveMaker request actually allowed to claim, and what evidence must exist
before any score is returned?**

No model was fitted, selected, or deployed in this phase. Public scoring
remains paused.

## Historical decision population

The integrated Capology source contains 2,805 observed incumbent-club
extension events across 2,096 players and 141 clubs in the Big Five leagues,
dated from July 1, 2018 through June 30, 2025.

It does not contain a documented population of:

- players whom a club considered but declined to extend;
- every incumbent player who was contractually eligible for renewal; or
- the club's internal shortlist, negotiation state, medical assessment, or
  alternative offer.

That missing denominator is a material selection limitation. It cannot be
repaired by relabeling other players as negative examples. A future V2 model
may estimate separate historical outcomes **conditional on an extension
scenario**. It may not be described as learning whether the club should offer
an extension or what would have happened without one.

## Request boundary

The machine-readable contract lives in
`models/engine_v2/validity_boundary.py`. It applies two levels of refusal.

### Global refusal

No module may run when:

- the request is not an incumbent-club extension scenario;
- the player or club identity is unresolved;
- the competition is outside the Big Five;
- the decision is future-dated;
- date of birth or broad position is unresolved;
- the player-club relationship lacks supported, dated evidence no more than
  365 days old; or
- the proposed term is outside the two-to-six-year product boundary.

An arbitrary player-club pairing is therefore a hard refusal in V2. V1's
warning-only incumbency path is not carried forward.

### Module-specific refusal

Independent modules remain separate:

- future role and club continuity require sporting data that reaches the
  decision date and at least 10 observed same-club games in the preceding
  365-day window;
- public-value downside additionally requires a positive, dated public value
  no more than 365 days old; and
- wage benchmarking requires a positive proposed fixed wage and a salary
  panel whose coverage reaches the decision date.

Missing evidence disables the affected module. It must never be converted
into an average score, silent older-season fallback, zero, or unsupported
percentile.

## Predictor roles

Phase 1 also freezes the decision-time role of each input group:

- post-extension minutes, departure, and valuation are targets only;
- proposed wage is a financial scenario input, not a sporting predictor;
- current/latest identity snapshots are not historical outcome predictors;
- market value is a public estimate, not sale proceeds or accounting value;
  and
- role and club evidence require as-of provenance.

Role-aware feature design, subgroup support, calibration, endpoint coherence,
and deployment parity remain later Engine V2 phases.

## Reproduce and verify

```bash
python models/run_engine_v2_validity_boundary.py
python scripts/verify_engine_v2_validity_boundary.py
```

Compact evidence is stored under
`Data/processed/engine_v2_validity_boundary/`. The build passes 8/8 checks;
the independent verifier passes 20/20 checks, including adversarial arbitrary
club, stale-source, future-date, unsupported-use-case, and module-isolation
tests. It also confirms all frozen V1 artifact hashes remain byte-identical.
