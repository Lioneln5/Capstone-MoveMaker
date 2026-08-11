# MoveMaker integration crosswalks

Generated from the cleaned FBref, current Transfermarkt, and supplemental Transfermarkt datalake layers.

## Identity policy

- Native Transfermarkt player, club, and competition IDs are authoritative across the two Transfermarkt collections.
- FBref players are auto-accepted only with deterministic name-plus-birth-year evidence or strong season/club evidence.
- FBref clubs are resolved at squad-competition-season grain first, then rolled up to a stable squad mapping.
- Ambiguous and unmatched records remain in the outputs with `manual_review_required = True`.
- Crosswalks do not overwrite or deduplicate source facts.

## Coverage

- Player identities: 170,991 canonical Transfermarkt IDs.
- FBref player identities accepted: 5,735 of 5,967.
- Club identities: 46,490 canonical Transfermarkt IDs.
- FBref club-season mappings accepted: 684 of 684.
- Competitions: 1,684 canonical codes; all 5 FBref codes covered.
- Season representations: 510.

See `crosswalk_dictionary.csv`, `crosswalk_checks.csv`, and `crosswalk_summary.csv` for the audit trail.
