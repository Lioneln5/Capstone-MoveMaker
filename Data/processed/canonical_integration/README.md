# MoveMaker canonical integration

This layer integrates the cleaned current Transfermarkt collection, the supplemental Transfermarkt datalake, and accepted FBref identity crosswalks.

## Policies

- Transfermarkt numeric player and club IDs remain the canonical native identifiers.
- 19,601 fact-only player IDs are retained as explicit stubs; no names or biographies are invented.
- Stable player and club attributes use documented source precedence with source-specific conflict flags.
- Current club, contract, agent, roster, and other latest-state fields remain in snapshot evidence tables and are not treated as timeless attributes.
- Valuation evidence is row-preserving. One canonical player-date spine row is created; conflicting values remain unresolved and the canonical value is blank.
- Transfer evidence is row-preserving. One canonical event spine is created for complete natural keys; incomplete keys remain source-row-specific.
- Literal zero fees remain `unclassified_zero`; they are never inferred to be free transfers.
- Player performance, injuries, national performance, teammates, hierarchy, and standings are not merged here.

## Result

- Canonical players: 190,592
- Canonical clubs: 46,490
- Valuation evidence: 1,409,244 rows → 1,408,834 canonical player-date rows
- Transfer evidence: 1,136,579 rows → 1,106,278 canonical event rows
- Blocking checks: 14; failures: 0
