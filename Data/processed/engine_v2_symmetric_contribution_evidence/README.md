# Engine V2 symmetric contribution-evidence repair

This stage implements the first required repair from the future-role target
audit. A contribution label is now available only when **both Year 1 and Year
2** independently have a fully observable window, at least 10 captured
extension-club matches, and a nonmissing same-club opportunity share.

## Result

- Frozen reference evaluation rows: 929
- Symmetric-evidence evaluation rows: 922
- Rows changed from a binary label to unavailable: 7
- Eligible rows whose binary label changed: 0
- Positive labels in the symmetric evaluation cohort: 453

The seven excluded rows are not recoded as failures. Their target is nullable
and carries an explicit unavailability reason.

## Boundary

This is **target-repair step 1 only**. It does not repair incomplete schedules
after relegation, resolve the six shares above 100%, fit a candidate model,
open the 2024+ final cohort, or authorize deployment. The target is named
**sustained realized extension-club contribution**, not future role when
available.
