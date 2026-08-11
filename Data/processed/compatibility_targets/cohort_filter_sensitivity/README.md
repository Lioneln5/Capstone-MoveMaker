# MoveMaker cohort-filter sensitivity diagnostic

This diagnostic leaves the frozen primary cohort and every upstream model stage unchanged. It relaxes the arrival window, 450 destination-minute threshold, and outfield-only restriction one at a time, plus all three together. Every comparison uses ridge with `player_history_baseline` versus ridge with `full_explicit_fit` across the same four expanding rolling origins.

The opportunity target has no primary destination-minute threshold, so `no_destination_minutes_threshold` is expected to reproduce its primary cohort exactly. All-season opportunity results include partial-season arrivals and are diagnostic rather than directly comparable full-season opportunity estimates.

## Results

Effect changes within +/-0.001 MAE are described as approximately unchanged. A confidence interval is described as meaningfully tighter or wider when its width changes by at least 10% relative to the primary row.

### Opportunity

Primary ridge compatibility improvement: 0.002999, 95% CI [-0.007879, 0.014107], pooled evaluation n=207.

- `all_season_arrivals`: eligible cohort 398 (+86); pooled n 262 (+55); improvement 0.001039, which decreased the point effect; CI [-0.006799, 0.009145] was meaningfully tighter.
- `no_destination_minutes_threshold`: eligible cohort 312 (+0); pooled n 207 (+0); improvement 0.002999, which left approximately unchanged the point effect; CI [-0.007879, 0.014107] was not meaningfully changed.
- `all_roles`: eligible cohort 335 (+23); pooled n 222 (+15); improvement 0.000317, which decreased the point effect; CI [-0.007361, 0.008180] was meaningfully tighter.
- `fully_relaxed`: eligible cohort 424 (+112); pooled n 279 (+72); improvement 0.001245, which decreased the point effect; CI [-0.006682, 0.009434] was meaningfully tighter.

### Performance

Primary ridge compatibility improvement: 0.013405, 95% CI [0.005459, 0.020917], pooled evaluation n=172.

- `all_season_arrivals`: eligible cohort 332 (+61); pooled n 212 (+40); improvement 0.000666, which decreased the point effect; CI [-0.007851, 0.009336] was meaningfully wider.
- `no_destination_minutes_threshold`: eligible cohort 312 (+41); pooled n 207 (+35); improvement 0.007005, which decreased the point effect; CI [-0.001017, 0.015122] was not meaningfully changed.
- `all_roles`: eligible cohort 271 (+0); pooled n 172 (+0); improvement 0.013405, which left approximately unchanged the point effect; CI [0.005459, 0.020917] was not meaningfully changed.
- `fully_relaxed`: eligible cohort 398 (+127); pooled n 262 (+90); improvement 0.016972, which increased the point effect; CI [0.007666, 0.026296] was meaningfully wider.

### Adaptation

Primary ridge compatibility improvement: -0.003910, 95% CI [-0.012700, 0.005108], pooled evaluation n=162.

- `all_season_arrivals`: eligible cohort 317 (+61); pooled n 202 (+40); improvement -0.002609, which increased the point effect; CI [-0.013469, 0.008401] was meaningfully wider.
- `no_destination_minutes_threshold`: eligible cohort 289 (+33); pooled n 189 (+27); improvement 0.001028, which increased the point effect; CI [-0.003451, 0.005510] was meaningfully tighter.
- `all_roles`: eligible cohort 256 (+0); pooled n 162 (+0); improvement -0.003910, which left approximately unchanged the point effect; CI [-0.012700, 0.005108] was not meaningfully changed.
- `fully_relaxed`: eligible cohort 373 (+117); pooled n 243 (+81); improvement 0.002977, which increased the point effect; CI [-0.006575, 0.012237] was not meaningfully changed.

## Plain pattern

- Opportunity does not strengthen with more rows. The primary improvement is 0.002999; it falls to 0.001039 with all-season arrivals, 0.000317 with all roles, and 0.001245 when fully relaxed. Those broader intervals are tighter because n grows, but they center much closer to zero.
- Removing only the destination-minute threshold changes performance from 0.013405 to 0.007005 and adaptation from -0.003910 to 0.001028; neither confidence interval tightens meaningfully. That filter is not the main sample-size limiter.
- The all-season performance and adaptation jumps are instability warnings, not clean strengthening. Their first-origin improvements are -0.022290 and -0.008476, respectively, while later-origin changes are near zero and the pooled intervals widen. Fully relaxed adaptation reverses sharply because its first-origin improvement is -0.007764.
- The fully relaxed performance cohort is the largest performance comparison and has a much smaller effect (0.016972) with a tighter interval. Fully relaxed adaptation becomes highly unstable (0.002977). Across targets, cohort composition and early-origin ridge extrapolation matter more than raw row count alone.

## Interpretation boundary

This is a cohort-sensitivity pattern check, not a replacement cohort and not a proof or disproof of compatibility. Larger samples mix in substantively different transfer timing, playing-time reliability, and goalkeeper populations. Point-effect direction, origin consistency, and confidence-interval width should therefore be read together.

## Verification

The separate verifier checks that every relaxed final cohort and every relaxed funnel stage contains its primary counterpart, all declared source hashes remain unchanged, only ridge and the two declared feature sets were scored, and every scored feature source season precedes its destination outcome season.
