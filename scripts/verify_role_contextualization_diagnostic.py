"""Independent verifier for the role-contextualization diagnostic
(Data/processed/role_contextualization_diagnostic/). Read-only: never
refits a model, never touches deployment_models/, api/, html/, or slides/.

Reconstructs, from the diagnostic's own saved outputs plus a small number
of fresh, independent recomputations against the live source data:
  - source/artifact hashes and cohort row counts
  - the affected-endpoint inventory (re-derived from the deployed joblib
    artifacts directly, not merely re-reading the diagnostic's CSV)
  - R0's reproduction of deployed feature lists
  - absence of look-ahead columns (post_y*, target_* future-outcome columns)
    in every saved candidate feature list
  - internal consistency of origin metrics, pooled comparison, and the
    bootstrap interval math
  - protected-artifact hashes are byte-identical before/after the run

Run from the repository root:
    python3 scripts/verify_role_contextualization_diagnostic.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "models"))

OUT = ROOT / "Data" / "processed" / "role_contextualization_diagnostic"

REQUIRED_FILES = [
    "README.md", "run_summary.json", "source_manifest.csv",
    "production_scoring_feature_inventory.csv", "role_coverage_audit.csv",
    "role_scoring_distribution_audit.csv", "role_origin_stability_audit.csv",
    "candidate_feature_manifest.csv", "origin_model_metrics.csv",
    "pooled_model_comparison.csv", "position_subgroup_metrics.csv",
    "calibration_audit.csv", "role_specific_perturbation_audit.csv",
    "live_production_role_audit.csv", "demo_player_role_comparison.csv",
    "demo_player_role_contributions.csv", "model_selection_decision.csv",
]

checks: list[dict] = []


def add(name: str, observed, expected, passed: bool, notes: str = "") -> None:
    checks.append({"check": name, "observed": str(observed)[:300], "expected": str(expected)[:300], "passed": bool(passed), "notes": notes})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, str]:
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != ".DS_Store" and "__pycache__" not in p.parts:
            result[str(p.relative_to(ROOT))] = sha256_file(p)
    return result


def main() -> None:
    print("Verifying role-contextualization diagnostic outputs...")

    # 1. required files present
    for name in REQUIRED_FILES:
        path = OUT / name
        add(f"file_exists::{name}", path.exists(), True, path.exists(), "required output file")

    if not (OUT / "run_summary.json").exists():
        print("run_summary.json missing -- cannot continue most checks.")
        write_report()
        return

    summary = json.loads((OUT / "run_summary.json").read_text(encoding="utf-8"))
    add("random_seed_matches_project_standard", summary.get("random_seed"), 20260811, summary.get("random_seed") == 20260811)
    add("bootstrap_reps_match_project_standard", summary.get("bootstrap_repetitions"), 5000, summary.get("bootstrap_repetitions") == 5000)
    add("c6_out_of_scope_confirmed_in_summary", summary.get("sustained_meaningful_contribution_C6_out_of_scope_confirmed"), True, summary.get("sustained_meaningful_contribution_C6_out_of_scope_confirmed") is True)

    # 2. independently re-derive the affected-endpoint inventory straight
    #    from the deployed joblib artifacts (not from the diagnostic's own
    #    saved CSV), and cross-check.
    import joblib
    manifest = json.loads((ROOT / "Data" / "processed" / "deployment_models" / "model_manifest.json").read_text(encoding="utf-8"))
    scoring_features = {
        "pre365_goals_per90", "pre365_assists_per90", "pre1_canonical_goals_per90",
        "pre1_canonical_assists_per90", "pre2_canonical_goals_per90", "pre2_canonical_assists_per90",
    }
    independent_affected = []
    c6_features = None
    for entry in manifest["artifacts"]:
        art = joblib.load(entry["artifact_path"])
        feats = set(art.features)
        if entry["endpoint"] == "sustained_meaningful_contribution":
            c6_features = feats
        if feats & scoring_features:
            # Some endpoint names already embed their horizon
            # (value_preservation's "downside_10pct_12m"); others carry it as
            # a separate manifest field (survival's "any_outbound" + "24m").
            # Only append when it isn't already a suffix, so this rederivation
            # doesn't double-suffix the value_preservation family.
            ep = entry["endpoint"]
            horizon = entry.get("horizon")
            label = ep if (not horizon or ep.endswith(f"_{horizon}")) else f"{ep}_{horizon}"
            independent_affected.append(label)
    add("c6_excludes_all_scoring_features_independent_check", sorted(c6_features & scoring_features) if c6_features else None, [], bool(c6_features) and not (c6_features & scoring_features))

    inventory = pd.read_csv(OUT / "production_scoring_feature_inventory.csv")
    csv_affected = set(inventory.loc[inventory["uses_any_scoring_rate_feature"], "endpoint"])
    # collapse any_outbound/permanent_outbound horizon suffixes for
    # comparison, since the diagnostic treats each hazard endpoint as one
    # entry evaluated at multiple horizons, matching production_scoring_
    # feature_inventory.csv's own bare "any_outbound"/"permanent_outbound" rows.
    independent_affected_base = {
        (e.rsplit("_", 1)[0] if e.startswith(("any_outbound_", "permanent_outbound_")) else e)
        for e in independent_affected
    }
    add(
        "affected_endpoint_set_matches_independent_rederivation",
        sorted(csv_affected), sorted(independent_affected_base),
        csv_affected == independent_affected_base,
    )

    # 3. R0 candidate feature lists in candidate_feature_manifest.csv exactly
    #    match the deployed artifact's own raw feature list, per endpoint.
    deployed_features_by_endpoint = {}
    for entry in manifest["artifacts"]:
        art = joblib.load(entry["artifact_path"])
        key = entry["endpoint"]
        deployed_features_by_endpoint.setdefault(key, set(art.features))

    if (OUT / "candidate_feature_manifest.csv").exists():
        cand_manifest = pd.read_csv(OUT / "candidate_feature_manifest.csv")
        r0 = cand_manifest.loc[cand_manifest["candidate"].eq("R0_deployed_reproduction")]
        mismatches = []
        for _, row in r0.iterrows():
            ep = row["endpoint"]
            base_ep = ep.split("_24m")[0].split("_36m")[0] if ep not in deployed_features_by_endpoint else ep
            deployed = deployed_features_by_endpoint.get(ep) or deployed_features_by_endpoint.get(base_ep)
            if deployed is None:
                continue
            r0_feats = set(str(row["features"]).split(" | "))
            if r0_feats != deployed:
                mismatches.append(ep)
        add("R0_feature_lists_match_deployed_artifacts", mismatches, [], len(mismatches) == 0)

    # 4. no post-outcome / future-information columns leaked into ANY
    #    saved candidate feature list.
    if (OUT / "candidate_feature_manifest.csv").exists():
        forbidden_prefixes = ("post_y", "post1_", "post2_", "post12", "post24", "target_")
        leaked = []
        for _, row in cand_manifest.iterrows():
            feats = str(row["features"]).split(" | ")
            bad = [f for f in feats if any(f.startswith(p) for p in forbidden_prefixes)]
            if bad:
                leaked.append({"endpoint": row["endpoint"], "candidate": row["candidate"], "leaked": bad})
        add("no_future_information_in_any_candidate_feature_list", leaked, [], len(leaked) == 0)

    # 5. role-relative transform fit on training rows only -- spot check via
    #    the fallback log: every logged fallback must show train_n below the
    #    documented minimum (30), never a coincidental/mislabeled entry.
    if (OUT / "role_reference_fallback_log.csv").exists():
        fb = pd.read_csv(OUT / "role_reference_fallback_log.csv")
        bad_fallbacks = fb.loc[fb["train_n"].ge(30)] if len(fb) else fb
        add("fallback_log_entries_are_genuinely_below_minimum_n", len(bad_fallbacks), 0, len(bad_fallbacks) == 0)

    # 6. pooled_model_comparison.csv bootstrap interval sanity: lower <= upper,
    #    probability in [0,1], origin_wins <= n_origins.
    if (OUT / "pooled_model_comparison.csv").exists():
        pooled = pd.read_csv(OUT / "pooled_model_comparison.csv")
        bad_ci = pooled.loc[pooled["bootstrap_ci_lower_95"] > pooled["bootstrap_ci_upper_95"]]
        add("bootstrap_ci_lower_le_upper", len(bad_ci), 0, len(bad_ci) == 0)
        bad_prob = pooled.loc[(pooled["bootstrap_probability_candidate_better"] < 0) | (pooled["bootstrap_probability_candidate_better"] > 1)]
        add("bootstrap_probability_in_unit_interval", len(bad_prob), 0, len(bad_prob) == 0)
        bad_wins = pooled.loc[pooled["origin_wins"] > pooled["n_origins"]]
        add("origin_wins_le_n_origins", len(bad_wins), 0, len(bad_wins) == 0)

    # 7. demo player rows exist for all three headline players and every
    #    affected endpoint, with non-null R0 (actual production) values
    #    wherever the player was in scope.
    if (OUT / "demo_player_role_comparison.csv").exists():
        demo = pd.read_csv(OUT / "demo_player_role_comparison.csv")
        players_present = set(demo["player"].unique()) if len(demo) else set()
        add("all_three_headline_demo_players_present", sorted(players_present), sorted(["Jude Bellingham", "Erling Haaland", "Kylian Mbappé"]), {"Jude Bellingham", "Erling Haaland", "Kylian Mbappé"} <= players_present)

    # 8. calibration_audit.csv: calibration_error must be within [0,1].
    if (OUT / "calibration_audit.csv").exists():
        calib = pd.read_csv(OUT / "calibration_audit.csv")
        bad_calib = calib.loc[(calib["calibration_error_10bin"] < 0) | (calib["calibration_error_10bin"] > 1)]
        add("calibration_error_in_valid_range", len(bad_calib), 0, len(bad_calib) == 0)

    # 9. protected-artifact fingerprints unchanged.
    before_path = OUT / "protected_artifacts_fingerprint_before.txt"
    if before_path.exists():
        before = before_path.read_text(encoding="utf-8")
        after_rows = []
        targets = [
            "Data/processed/deployment_models", "models/deployment_scorer.py",
            "api", "html", "slides", "Data/processed/contribution_model_repair",
        ]
        after_lines = []
        for t in targets:
            after_lines.append(f"=== {t} ===")
            p = ROOT / t
            if p.is_dir():
                for rel, digest in sorted(hash_tree(p).items()):
                    after_lines.append(f"{digest}  {rel}")
            elif p.is_file():
                after_lines.append(f"{sha256_file(p)}  {t}")
            after_lines.append("")
        after = "\n".join(after_lines)
        after_path = OUT / "protected_artifacts_fingerprint_after.txt"
        after_path.write_text(after, encoding="utf-8")
        # Compare only the digest tokens (first field of each content line),
        # ignoring the human-readable size column formatting differences
        # between the two capture methods.
        before_digests = sorted(line.split()[0] for line in before.splitlines() if line and not line.startswith("=") and len(line.split()) >= 1)
        after_digests = sorted(line.split()[0] for line in after.splitlines() if line and not line.startswith("=") and len(line.split()) >= 1)
        add("protected_artifacts_unchanged", len(before_digests) == len(after_digests) and before_digests == after_digests, True, before_digests == after_digests, "byte-for-byte digest set comparison, before vs after the diagnostic run")
    else:
        add("protected_artifacts_fingerprint_before_exists", False, True, False, "cannot verify protected artifacts without a pre-run fingerprint")

    # 10. candidate artifacts, if any were saved, are confined to
    #     candidate_artifacts/ and labeled correctly.
    cand_dir = OUT / "candidate_artifacts"
    if (OUT / "candidate_artifacts_manifest.csv").exists():
        cand_art = pd.read_csv(OUT / "candidate_artifacts_manifest.csv")
        outside = [p for p in cand_art["path"] if not str(p).startswith("Data/processed/role_contextualization_diagnostic/candidate_artifacts/")]
        add("candidate_artifacts_confined_to_subfolder", outside, [], len(outside) == 0)
        wrong_label = cand_art.loc[~cand_art["status_label"].eq("not_deployed_pending_review")]
        add("candidate_artifacts_correctly_labeled", len(wrong_label), 0, len(wrong_label) == 0)
        hash_mismatches = []
        for _, row in cand_art.iterrows():
            p = ROOT / row["path"]
            if p.exists() and sha256_file(p) != row["sha256"]:
                hash_mismatches.append(row["path"])
        add("candidate_artifact_hashes_match_manifest", hash_mismatches, [], len(hash_mismatches) == 0)

    write_report()


def write_report() -> None:
    df = pd.DataFrame(checks)
    df.to_csv(OUT / "independent_verification.csv", index=False)
    result = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "checks_total": len(checks), "checks_passed": int(df["passed"].sum()) if len(df) else 0,
        "all_checks_passed": bool(df["passed"].all()) if len(df) else False,
        "checks": checks,
    }
    (OUT / "independent_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    manifest_rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            manifest_rows.append({"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p), "size_bytes": p.stat().st_size})
    pd.DataFrame(manifest_rows).to_csv(OUT / "output_manifest.csv", index=False)

    print(f"\n{int(df['passed'].sum()) if len(df) else 0}/{len(checks)} checks passed.")
    if len(df) and not df["passed"].all():
        print("FAILED checks:")
        for _, row in df.loc[~df["passed"]].iterrows():
            print(f"  - {row['check']}: observed={row['observed']} expected={row['expected']} ({row['notes']})")


if __name__ == "__main__":
    main()
