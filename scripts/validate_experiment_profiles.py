from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

if __package__:
    from scripts.neggrad_plus_rules import select_candidate
    from scripts.recommend_main_counts import validate_decision_artifact
    from scripts.scrub_sanity_rules import evaluate_mechanism
    from scripts.validate_fine_tune_artifact import (
        validate_artifact as validate_fine_tune_artifact,
    )
    from scripts.validate_scrub_tuning_artifact import (
        validate_artifact as validate_scrub_tuning_artifact,
    )
    from scripts.validate_ssd_artifacts import (
        validate_artifacts as validate_ssd_reproduction_artifacts,
    )
    from scripts.validate_ssd_fisher_sensitivity_artifact import (
        validate_artifact as validate_ssd_fisher_sensitivity_artifact,
    )
else:
    from neggrad_plus_rules import select_candidate
    from recommend_main_counts import validate_decision_artifact
    from scrub_sanity_rules import evaluate_mechanism
    from validate_fine_tune_artifact import validate_artifact as validate_fine_tune_artifact
    from validate_scrub_tuning_artifact import validate_artifact as validate_scrub_tuning_artifact
    from validate_ssd_artifacts import validate_artifacts as validate_ssd_reproduction_artifacts
    from validate_ssd_fisher_sensitivity_artifact import (
        validate_artifact as validate_ssd_fisher_sensitivity_artifact,
    )

REQUIRED = ("pilot.yaml", "main.yaml", "extended.yaml")
READINESS = Path("research/method_readiness.yaml")


def load_profile(path: Path) -> dict[str, object]:
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict) or profile.get("schema_version") != 1:
        raise ValueError(f"{path}: invalid profile schema")
    if not isinstance(profile.get("launch_authorized"), bool):
        raise ValueError(f"{path}: launch_authorized must be explicit")
    if not profile.get("profile") or not profile.get("status"):
        raise ValueError(f"{path}: profile identity and status are required")
    return profile


def validate_scrub_sanity_artifact(result_path: Path, protocol_path: Path) -> dict[str, object]:
    if not result_path.is_file():
        return {"eligible": False, "status": "MISSING", "failures": ["artifact_missing"]}
    failures: list[str] = []
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, yaml.YAMLError) as error:
        return {"eligible": False, "status": "INVALID", "failures": [type(error).__name__]}
    expected_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if result.get("experiment_id") != protocol.get("experiment_id"):
        failures.append("experiment_id_mismatch")
    if result.get("publication_evidence") is not False:
        failures.append("publication_role_invalid")
    if result.get("protocol", {}).get("sha256") != expected_hash:
        failures.append("protocol_hash_mismatch")
    expected_seeds = [int(seed) for seed in protocol["scope"]["original_seeds"]]
    runs = result.get("runs", [])
    if [run.get("original_seed") for run in runs] != expected_seeds:
        failures.append("seed_set_mismatch")
    for run in runs:
        trace = run.get("mechanism_trace", [])
        if not trace:
            failures.append(f"seed_{run.get('original_seed')}_trace_missing")
            continue
        recomputed = evaluate_mechanism(trace)
        if run.get("acceptance") != recomputed or run.get("accepted") is not all(
            recomputed.values()
        ):
            failures.append(f"seed_{run.get('original_seed')}_acceptance_mismatch")
        for label in ("original_checkpoint", "unlearned_checkpoint"):
            reference = run.get(label, {})
            checkpoint = Path(str(reference.get("path", "")))
            if not checkpoint.is_file():
                failures.append(f"seed_{run.get('original_seed')}_{label}_missing")
            elif hashlib.sha256(checkpoint.read_bytes()).hexdigest() != reference.get("sha256"):
                failures.append(f"seed_{run.get('original_seed')}_{label}_hash_mismatch")
    if result.get("overall_acceptance") is not True or not runs:
        failures.append("overall_acceptance_not_true")
    return {
        "eligible": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "artifact": str(result_path),
        "protocol_sha256": expected_hash,
    }


def _checkpoint_valid(reference: dict[str, object]) -> bool:
    path = Path(str(reference.get("path", "")))
    return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == reference.get(
        "sha256"
    )


def validate_neggrad_plus_artifacts(
    stage1_path: Path, stage2_path: Path, protocol_path: Path
) -> dict[str, object]:
    missing = [str(path) for path in (stage1_path, stage2_path) if not path.is_file()]
    if missing:
        return {"eligible": False, "status": "MISSING", "failures": ["artifact_missing"]}
    failures: list[str] = []
    try:
        stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
        stage2 = json.loads(stage2_path.read_text(encoding="utf-8"))
        protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, yaml.YAMLError) as error:
        return {"eligible": False, "status": "INVALID", "failures": [type(error).__name__]}
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    expected_seeds = [int(seed) for seed in protocol["scope"]["original_seeds"]]
    expected_ids = {
        protocol["artifact_policy"]["stage_1_experiment_id"],
        protocol["artifact_policy"]["stage_2_experiment_id"],
    }
    if {stage1.get("experiment_id"), stage2.get("experiment_id")} != expected_ids:
        failures.append("experiment_id_mismatch")
    for stage in (stage1, stage2):
        if stage.get("publication_evidence") is not False:
            failures.append("publication_role_invalid")
        if stage.get("protocol", {}).get("sha256") != protocol_hash:
            failures.append("protocol_hash_mismatch")
    sanity_runs = stage1.get("runs", [])
    if [run.get("seed") for run in sanity_runs] != expected_seeds:
        failures.append("sanity_seed_set_mismatch")
    for run in sanity_runs:
        diagnostics = run.get("diagnostics", {})
        recomputed = diagnostics.get("gradient_l2_mean", 0.0) > 0 and diagnostics.get(
            "after", {}
        ).get("forget_cross_entropy", float("-inf")) > diagnostics.get("before", {}).get(
            "forget_cross_entropy", float("inf")
        )
        if run.get("accepted") is not recomputed:
            failures.append(f"seed_{run.get('seed')}_sanity_acceptance_mismatch")
    if stage1.get("overall_acceptance") is not True or not sanity_runs:
        failures.append("sanity_overall_acceptance_not_true")
    candidate_runs = stage2.get("candidate_runs", [])
    expected_candidates = int(protocol["stage_2_validation_only_tuning"]["candidate_count"])
    expected_pairs = {
        (f"candidate-{index:02d}", seed)
        for index in range(1, expected_candidates + 1)
        for seed in expected_seeds
    }
    observed_pairs = {
        (str(run.get("candidate_id")), int(run.get("original_seed", -1))) for run in candidate_runs
    }
    if len(candidate_runs) != len(expected_pairs) or observed_pairs != expected_pairs:
        failures.append("candidate_seed_grid_incomplete_or_duplicated")
    for run in candidate_runs:
        if not _checkpoint_valid(run.get("checkpoint", {})):
            failures.append(
                f"{run.get('candidate_id')}_seed_{run.get('original_seed')}_checkpoint_invalid"
            )
    originals = stage2.get("original_checkpoints", {})
    for seed in expected_seeds:
        if not _checkpoint_valid(originals.get(str(seed), originals.get(seed, {}))):
            failures.append(f"original_seed_{seed}_checkpoint_invalid")
    if candidate_runs:
        try:
            recomputed_selection = select_candidate(candidate_runs)
        except ValueError:
            failures.append("no_feasible_candidate")
        else:
            if stage2.get("selection") != recomputed_selection:
                failures.append("selection_mismatch")
            final_seeds = [item.get("original_seed") for item in stage2.get("final_test", [])]
            if final_seeds != expected_seeds:
                failures.append("winner_only_test_evaluation_incomplete")
    return {
        "eligible": not failures,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "artifacts": [str(stage1_path), str(stage2_path)],
        "protocol_sha256": protocol_hash,
    }


def _method_readiness(
    profile: dict[str, object], profile_name: str, readiness_path: Path = READINESS
) -> dict[str, dict[str, object]]:
    readiness = {}
    methods = set(profile.get("approximate_methods", []))
    requirements = profile.get("method_requirements", {})
    catalog = yaml.safe_load(readiness_path.read_text(encoding="utf-8"))["methods"]
    profile_class = "pilot" if profile_name == "pilot.yaml" else "main"
    for method in sorted(methods):
        record = catalog.get(method)
        if record is None:
            readiness[method] = {
                "eligible": False,
                "status": "MISSING_READINESS_RECORD",
                "failures": ["method_readiness_record_missing"],
            }
            continue
        failures = []
        if record.get("registered") is not True:
            failures.append("method_not_registered")
        if profile_class not in record.get("eligible_profiles", []):
            failures.extend(record.get(f"blockers_for_{profile_class}", ["profile_not_eligible"]))
        readiness[method] = {
            "eligible": not failures,
            "status": "PASS" if not failures else "BLOCKED",
            "failures": failures,
        }
    if "scrub" in methods:
        requirement = requirements.get("scrub")
        if not requirement:
            readiness["scrub"]["eligible"] = False
            readiness["scrub"]["status"] = "MISSING_REQUIREMENT"
            readiness["scrub"]["failures"].append("profile_requirement_missing")
        else:
            artifact = validate_scrub_sanity_artifact(
                Path(requirement["required_artifact"]), Path(requirement["protocol"])
            )
            if artifact["eligible"]:
                readiness["scrub"]["failures"] = [
                    failure
                    for failure in readiness["scrub"]["failures"]
                    if failure != "passing_scrub_sanity_artifact"
                ]
            else:
                readiness["scrub"]["eligible"] = False
                readiness["scrub"]["status"] = artifact["status"]
                readiness["scrub"]["failures"].extend(artifact["failures"])
            readiness["scrub"]["artifact_validation"] = artifact
            if profile_class == "main":
                tuning_artifact = validate_scrub_tuning_artifact(
                    Path(requirement["required_tuning_artifact"]),
                    Path(requirement["tuning_protocol"]),
                )
                readiness["scrub"]["tuning_artifact_validation"] = tuning_artifact
                if tuning_artifact["eligible"]:
                    readiness["scrub"]["failures"] = [
                        failure
                        for failure in readiness["scrub"]["failures"]
                        if failure != "passing_scrub_tuning_artifact"
                    ]
                else:
                    readiness["scrub"]["failures"].extend(tuning_artifact["failures"])
                    readiness["scrub"]["status"] = tuning_artifact["status"]
                readiness["scrub"]["eligible"] = not readiness["scrub"]["failures"]
            if artifact["eligible"] and (profile_class != "main" or tuning_artifact["eligible"]):
                readiness["scrub"]["eligible"] = not readiness["scrub"]["failures"]
                readiness["scrub"]["status"] = (
                    "PASS" if readiness["scrub"]["eligible"] else "BLOCKED"
                )
    if "neggrad_plus" in methods:
        requirement = requirements.get("neggrad_plus")
        if not requirement:
            readiness["neggrad_plus"]["failures"].append("profile_requirement_missing")
        else:
            artifacts = [Path(value) for value in requirement["required_artifacts"]]
            artifact = validate_neggrad_plus_artifacts(
                artifacts[0], artifacts[1], Path(requirement["protocol"])
            )
            readiness["neggrad_plus"]["artifact_validation"] = artifact
            if artifact["eligible"]:
                readiness["neggrad_plus"]["failures"] = [
                    failure
                    for failure in readiness["neggrad_plus"]["failures"]
                    if failure != "passing_neggrad_plus_reproduction_artifacts"
                ]
            else:
                readiness["neggrad_plus"]["failures"].extend(artifact["failures"])
                readiness["neggrad_plus"]["status"] = artifact["status"]
            readiness["neggrad_plus"]["eligible"] = not readiness["neggrad_plus"]["failures"]
    if "fine_tune" in methods and profile_class == "main":
        requirement = requirements.get("fine_tune")
        if not requirement:
            readiness["fine_tune"]["failures"].append("profile_requirement_missing")
        else:
            artifact = validate_fine_tune_artifact(
                Path(requirement["required_artifact"]), Path(requirement["protocol"])
            )
            readiness["fine_tune"]["artifact_validation"] = artifact
            if artifact["eligible"]:
                readiness["fine_tune"]["failures"] = [
                    failure
                    for failure in readiness["fine_tune"]["failures"]
                    if failure != "passing_fine_tune_tuning_artifact"
                ]
            else:
                readiness["fine_tune"]["failures"].extend(artifact["failures"])
                readiness["fine_tune"]["status"] = artifact["status"]
            readiness["fine_tune"]["eligible"] = not readiness["fine_tune"]["failures"]
    if "ssd" in methods:
        requirement = requirements.get("ssd")
        if not requirement:
            readiness["ssd"]["failures"].append("profile_requirement_missing")
        else:
            sensitivity = validate_ssd_fisher_sensitivity_artifact(
                Path(requirement["required_sensitivity_artifact"]),
                Path(requirement["sensitivity_protocol"]),
            )
            reproduction_paths = [
                Path(value) for value in requirement["required_reproduction_artifacts"]
            ]
            reproduction = validate_ssd_reproduction_artifacts(
                reproduction_paths[0],
                reproduction_paths[1],
                Path(requirement["reproduction_protocol"]),
            )
            readiness["ssd"]["sensitivity_artifact_validation"] = sensitivity
            readiness["ssd"]["reproduction_artifact_validation"] = reproduction
            if sensitivity["eligible"]:
                readiness["ssd"]["failures"] = [
                    failure
                    for failure in readiness["ssd"]["failures"]
                    if failure != "passing_fisher_estimator_sensitivity_artifact"
                ]
            else:
                readiness["ssd"]["failures"].extend(sensitivity["failures"])
            if reproduction["eligible"]:
                readiness["ssd"]["failures"] = [
                    failure
                    for failure in readiness["ssd"]["failures"]
                    if failure != "passing_ssd_reproduction_artifacts"
                ]
            else:
                readiness["ssd"]["failures"].extend(reproduction["failures"])
            readiness["ssd"]["eligible"] = not readiness["ssd"]["failures"]
    for record in readiness.values():
        record["failures"] = list(dict.fromkeys(record["failures"]))
    return readiness


def pilot_job_count(profile: dict[str, object]) -> dict[str, int]:
    datasets = profile["datasets"]
    conditions = profile["deletion_conditions"]
    original_seeds = profile["original_seeds"]
    exact_seeds = profile["exact_retraining_seeds"]
    methods = profile["approximate_methods"]
    candidate_seeds = profile["candidate_seeds"]
    architecture_count = sum(len(item["architectures"]) for item in datasets)
    condition_count = sum(int(item["manifests"]) for item in conditions)
    original_jobs = architecture_count * len(original_seeds)
    exact_jobs = architecture_count * condition_count * len(exact_seeds)
    approximate_jobs = architecture_count * condition_count * len(methods) * len(candidate_seeds)
    return {
        "original_training_jobs": original_jobs,
        "exact_retraining_jobs": exact_jobs,
        "approximate_unlearning_jobs": approximate_jobs,
        "total_jobs_excluding_sequential_development": original_jobs
        + exact_jobs
        + approximate_jobs,
    }


def validate_main_count_artifact(profile: dict[str, object]) -> dict[str, object]:
    requirement = profile.get("count_decision_requirement", {})
    protocol_path = Path(str(requirement.get("protocol", "")))
    artifact_path = Path(str(requirement.get("required_artifact", "")))
    populated = {
        key: profile.get(key)
        for key in ("original_seed_count", "exact_retraining_seed_count", "candidate_seed_count")
        if profile.get(key) is not None
    }
    if not populated:
        return {"status": "NOT_APPLICABLE", "eligible": False, "failures": []}
    validation = validate_decision_artifact(
        protocol_path,
        artifact_path,
        required_status=str(requirement.get("required_status", "READY_FOR_FREEZE")),
    )
    if artifact_path.is_file():
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
        else:
            recommendation = artifact.get("recommendation", {})
            if any(recommendation.get(key) != value for key, value in populated.items()):
                validation["failures"].append("profile_count_does_not_match_decision_artifact")
                validation["eligible"] = False
                validation["status"] = "FAIL"
    validation["failures"] = list(dict.fromkeys(validation["failures"]))
    return validation


def validate_all(root: Path = Path("experiment_profiles")) -> dict[str, object]:
    profiles = {name: load_profile(root / name) for name in REQUIRED}
    main = profiles["main.yaml"]
    readiness = {
        name: _method_readiness(profile, name)
        for name, profile in profiles.items()
        if profile.get("approximate_methods")
    }
    readiness.setdefault("extended.yaml", {})
    unresolved = [
        key
        for key in ("original_seed_count", "exact_retraining_seed_count", "candidate_seed_count")
        if main.get(key) is None
    ]
    count_decision = validate_main_count_artifact(main)
    if count_decision["status"] != "NOT_APPLICABLE" and not count_decision["eligible"]:
        raise ValueError(f"main.yaml: invalid count decision: {count_decision['failures']}")
    if not unresolved:
        raise ValueError("main profile unexpectedly appears executable without a frozen audit")
    for name, profile in profiles.items():
        blockers = [method for method, state in readiness[name].items() if not state["eligible"]]
        if profile["launch_authorized"] and blockers:
            raise ValueError(f"{name}: launch blocked by method readiness: {blockers}")
        if name == "main.yaml" and profile["launch_authorized"] and unresolved:
            raise ValueError(f"{name}: launch blocked by unresolved design fields: {unresolved}")
    return {
        "schema_version": 1,
        "status": "PASS_WITH_MAIN_PROFILE_BLOCKED",
        "profiles": list(REQUIRED),
        "pilot_job_count": pilot_job_count(profiles["pilot.yaml"]),
        "pilot_method_blockers": sorted(
            set(profiles["pilot.yaml"].get("method_blockers", {}))
            | {method for method, state in readiness["pilot.yaml"].items() if not state["eligible"]}
        ),
        "method_readiness": readiness,
        "main_unresolved_fields": unresolved,
        "main_count_decision": count_decision,
    }


def main() -> None:
    result = validate_all()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
