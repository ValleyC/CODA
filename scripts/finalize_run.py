"""Validate full-run evidence and atomically emit the completion manifest."""

import argparse
import hashlib
import json
import math
import os
import re


EXPECTED_EVALUATION_COUNTS = {
    "standard": 94,
    "repeat": 66,
    "random": 28,
}
EXPECTED_PHASE2_EPOCHS = 20
EXPECTED_PHASE2_SCHEDULE = {
    "scheduled_sampling": True,
    "ss_max_p": 0.7,
    "ss_ramp_epochs": 5,
}
REQUIRED_RESUME_KEYS = {
    "model_state_dict",
    "criterion_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "rng_state",
}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_torch(path):
    """Load trusted local training artifacts across supported PyTorch versions."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only keyword.
        return torch.load(path, map_location="cpu")


def assert_finite_state(value, artifact_name, location="root"):
    """Recursively reject non-finite tensors and scalar checkpoint metadata."""
    import torch

    tensor_count = 0
    if torch.is_tensor(value):
        tensor_count = 1
        if (value.is_floating_point() or value.is_complex()) \
                and not torch.isfinite(value).all().item():
            raise RuntimeError(
                f"{artifact_name} contains non-finite tensor state at {location}"
            )
    elif isinstance(value, dict):
        for key, child in value.items():
            tensor_count += assert_finite_state(
                child, artifact_name, f"{location}.{key}"
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            tensor_count += assert_finite_state(
                child, artifact_name, f"{location}[{index}]"
            )
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(
            f"{artifact_name} contains non-finite scalar state at {location}"
        )
    elif isinstance(value, complex) and not (
            math.isfinite(value.real) and math.isfinite(value.imag)):
        raise RuntimeError(
            f"{artifact_name} contains non-finite scalar state at {location}"
        )
    return tensor_count


def assert_same_model_state(checkpoint_state, model_state):
    """Prove best_model.pt is exactly the model stored in best_checkpoint.pt."""
    import torch

    if set(checkpoint_state) != set(model_state):
        raise RuntimeError(
            "Phase-2 best model keys do not match its best checkpoint"
        )
    for key in checkpoint_state:
        left = checkpoint_state[key]
        right = model_state[key]
        if not torch.is_tensor(left) or not torch.is_tensor(right) \
                or left.dtype != right.dtype or left.shape != right.shape \
                or not torch.equal(left, right):
            raise RuntimeError(
                "Phase-2 best model differs from its best checkpoint at "
                f"model_state_dict.{key}"
            )


def validate_phase2(phase2_dir, expected_epochs=EXPECTED_PHASE2_EPOCHS):
    """Validate the complete scheduled-sampling curriculum and selected model."""
    if expected_epochs <= 0:
        raise ValueError("expected Phase-2 epochs must be positive")

    expected_names = {
        f"checkpoint_epoch_{epoch:03d}.pt" for epoch in range(expected_epochs)
    }
    actual_names = {
        name for name in os.listdir(phase2_dir)
        if re.fullmatch(r"checkpoint_epoch_\d+\.pt", name)
    }
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise RuntimeError(
            "Phase 2 does not contain exactly "
            f"{expected_epochs} epoch checkpoints: {'; '.join(details)}"
        )

    checkpoint_paths = []
    final_checkpoint = None
    for epoch in range(expected_epochs):
        path = os.path.join(phase2_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Phase 2 is incomplete: missing checkpoint for epoch {epoch}"
            )
        checkpoint = load_torch(path)
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Phase-2 epoch {epoch} checkpoint is not a mapping")
        if checkpoint.get("epoch") != epoch:
            raise RuntimeError(
                f"Phase-2 checkpoint_epoch_{epoch:03d}.pt records epoch "
                f"{checkpoint.get('epoch')}"
            )
        missing = REQUIRED_RESUME_KEYS.difference(checkpoint)
        if missing:
            raise RuntimeError(
                f"Phase-2 epoch {epoch} is not resumable; missing "
                f"{', '.join(sorted(missing))}"
            )
        args = checkpoint.get("args")
        if not isinstance(args, dict):
            raise RuntimeError(f"Phase-2 epoch {epoch} has no argument record")
        expected_args = {
            "num_epochs": expected_epochs,
            **EXPECTED_PHASE2_SCHEDULE,
        }
        for name, expected in expected_args.items():
            actual = args.get(name)
            matches = (
                math.isclose(float(actual), expected, rel_tol=0, abs_tol=1e-12)
                if name == "ss_max_p" and actual is not None
                else actual == expected
            )
            if not matches:
                raise RuntimeError(
                    f"Phase-2 epoch {epoch} records {name}={actual!r}; "
                    f"expected {expected!r}"
                )
        if assert_finite_state(checkpoint, f"Phase-2 epoch {epoch}") == 0:
            raise RuntimeError(f"Phase-2 epoch {epoch} contains no tensor state")
        checkpoint_paths.append(path)
        final_checkpoint = checkpoint

    best_checkpoint_path = os.path.join(phase2_dir, "best_checkpoint.pt")
    best_model_path = os.path.join(phase2_dir, "best_model.pt")
    best_checkpoint = load_torch(best_checkpoint_path)
    best_model = load_torch(best_model_path)
    if not isinstance(best_checkpoint, dict) \
            or not isinstance(best_checkpoint.get("model_state_dict"), dict):
        raise RuntimeError("Phase-2 best checkpoint has no model state")
    if not isinstance(best_model, dict):
        raise RuntimeError("Phase-2 best model is not a state mapping")
    if assert_finite_state(best_checkpoint, "Phase-2 best checkpoint") == 0:
        raise RuntimeError("Phase-2 best checkpoint contains no tensor state")
    if assert_finite_state(best_model, "Phase-2 best model") == 0:
        raise RuntimeError("Phase-2 best model contains no tensor state")

    best_epoch = final_checkpoint.get("best_epoch")
    if not isinstance(best_epoch, int) or not 0 <= best_epoch < expected_epochs:
        raise RuntimeError(
            f"Final Phase-2 checkpoint records invalid best_epoch={best_epoch!r}"
        )
    if best_checkpoint.get("epoch") != best_epoch:
        raise RuntimeError(
            "Phase-2 best checkpoint does not match the final checkpoint's "
            f"best_epoch={best_epoch}"
        )
    assert_same_model_state(best_checkpoint["model_state_dict"], best_model)
    return {
        "expected_epochs": expected_epochs,
        "completed_epochs": expected_epochs,
        "first_epoch": 0,
        "last_epoch": expected_epochs - 1,
        "best_epoch": best_epoch,
        **EXPECTED_PHASE2_SCHEDULE,
        "checkpoint_paths": checkpoint_paths,
    }


def finalize_run(run_id, phase1_dir, phase2_dir, result_root, run_root,
                 jump_manifest_path, test_log_path, final_model_path=None,
                 early_stop_reason=None,
                 expected_phase2_epochs=EXPECTED_PHASE2_EPOCHS):
    """Prove every required artifact is complete before writing status=complete."""
    if phase2_dir is None and not early_stop_reason:
        raise RuntimeError(
            "phase2_dir is required unless an early-stop reason is recorded"
        )

    phase2_evidence = None
    if phase2_dir is not None:
        phase2_evidence = validate_phase2(
            phase2_dir, expected_epochs=expected_phase2_epochs
        )

    evaluations = {}
    for name, expected_count in EXPECTED_EVALUATION_COUNTS.items():
        path = os.path.join(result_root, f"{name}_summary.json")
        summary = load_json(path)
        if not summary.get("complete"):
            raise RuntimeError(f"{name} evaluation summary is not complete")
        if summary.get("requested_pieces") != expected_count:
            raise RuntimeError(
                f"{name} requested {summary.get('requested_pieces')} pieces; "
                f"expected {expected_count}"
            )
        if summary.get("successful_pieces") != expected_count:
            raise RuntimeError(
                f"{name} completed {summary.get('successful_pieces')} pieces; "
                f"expected {expected_count}"
            )
        if summary.get("failed_pieces"):
            raise RuntimeError(f"{name} contains failed pieces")
        evaluations[name] = {
            "summary": os.path.abspath(path),
            "sha256": sha256(path),
            "pieces": expected_count,
        }

    jump_manifest = load_json(jump_manifest_path)
    if jump_manifest.get("repeat_pieces") != 66:
        raise RuntimeError("jump manifest does not contain 66 annotated pieces")
    if jump_manifest.get("random_pieces") != 28:
        raise RuntimeError("jump manifest does not contain 28 random pieces")
    if len(jump_manifest.get("random_piece_seeds", {})) != 28:
        raise RuntimeError("jump manifest does not record all 28 per-piece seeds")

    with open(test_log_path, encoding="utf-8", errors="replace") as handle:
        tests_text = handle.read()
    test_counts = [
        int(value) for value in re.findall(r"Ran (\d+) tests?", tests_text)
    ]
    if not test_counts or "OK" not in tests_text:
        raise RuntimeError("test log does not prove a passing suite")

    artifact_paths = {
        "phase1_best_checkpoint": os.path.join(
            phase1_dir, "best_checkpoint.pt"
        ),
        "jump_manifest": jump_manifest_path,
        "test_log": test_log_path,
    }
    if phase2_dir is not None:
        artifact_paths["phase2_best_checkpoint"] = os.path.join(
            phase2_dir, "best_checkpoint.pt"
        )
        for epoch, checkpoint_path in enumerate(
                phase2_evidence["checkpoint_paths"]):
            artifact_paths[f"phase2_checkpoint_epoch_{epoch:03d}"] = \
                checkpoint_path
    if final_model_path is None:
        model_dir = phase2_dir if phase2_dir is not None else phase1_dir
        final_model_path = os.path.join(model_dir, "best_model.pt")
    artifact_paths["final_model"] = final_model_path
    artifacts = {}
    for name, path in artifact_paths.items():
        if not os.path.isfile(path):
            raise RuntimeError(f"required artifact is missing: {path}")
        artifacts[name] = {
            "path": os.path.abspath(path),
            "sha256": sha256(path),
        }

    manifest = {
        "format_version": 4,
        "run_id": run_id,
        "phase1_dir": os.path.abspath(phase1_dir),
        "phase2_dir": (
            os.path.abspath(phase2_dir) if phase2_dir is not None else None
        ),
        "result_root": os.path.abspath(result_root),
        "tests_passed": max(test_counts),
        "training": {
            "phase2_completed": phase2_dir is not None,
            "early_stopped": phase2_dir is None,
            "early_stop_reason": early_stop_reason,
            "phase2": (
                {key: value for key, value in phase2_evidence.items()
                 if key != "checkpoint_paths"}
                if phase2_evidence is not None else None
            ),
        },
        "artifacts": artifacts,
        "evaluations": evaluations,
        "status": "complete",
    }
    os.makedirs(run_root, exist_ok=True)
    path = os.path.join(run_root, "run_manifest.json")
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return path, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--phase1_dir", required=True)
    parser.add_argument("--phase2_dir")
    parser.add_argument("--final_model")
    parser.add_argument("--early_stop_reason")
    parser.add_argument(
        "--expected_phase2_epochs", type=int,
        default=EXPECTED_PHASE2_EPOCHS,
    )
    parser.add_argument("--result_root", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--jump_manifest", required=True)
    parser.add_argument("--test_log", required=True)
    args = parser.parse_args()
    path, _ = finalize_run(
        args.run_id, args.phase1_dir, args.phase2_dir, args.result_root,
        args.run_root, args.jump_manifest, args.test_log,
        final_model_path=args.final_model,
        early_stop_reason=args.early_stop_reason,
        expected_phase2_epochs=args.expected_phase2_epochs,
    )
    print("Run manifest:", path)


if __name__ == "__main__":
    main()
