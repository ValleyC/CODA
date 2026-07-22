#!/usr/bin/env bash
set -Eeuo pipefail

# Full paper-scale CODA pipeline: 30-epoch GT routing, 20-epoch scheduled
# sampling, then standard and jump-recovery evaluation. The test set is never
# passed to training or model selection.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ID=${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}
RUN_ROOT=${RUN_ROOT:-params/full_${RUN_ID}}
RESULT_ROOT=${RESULT_ROOT:-results/full_${RUN_ID}}
LOG_ROOT=${LOG_ROOT:-logs/full_${RUN_ID}}
PHASE1_RESUME_STATE=${PHASE1_RESUME_STATE:-}
PHASE2_RESUME_STATE=${PHASE2_RESUME_STATE:-}
AMP_DTYPE=${AMP_DTYPE:-}
CODA_NUM_WORKERS=${CODA_NUM_WORKERS:-4}
CODA_VAL_NUM_WORKERS=${CODA_VAL_NUM_WORKERS:-$CODA_NUM_WORKERS}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

if [[ ! "$CODA_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "CODA_NUM_WORKERS must be a non-negative integer, got: $CODA_NUM_WORKERS" >&2
    exit 8
fi
if [[ ! "$CODA_VAL_NUM_WORKERS" =~ ^[0-9]+$ ]]; then
    echo "CODA_VAL_NUM_WORKERS must be a non-negative integer, got: $CODA_VAL_NUM_WORKERS" >&2
    exit 9
fi
echo "Data-loader workers: train=$CODA_NUM_WORKERS validation=$CODA_VAL_NUM_WORKERS"

mkdir -p "$RUN_ROOT/phase1" "$RUN_ROOT/phase2" "$RESULT_ROOT" "$LOG_ROOT"

for required in \
    data/msmd/msmd_train \
    data/msmd/msmd_valid \
    data/msmd/msmd_test; do
    if [[ ! -d "$required" ]]; then
        echo "Required dataset directory is missing: $required" >&2
        exit 2
    fi
done

IR_ARGS=()
if [[ -d data/irs/openair ]]; then
    IR_ARGS=(--ir_path data/irs/openair)
else
    echo "Optional data/irs/openair not found; training without room-response augmentation"
fi

"$PYTHON_BIN" - <<'PY'
import torch
import torchvision
from mamba_ssm.modules.mamba_simple import Mamba

assert torch.cuda.is_available(), "The full pipeline requires a CUDA GPU"
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Mamba:", Mamba.__name__)
PY

if [[ -z "$AMP_DTYPE" ]]; then
    AMP_DTYPE=$("$PYTHON_BIN" - <<'PY'
import torch
print("bfloat16" if torch.cuda.is_bf16_supported() else "float16")
PY
    )
fi
case "$AMP_DTYPE" in
    bfloat16|float16) ;;
    *)
        echo "AMP_DTYPE must be bfloat16 or float16, got: $AMP_DTYPE" >&2
        exit 7
        ;;
esac
echo "Training autocast dtype: $AMP_DTYPE"

echo "[$(date -u +%FT%TZ)] Test suite gate"
"$PYTHON_BIN" -m unittest discover -s tests -v \
    2>&1 | tee "$LOG_ROOT/tests.log"

echo "[$(date -u +%FT%TZ)] Phase 1: ground-truth routing (30 epochs)"
PHASE1_INIT_ARGS=()
PHASE1_TEE_ARGS=()
if [[ -n "$PHASE1_RESUME_STATE" ]]; then
    [[ -f "$PHASE1_RESUME_STATE" ]] || {
        echo "Phase 1 resume checkpoint is missing: $PHASE1_RESUME_STATE" >&2
        exit 5
    }
    PHASE1_INIT_ARGS=(--resume_state "$PHASE1_RESUME_STATE")
    PHASE1_TEE_ARGS=(-a)
fi
"$PYTHON_BIN" scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    --dump_root "$RUN_ROOT/phase1" \
    --log_root "$LOG_ROOT/tensorboard_phase1" \
    --tag coda_full_phase1 \
    --temporal_priors \
    --augment \
    "${IR_ARGS[@]}" \
    --cold_start_prob 0.15 \
    --jump_prob 0.10 \
    --loss_calibration uncertainty \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 5e-4 \
    --num_workers "$CODA_NUM_WORKERS" \
    --val_num_workers "$CODA_VAL_NUM_WORKERS" \
    --amp \
    --amp_dtype "$AMP_DTYPE" \
    --save_every_epoch \
    "${PHASE1_INIT_ARGS[@]}" \
    2>&1 | tee "${PHASE1_TEE_ARGS[@]}" "$LOG_ROOT/phase1.log"

PHASE1_DIR=$(find "$RUN_ROOT/phase1" -mindepth 1 -maxdepth 1 -type d -name '*coda_full_phase1' -print | sort | tail -n 1)
PHASE1_CHECKPOINT="$PHASE1_DIR/best_checkpoint.pt"
if [[ ! -f "$PHASE1_CHECKPOINT" ]]; then
    echo "Phase 1 did not produce $PHASE1_CHECKPOINT" >&2
    exit 3
fi

echo "[$(date -u +%FT%TZ)] Phase 2: scheduled sampling (20 epochs)"
PHASE2_INIT_ARGS=(--param_path "$PHASE1_CHECKPOINT")
PHASE2_TEE_ARGS=()
if [[ -n "$PHASE2_RESUME_STATE" ]]; then
    [[ -f "$PHASE2_RESUME_STATE" ]] || {
        echo "Phase 2 resume checkpoint is missing: $PHASE2_RESUME_STATE" >&2
        exit 6
    }
    PHASE2_INIT_ARGS=(--resume_state "$PHASE2_RESUME_STATE")
    PHASE2_TEE_ARGS=(-a)
fi
"$PYTHON_BIN" scripts/train.py \
    --config configs/coda.yaml \
    --train_sets data/msmd/msmd_train \
    --val_sets data/msmd/msmd_valid \
    "${PHASE2_INIT_ARGS[@]}" \
    --dump_root "$RUN_ROOT/phase2" \
    --log_root "$LOG_ROOT/tensorboard_phase2" \
    --tag coda_full_phase2 \
    --temporal_priors \
    --augment \
    "${IR_ARGS[@]}" \
    --cold_start_prob 0.15 \
    --jump_prob 0.10 \
    --loss_calibration uncertainty \
    --scheduled_sampling \
    --ss_max_p 0.7 \
    --ss_ramp_epochs 5 \
    --batch_size 16 \
    --num_epochs 20 \
    --lr 1e-4 \
    --num_workers "$CODA_NUM_WORKERS" \
    --val_num_workers "$CODA_VAL_NUM_WORKERS" \
    --amp \
    --amp_dtype "$AMP_DTYPE" \
    --save_every_epoch \
    2>&1 | tee "${PHASE2_TEE_ARGS[@]}" "$LOG_ROOT/phase2.log"

PHASE2_DIR=$(find "$RUN_ROOT/phase2" -mindepth 1 -maxdepth 1 -type d -name '*coda_full_phase2' -print | sort | tail -n 1)
FINAL_MODEL="$PHASE2_DIR/best_model.pt"
if [[ ! -f "$FINAL_MODEL" ]]; then
    echo "Phase 2 did not produce $FINAL_MODEL" >&2
    exit 4
fi

echo "[$(date -u +%FT%TZ)] Regenerating clean, fixed repeat benchmarks"
"$PYTHON_BIN" scripts/generate_repeat_test.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --annotations data/repeat_annotations.json \
    --seed 42 \
    --clean_output 2>&1 | tee "$LOG_ROOT/generate_repeat.log"

echo "[$(date -u +%FT%TZ)] Standard test evaluation"
"$PYTHON_BIN" scripts/evaluate_batch.py \
    --param_path "$FINAL_MODEL" \
    --test_dir data/msmd/msmd_test \
    --label "CODA full retrain" \
    --metrics_dir "$RESULT_ROOT/metrics/standard" \
    --save_summary "$RESULT_ROOT/standard_summary.json" \
    --benchmark 2>&1 | tee "$LOG_ROOT/evaluate_standard.log"

for subset in repeat random; do
    echo "[$(date -u +%FT%TZ)] Jump-recovery evaluation: $subset"
    "$PYTHON_BIN" scripts/evaluate_batch.py \
        --param_path "$FINAL_MODEL" \
        --test_dir "data/msmd/msmd_test_jump/$subset" \
        --break_mode \
        --label "CODA full retrain - $subset" \
        --metrics_dir "$RESULT_ROOT/metrics/$subset" \
        --save_summary "$RESULT_ROOT/${subset}_summary.json" \
        --benchmark 2>&1 | tee "$LOG_ROOT/evaluate_${subset}.log"
done

"$PYTHON_BIN" scripts/finalize_run.py \
    --run_id "$RUN_ID" \
    --phase1_dir "$PHASE1_DIR" \
    --phase2_dir "$PHASE2_DIR" \
    --result_root "$RESULT_ROOT" \
    --run_root "$RUN_ROOT" \
    --jump_manifest data/msmd/msmd_test_jump/manifest.json \
    --test_log "$LOG_ROOT/tests.log"

echo "[$(date -u +%FT%TZ)] Full pipeline complete"
