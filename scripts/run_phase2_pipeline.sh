#!/usr/bin/env bash
set -Eeuo pipefail

# Continue a converged ground-truth-routing checkpoint into the manuscript's
# scheduled-sampling curriculum, then evaluate the best Phase-2 model.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ID=${RUN_ID:?RUN_ID must identify the training run}
PHASE1_DIR=${PHASE1_DIR:?PHASE1_DIR must identify the converged Phase-1 run}
PHASE1_CHECKPOINT=${PHASE1_CHECKPOINT:-$PHASE1_DIR/best_checkpoint.pt}
PHASE2_RESUME_STATE=${PHASE2_RESUME_STATE:-}
RUN_ROOT=${RUN_ROOT:-params/full_${RUN_ID}}
RESULT_ROOT=${RESULT_ROOT:-results/full_${RUN_ID}}
LOG_ROOT=${LOG_ROOT:-logs/full_${RUN_ID}}
AMP_DTYPE=${AMP_DTYPE:-bfloat16}
CODA_NUM_WORKERS=${CODA_NUM_WORKERS:-4}
CODA_VAL_NUM_WORKERS=${CODA_VAL_NUM_WORKERS:-$CODA_NUM_WORKERS}

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

for value_name in CODA_NUM_WORKERS CODA_VAL_NUM_WORKERS; do
    value=${!value_name}
    if [[ ! "$value" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a non-negative integer, got: $value" >&2
        exit 2
    fi
done
case "$AMP_DTYPE" in
    bfloat16|float16) ;;
    *) echo "AMP_DTYPE must be bfloat16 or float16, got: $AMP_DTYPE" >&2; exit 3 ;;
esac
if [[ ! -f "$PHASE1_CHECKPOINT" ]]; then
    echo "Phase-1 checkpoint is missing: $PHASE1_CHECKPOINT" >&2
    exit 4
fi

mkdir -p "$RUN_ROOT/phase2" "$RESULT_ROOT" "$LOG_ROOT"
PID_FILE="$LOG_ROOT/phase2_pipeline.pid"
if [[ -s "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Phase-2 pipeline is already running as PID $old_pid" >&2
        exit 5
    fi
fi
printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
    rm -f "$PID_FILE"
}
trap cleanup EXIT

echo "[$(date -u +%FT%TZ)] Test suite gate"
"$PYTHON_BIN" -m unittest discover -s tests -v \
    2>&1 | tee "$LOG_ROOT/tests_phase2.log"

IR_ARGS=()
if [[ -d data/irs/openair ]]; then
    IR_ARGS=(--ir_path data/irs/openair)
fi
PHASE2_INIT_ARGS=(--param_path "$PHASE1_CHECKPOINT")
PHASE2_TEE_ARGS=()
if [[ -n "$PHASE2_RESUME_STATE" ]]; then
    if [[ ! -f "$PHASE2_RESUME_STATE" ]]; then
        echo "Phase-2 resume checkpoint is missing: $PHASE2_RESUME_STATE" >&2
        exit 6
    fi
    PHASE2_INIT_ARGS=(--resume_state "$PHASE2_RESUME_STATE")
    PHASE2_TEE_ARGS=(-a)
fi

echo "[$(date -u +%FT%TZ)] Phase 2: scheduled sampling (20 epochs)"
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

PHASE2_DIR=$(find "$RUN_ROOT/phase2" -mindepth 1 -maxdepth 1 \
    -type d -name '*coda_full_phase2' -print | sort | tail -n 1)
if [[ -z "$PHASE2_DIR" || ! -f "$PHASE2_DIR/best_model.pt" || \
      ! -f "$PHASE2_DIR/best_checkpoint.pt" ]]; then
    echo "Phase 2 did not produce complete best-model artifacts" >&2
    exit 7
fi

env \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_ID="$RUN_ID" \
    PARAM_PATH="$PHASE2_DIR/best_model.pt" \
    PHASE1_DIR="$PHASE1_DIR" \
    PHASE2_DIR="$PHASE2_DIR" \
    RUN_ROOT="$RUN_ROOT" \
    RESULT_ROOT="$RESULT_ROOT" \
    LOG_ROOT="$LOG_ROOT" \
    EVAL_LABEL="CODA full curriculum" \
    bash scripts/run_final_evaluation.sh

echo "[$(date -u +%FT%TZ)] Phase-2 curriculum pipeline complete"
