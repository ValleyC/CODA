#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate an already-selected checkpoint without starting another training
# phase. This is used when streaming validation has converged and the remaining
# training schedule is intentionally stopped early.

REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
RUN_ID=${RUN_ID:?RUN_ID must identify the training run}
PARAM_PATH=${PARAM_PATH:?PARAM_PATH must point to the selected model}
PHASE1_DIR=${PHASE1_DIR:-$(dirname "$PARAM_PATH")}
PHASE2_DIR=${PHASE2_DIR:-}
RUN_ROOT=${RUN_ROOT:-params/full_${RUN_ID}}
RESULT_ROOT=${RESULT_ROOT:-results/full_${RUN_ID}}
LOG_ROOT=${LOG_ROOT:-logs/full_${RUN_ID}}
EARLY_STOP_REASON=${EARLY_STOP_REASON:-streaming validation converged}
if [[ -n "$PHASE2_DIR" ]]; then
    EVAL_LABEL=${EVAL_LABEL:-CODA full curriculum}
else
    EVAL_LABEL=${EVAL_LABEL:-CODA full retrain (early-stopped)}
fi

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

for required_file in \
    "$PARAM_PATH" \
    "$PHASE1_DIR/best_checkpoint.pt" \
    data/repeat_annotations.json; do
    if [[ ! -f "$required_file" ]]; then
        echo "Required file is missing: $required_file" >&2
        exit 2
    fi
done
if [[ -n "$PHASE2_DIR" ]]; then
    for required_file in \
        "$PHASE2_DIR/best_checkpoint.pt" \
        "$PHASE2_DIR/best_model.pt"; do
        if [[ ! -f "$required_file" ]]; then
            echo "Required Phase-2 file is missing: $required_file" >&2
            exit 5
        fi
    done
fi
for required_dir in data/msmd/msmd_test; do
    if [[ ! -d "$required_dir" ]]; then
        echo "Required directory is missing: $required_dir" >&2
        exit 3
    fi
done

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
PID_FILE="$LOG_ROOT/evaluation.pid"
if [[ -s "$PID_FILE" ]]; then
    old_pid=$(cat "$PID_FILE")
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Evaluation is already running as PID $old_pid" >&2
        exit 4
    fi
fi
printf '%s\n' "$$" > "$PID_FILE"
cleanup() {
    rm -f "$PID_FILE"
}
trap cleanup EXIT

TEST_LOG="$LOG_ROOT/tests_evaluation.log"
echo "[$(date -u +%FT%TZ)] Test suite gate"
"$PYTHON_BIN" -m unittest discover -s tests -v \
    2>&1 | tee "$TEST_LOG"

echo "[$(date -u +%FT%TZ)] Regenerating clean, fixed repeat benchmarks"
"$PYTHON_BIN" scripts/generate_repeat_test.py \
    --input_dir data/msmd/msmd_test \
    --output_dir data/msmd/msmd_test_jump \
    --annotations data/repeat_annotations.json \
    --seed 42 \
    --clean_output 2>&1 | tee "$LOG_ROOT/generate_repeat.log"

echo "[$(date -u +%FT%TZ)] Standard test evaluation"
"$PYTHON_BIN" scripts/evaluate_batch.py \
    --param_path "$PARAM_PATH" \
    --test_dir data/msmd/msmd_test \
    --label "$EVAL_LABEL" \
    --metrics_dir "$RESULT_ROOT/metrics/standard" \
    --save_summary "$RESULT_ROOT/standard_summary.json" \
    --benchmark 2>&1 | tee "$LOG_ROOT/evaluate_standard.log"

for subset in repeat random; do
    echo "[$(date -u +%FT%TZ)] Jump-recovery evaluation: $subset"
    "$PYTHON_BIN" scripts/evaluate_batch.py \
        --param_path "$PARAM_PATH" \
        --test_dir "data/msmd/msmd_test_jump/$subset" \
        --break_mode \
        --label "$EVAL_LABEL - $subset" \
        --metrics_dir "$RESULT_ROOT/metrics/$subset" \
        --save_summary "$RESULT_ROOT/${subset}_summary.json" \
        --benchmark 2>&1 | tee "$LOG_ROOT/evaluate_${subset}.log"
done

FINALIZE_TRAINING_ARGS=(--final_model "$PARAM_PATH")
if [[ -n "$PHASE2_DIR" ]]; then
    FINALIZE_TRAINING_ARGS+=(--phase2_dir "$PHASE2_DIR")
else
    FINALIZE_TRAINING_ARGS+=(--early_stop_reason "$EARLY_STOP_REASON")
fi
"$PYTHON_BIN" scripts/finalize_run.py \
    --run_id "$RUN_ID" \
    --phase1_dir "$PHASE1_DIR" \
    "${FINALIZE_TRAINING_ARGS[@]}" \
    --result_root "$RESULT_ROOT" \
    --run_root "$RUN_ROOT" \
    --jump_manifest data/msmd/msmd_test_jump/manifest.json \
    --test_log "$TEST_LOG"

echo "[$(date -u +%FT%TZ)] Evaluation pipeline complete"
