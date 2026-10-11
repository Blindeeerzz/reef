#!/usr/bin/env bash
# OPD on mathematical reasoning (recipes/opd/examples/math): run.py samples,
# reports and evaluates against the Reef stack this script serves. Setup
# (once): prepare the data and the SFT adapter as the README describes. State
# goes to $OPD_RUN_DIR; the campaign's responses, scores and curve to
# $OPD_RUN_DIR/results.
#
#   ./run.sh                                 the LoRA protocol: 30 updates, evaluated at 0 and 30
#   OPD_STEPS=1 OPD_EVAL_REPEATS=1 ./run.sh  one update and one sample per question, to check the stack
set -euo pipefail
cd "$(dirname "$0")"
export OPD_MODEL_PATH="${OPD_MODEL_PATH:?the student weights, the pinned Qwen3.5-9B-Base directory}"
export OPD_TEACHER_PATH="${OPD_TEACHER_PATH:?the frozen teacher, the pinned Qwen3.5-9B directory}"
export OPD_ADAPTER_PATH="${OPD_ADAPTER_PATH:?the SFT adapter directory, the final/ output of sft.py}"
export OPD_RUN_DIR="${OPD_RUN_DIR:-$PWD/work}"
OPD_DATA="${OPD_DATA:-/work/lora-small/data}"   # prepare_small.py's output
OPD_CONFIG="${OPD_CONFIG:-$PWD/serve.lora-small.yaml}"
OPD_STEPS="${OPD_STEPS:-30}"
REEF_PORT="${REEF_PORT:-28994}"                  # serve.lora-small.yaml's reef.port
REPO_ROOT="$(cd ../../../.. && pwd)"

# Prerequisites: the Slime GPU environment (docker/README.md), the models and the prepared data.
python -c "import reef, reef_client" || { echo "run.sh: run inside the Reef GPU environment" >&2; exit 1; }
for path in "$OPD_MODEL_PATH" "$OPD_TEACHER_PATH" "$OPD_ADAPTER_PATH" "$OPD_DATA/deepmath.jsonl" "$OPD_DATA/aime24.jsonl"; do
  [ -e "$path" ] || { echo "run.sh: $path not found; see README" >&2; exit 1; }
done
mkdir -p "$OPD_RUN_DIR"
[ -f "$OPD_RUN_DIR/token" ] || openssl rand -hex 16 > "$OPD_RUN_DIR/token"
export REEF_TOKEN="$(cat "$OPD_RUN_DIR/token")"
# The example scripts import recipes.opd.examples.math.*, so the repository root must be importable.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -m reef serve -c "$OPD_CONFIG" > "$OPD_RUN_DIR/reef.log" 2>&1 &
reef_pid=$!
cleanup() {
  kill "$reef_pid" 2>/dev/null || true
  wait "$reef_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for _ in $(seq 1 720); do
  if ! kill -0 "$reef_pid" 2>/dev/null; then
    tail -100 "$OPD_RUN_DIR/reef.log"
    exit 1
  fi
  if curl --silent --fail --max-time 5 "http://127.0.0.1:$REEF_PORT/healthz" >/dev/null; then
    ready=1
    break
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "run.sh: Reef did not become ready; see $OPD_RUN_DIR/reef.log" >&2; exit 1; }

# The campaign: 64 prompts by four responses per update under the 16384/64000 budgets,
# the baseline and the final update evaluated on all 30 AIME'24 questions.
python run.py --config "$OPD_CONFIG" --url "http://127.0.0.1:$REEF_PORT" \
  --scenario "${REEF_SCENARIO:-opd-lora-small}" \
  --train-data "$OPD_DATA/deepmath.jsonl" --eval-data "$OPD_DATA/aime24.jsonl" \
  --output "$OPD_RUN_DIR/results" \
  --steps "$OPD_STEPS" --prompts-per-step 64 --samples-per-prompt 4 \
  --train-tokens 16384 --eval-tokens 64000 --eval-repeats "${OPD_EVAL_REPEATS:-16}" \
  --eval-every "$OPD_STEPS" --concurrency 64 "$@"
python analyze.py "$OPD_RUN_DIR/results" --final-step "$OPD_STEPS" --eval-every "$OPD_STEPS" \
  --target-improvement 0.03 --training-mode LoRA
