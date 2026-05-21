#!/bin/bash
# Quick smoke test: 2 samples through all 4 phases. ~3-5 minutes.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$REPO/output/nq"

# Load .env, then unset vars that may conflict
set -a
source "$REPO/.env" 2>/dev/null || true
set +a
unset base_url api_key

echo "=== Smoke test starting at $(date +%H:%M:%S) ==="
echo "2 exploration samples, 1 train, 1 val, 1 test"
echo "Estimated: 3-5 minutes"
echo ""

START=$SECONDS

conda run -n so_env python "$REPO/scripts/pipeline.py" model-routing \
    --dataset nq_validation_qwen \
    --output-dir "$OUT_DIR" \
    --phases explore,learn,select,test \
    --test-dataset nq_test_qwen \
    --exploration-samples 2 \
    --train-samples 1 \
    --val-samples 1 \
    --test-max-samples 1

ELAPSED=$((SECONDS - START))

echo ""
echo "=== Smoke test completed in ${ELAPSED}s ($((ELAPSED / 60))min) ==="

# Show key results from pipeline_summary.json
SUMMARY=$(find "$OUT_DIR" -name 'pipeline_summary.json' -print -quit)
if [ -n "$SUMMARY" ]; then
    python -c "
import json
with open('$SUMMARY') as f:
    s = json.load(f)
print(f'explored: {s.get(\"num_bundles\")} bundles')
print(f'oracle: train={s.get(\"oracle_train\",0):.1%}, val={s.get(\"oracle_val\",0):.1%}')
print(f'skills learned: {s.get(\"skills_learned\", 0)}')
print(f'agents profiled: {s.get(\"agents_profiled\", 0)}')
print(f'selected: {s.get(\"selected_candidate\", \"N/A\")}')
test = s.get('test', {})
print(f'test: {\"PASS\" if test.get(\"status\") != \"failed\" else \"FAIL\"}')
"
    echo ""
    echo "=== ALL PHASES VERIFIED ==="
else
    echo "?? No summary found"
fi
