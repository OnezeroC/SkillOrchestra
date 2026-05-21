#!/bin/bash
# Run full SkillOrchestra pipeline (explore→learn→select→test)
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$REPO/output/nq"

# Load .env, then unset vars that may conflict
set -a
source "$REPO/.env" 2>/dev/null || true
set +a
unset base_url api_key

echo "Starting full pipeline at $(date +%H:%M:%S)"
echo "Dataset: nq_validation_qwen (50 samples) → nq_test_qwen (300 samples)"
echo "Phases: explore → learn → select → test"
echo "============================================================"
echo ""

START=$SECONDS

conda run -n so_env python "$REPO/scripts/pipeline.py" model-routing \
    --dataset nq_validation_qwen \
    --output-dir "$OUT_DIR" \
    --phases explore,learn,select,test \
    --test-dataset nq_test_qwen \
    --exploration-samples 50 \
    --test-max-samples 300

ELAPSED=$((SECONDS - START))

echo ""
echo "============================================================"
echo "Pipeline completed in ${ELAPSED}s ($((ELAPSED / 60))min)"
echo "Finished at $(date +%H:%M:%S)"
echo "============================================================"

# Show key results
find "$OUT_DIR" -name 'pipeline.log' -exec grep -iE 'accuracy|score|correct|completed|result|summary|final' {} \;
