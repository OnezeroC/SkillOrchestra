#!/bin/bash
# Full pipeline: explore→learn→select with 50 samples,
# then test with search1 and search2 handbooks on 500 samples.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$REPO/output/oracle_baseline"

# Load .env
set -a
source "$REPO/.env" 2>/dev/null || true
set +a
unset base_url api_key

mkdir -p "$OUT_DIR"

START_ALL=$SECONDS

echo "============================================================"
echo "Phase 1: explore → learn → select"
echo "Dataset: nq_validation_qwen (50 samples)"
echo "Pool: 6 models"
echo "Started at $(date +%H:%M:%S)"
echo "============================================================"

conda run -n so_env python "$REPO/scripts/pipeline.py" model-routing \
    --dataset nq_validation_qwen \
    --run-dir "$OUT_DIR" \
    --phases explore,learn,select \
    --exploration-samples 50 \
    --router-model qwen2.5-3b-instruct

ELAPSED1=$((SECONDS - START_ALL))
echo ""
echo "Phase 1 done in ${ELAPSED1}s ($((ELAPSED1 / 60))min)"
echo ""

# ---------------------------------------------------------------------------
# Find the latest candidate handbooks
# ---------------------------------------------------------------------------
CANDIDATES_DIR="$OUT_DIR/model-routing_nq_validation_qwen/candidates"

SEARCH1=$(find "$CANDIDATES_DIR" -name "search1.json" -type f 2>/dev/null | head -1)
SEARCH2=$(find "$CANDIDATES_DIR" -name "search2.json" -type f 2>/dev/null | head -1)

if [ -z "$SEARCH1" ] || [ -z "$SEARCH2" ]; then
    echo "ERROR: Could not find search1.json or search2.json in $CANDIDATES_DIR"
    exit 1
fi

echo "Found handbooks:"
echo "  search1: $SEARCH1"
echo "  search2: $SEARCH2"
echo ""

# ---------------------------------------------------------------------------
# Phase 2: Test with search1 handbook
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Phase 2a: Test with search1 handbook (500 samples)"
echo "Started at $(date +%H:%M:%S)"
echo "============================================================"

START=$SECONDS

conda run -n so_env python "$REPO/model_routing/test_skill_routing.py" \
    --handbook "$SEARCH1" \
    --dataset nq_test_qwen \
    --max-samples 500 \
    --output-dir "$OUT_DIR/test_search1" \
    --router-model qwen2.5-3b-instruct \
    --routing-strategy weighted_avg \
    --always-use-original-query

ELAPSED=$((SECONDS - START))
echo ""
echo "search1 test done in ${ELAPSED}s ($((ELAPSED / 60))min)"
echo ""

# ---------------------------------------------------------------------------
# Phase 3: Test with search2 handbook
# ---------------------------------------------------------------------------
echo "============================================================"
echo "Phase 2b: Test with search2 handbook (500 samples)"
echo "Started at $(date +%H:%M:%S)"
echo "============================================================"

START=$SECONDS

conda run -n so_env python "$REPO/model_routing/test_skill_routing.py" \
    --handbook "$SEARCH2" \
    --dataset nq_test_qwen \
    --max-samples 500 \
    --output-dir "$OUT_DIR/test_search2" \
    --router-model qwen2.5-3b-instruct \
    --routing-strategy weighted_avg \
    --always-use-original-query

ELAPSED=$((SECONDS - START))
echo ""
echo "search2 test done in ${ELAPSED}s ($((ELAPSED / 60))min)"
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
TOTAL=$((SECONDS - START_ALL))
echo "============================================================"
echo "All phases completed in ${TOTAL}s ($((TOTAL / 60))min)"
echo "Finished at $(date +%H:%M:%S)"
echo "============================================================"

echo ""
echo "Results:"
echo "  Select:    $CANDIDATES_DIR/"
echo "  search1:   $OUT_DIR/test_search1/summary.json"
echo "  search2:   $OUT_DIR/test_search2/summary.json"

# Show key EM numbers if available
for f in "$OUT_DIR/test_search1/summary.json" "$OUT_DIR/test_search2/summary.json"; do
    if [ -f "$f" ]; then
        name=$(basename "$(dirname "$f")")
        em=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('exact_match','N/A'))" 2>/dev/null || echo "N/A")
        echo "  $name EM: $em"
    fi
done
