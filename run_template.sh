#!/bin/bash
# Run full SkillOrchestra pipeline across all 7 QA subsets
# Resume-friendly: skips completed exploration subsets, retries on network errors

REPO="$(cd "$(dirname "$0")" && pwd)"
RUN_DIR="$REPO/output/nq/20260515_multiset"

# Load .env, then unset vars that may conflict
set -a
source "$REPO/.env" 2>/dev/null || true
set +a
unset base_url api_key

# 7 validation subsets for exploration
VAL_SUBSETS=(
    "2wikimultihopqa_validation_qwen"
    "bamboogle_validation_qwen"
    "hotpotqa_validation_qwen"
    "musique_validation_qwen"
    "nq_validation_qwen"
    "popqa_validation_qwen"
    "triviaqa_validation_qwen"
)

# 7 test subsets (excluding "default")
TEST_SUBSETS=(
    "2wikimultihopqa_test_qwen"
    "bamboogle_test_qwen"
    "hotpotqa_test_qwen"
    "musique_test_qwen"
    "nq_test_qwen"
    "popqa_test_qwen"
    "triviaqa_test_qwen"
)

POOL_MODELS="qwen2.5-7b-instruct,llama3.1-8b-instruct,llama3.1-70b-instruct,mistral-7b-instruct,mixtral-8x22b-instruct,gemma2-27b-it"

START=$SECONDS

echo "============================================================"
echo "Phase 1: Explore (7 subsets x 50 samples = 350 total)"
echo "Started at $(date +%H:%M:%S)"
echo "============================================================"

mkdir -p "$RUN_DIR/exploration"

for i in "${!VAL_SUBSETS[@]}"; do
    subset="${VAL_SUBSETS[$i]}"
    out_file="$RUN_DIR/exploration/$subset/inference_results.jsonl"

    # Skip if already completed
    if [ -f "$out_file" ] && [ "$(wc -l < "$out_file")" -ge 50 ]; then
        echo ""
        echo "[$((i+1))/7] SKIP $subset (already done: $(wc -l < "$out_file")/50)"
        continue
    fi

    echo ""
    echo "[$((i+1))/7] Exploring $subset (50 samples)..."

    # Retry up to 3 times on network errors
    attempt=0
    while [ $attempt -lt 3 ]; do
        if conda run -n so_env python "$REPO/model_routing/explore.py" \
            --dataset "$subset" \
            --output-dir "$RUN_DIR/exploration/$subset" \
            --max-samples 50 \
            --pool-models "$POOL_MODELS"; then
            break
        else
            attempt=$((attempt + 1))
            if [ $attempt -lt 3 ]; then
                echo "  Retry $attempt/3 in 30s..."
                sleep 30
            else
                echo "  FAILED after 3 attempts: $subset"
            fi
        fi
    done
done

# Merge all exploration results with renumbered sample_ids
echo ""
echo "Merging exploration results with sequential sample_ids..."
conda run -n so_env python -c "
import json, glob, os

base = '$RUN_DIR/exploration'
out_file = os.path.join(base, 'inference_results.jsonl')
files = sorted(glob.glob(os.path.join(base, '*/inference_results.jsonl')))
print(f'Found {len(files)} files to merge')

seq = 0
with open(out_file, 'w') as out:
    for f in files:
        subset = os.path.basename(os.path.dirname(f))
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    obj = json.loads(line)
                    obj['sample_id'] = seq
                    obj['_source'] = subset
                    seq += 1
                    out.write(json.dumps(obj, ensure_ascii=False) + '\n')

print(f'Merged {seq} samples -> {out_file}')
"
TOTAL=$(wc -l < "$RUN_DIR/exploration/inference_results.jsonl")
echo "Total exploration samples: $TOTAL"

echo ""
echo "============================================================"
echo "Phase 2: Learn + Select"
echo "Started at $(date +%H:%M:%S)"
echo "============================================================"

conda run -n so_env python "$REPO/scripts/pipeline.py" model-routing \
    --dataset nq_validation_qwen \
    --output-dir "$RUN_DIR" \
    --phases learn,select \
    --run-dir "$RUN_DIR" \
    --exploration-data "$RUN_DIR/exploration/inference_results.jsonl" \
    --test-dataset nq_test_qwen

echo ""
echo "============================================================"
echo "Phase 3: Test (7 subsets x 100 samples = 700 total)"
echo "Started at $(date +%H:%M:%S)"
echo "============================================================"

# Convert selected handbook to RSL format for testing
SELECTED_HANDBOOK="$RUN_DIR/model-routing_nq_validation_qwen/selected/default.json"
RSL_HANDBOOK="$RUN_DIR/test/rsl_handbook.json"

echo "Converting selected handbook to RSL..."
conda run -n so_env python -c "
from skillorchestra.core.handbook import SkillHandbook
from skillorchestra.converters.to_ar import save_as_rsl
hb = SkillHandbook.load('$SELECTED_HANDBOOK')
save_as_rsl(hb, '$RSL_HANDBOOK')
print(f'RSL handbook saved to $RSL_HANDBOOK')
"

for i in "${!TEST_SUBSETS[@]}"; do
    subset="${TEST_SUBSETS[$i]}"
    echo ""
    echo "[$((i+1))/7] Testing $subset (100 samples)..."
    conda run -n so_env python "$REPO/model_routing/test_skill_routing.py" \
        --handbook "$RSL_HANDBOOK" \
        --dataset "$subset" \
        --output-dir "$RUN_DIR/test/$subset" \
        --router-model qwen2.5-3b-instruct \
        --routing-strategy weighted_avg \
        --max-samples 100
done

# Summary
ELAPSED=$((SECONDS - START))
echo ""
echo "============================================================"
echo "Pipeline completed in ${ELAPSED}s ($((ELAPSED / 60))min)"
echo "Finished at $(date +%H:%M:%S)"
echo "============================================================"
echo ""
echo "Per-subset test results:"
for subset in "${TEST_SUBSETS[@]}"; do
    result_file="$RUN_DIR/test/$subset/results/rsl_routing_results.json"
    if [ -f "$result_file" ]; then
        em=$(conda run -n so_env python -c "import json; d=json.load(open('$result_file')); print(d.get('summary',{}).get('exact_match','N/A'))" 2>/dev/null || echo "N/A")
        echo "  $subset: EM=$em"
    else
        echo "  $subset: no results file"
    fi
done
