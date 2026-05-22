#!/usr/bin/env python3
"""Run all 5 pool models individually on 500 test questions. No routing."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_routing.load_qa_dataset import load_qa_dataset_raw
from skillorchestra.routing.pool_service import call_pool_models, PoolCallResult
from skillorchestra.eval import compute_exact_match

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("model_pool_baseline")

POOL_MODELS = [
    "qwen2.5-7b-instruct",
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "mistral-7b-instruct",
    "mixtral-8x22b-instruct",
    "gemma2-27b-it",
]
OUTPUT_DIR = Path("output/model_pool_baseline")
DATASET = "nq_test_qwen"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset: {DATASET}")
    samples = load_qa_dataset_raw(DATASET, max_samples=500)
    logger.info(f"Loaded {len(samples)} samples")

    results = {m: {"correct": 0, "total": 0, "errors": 0, "records": []} for m in POOL_MODELS}

    # Global per-model stats
    per_model_correct = {m: 0 for m in POOL_MODELS}
    per_model_total = {m: 0 for m in POOL_MODELS}
    per_model_errors = {m: 0 for m in POOL_MODELS}

    t0 = time.time()

    for i, sample in enumerate(samples):
        question = sample["question"]
        golden = sample.get("golden_answers", [])

        logger.info(f"[{i+1}/{len(samples)}] {question[:80]}...")

        # Call all 5 models in parallel on this question
        pool_results: dict[str, PoolCallResult] = call_pool_models(
            model_keys=POOL_MODELS,
            query=question,
            max_tokens=1024,
            temperature=0.6,
        )

        for model_key, result in pool_results.items():
            per_model_total[model_key] += 1
            if not result.success:
                per_model_errors[model_key] += 1
                results[model_key]["records"].append({
                    "id": sample["id"],
                    "question": question,
                    "golden_answers": golden,
                    "response": "",
                    "error": result.error,
                    "em": 0.0,
                })
                continue

            response = result.response
            em = compute_exact_match(response, golden)
            if em == 1.0:
                per_model_correct[model_key] += 1

            results[model_key]["records"].append({
                "id": sample["id"],
                "question": question,
                "golden_answers": golden,
                "response": response,
                "em": em,
            })

        # Log per-model running accuracy every 10 samples
        if (i + 1) % 10 == 0:
            parts = []
            for m in POOL_MODELS:
                t = per_model_total[m]
                c = per_model_correct[m]
                acc = c / t * 100 if t > 0 else 0
                parts.append(f"{m.split('-')[0][:6]}={acc:.1f}%")
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            logger.info(f"  [{i+1}/{len(samples)}] {' | '.join(parts)} | {rate:.1f} qpm")

    elapsed = time.time() - t0

    # --- Save per-model results ---
    summary = {}
    for model_key in POOL_MODELS:
        records = results[model_key]["records"]
        n = len(records)
        correct = sum(r["em"] for r in records)
        n_errors = sum(1 for r in records if r.get("error"))
        n_valid = n - n_errors
        valid_correct = sum(r["em"] for r in records if not r.get("error"))

        out_path = OUTPUT_DIR / f"{model_key}.jsonl"
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        summary[model_key] = {
            "total": n,
            "correct": int(correct),
            "em": round(correct / n, 4) if n else 0,
            "errors": n_errors,
            "valid_samples": n_valid,
            "em_valid": round(valid_correct / n_valid, 4) if n_valid else 0,
        }
        logger.info(f"{model_key}: EM={correct}/{n} = {correct/n*100:.2f}% "
                     f"(valid: {valid_correct}/{n_valid} = {valid_correct/n_valid*100:.2f}% "
                     f"errors: {n_errors})")

    overall = {
        "dataset": DATASET,
        "num_samples": len(samples),
        "models": POOL_MODELS,
        "per_model": summary,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    # Print final table
    logger.info("=" * 60)
    logger.info("Final Results:")
    for m in POOL_MODELS:
        s = summary[m]
        logger.info(f"  {m}: EM={s['em']*100:.2f}% (valid={s['em_valid']*100:.2f}%, errors={s['errors']})")

    # Oracle: any model correct
    oracle_correct = 0
    for i in range(len(samples)):
        if any(results[m]["records"][i]["em"] > 0 for m in POOL_MODELS):
            oracle_correct += 1
    oracle_em = oracle_correct / len(samples)
    logger.info(f"  Oracle (any model): {oracle_correct}/{len(samples)} = {oracle_em*100:.2f}%")
    overall["oracle_em"] = round(oracle_em, 4)

    # Print discriminability distribution
    logger.info("=" * 60)
    logger.info("Solvability Cardinality Distribution:")
    n_models = len(POOL_MODELS)
    cardinality_counts = {k: 0 for k in range(n_models + 1)}
    for i in range(len(samples)):
        n_correct = sum(1 for m in POOL_MODELS if results[m]["records"][i]["em"] > 0)
        cardinality_counts[n_correct] += 1
    for k in range(n_models + 1):
        logger.info(f"  {k}/{n_models} models correct: {cardinality_counts[k]} questions ({cardinality_counts[k]/len(samples)*100:.1f}%)")

    overall["cardinality_distribution"] = cardinality_counts

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(overall, f, indent=2)

    logger.info(f"Results saved to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
