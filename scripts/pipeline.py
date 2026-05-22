#!/usr/bin/env python3
"""
Main script to run the full SkillOrchestra Pipeline: explore → learn → select → test.

Usage examples:

  # --- Model routing ---

  # Full pipeline (explore → learn → select → test)
  python scripts/pipeline.py model-routing \\
      --dataset nq_validation_qwen \\
      --output-dir /tmp/so_pipeline/nq \\
      --test-dataset nq_test_qwen

  # Full pipeline with explicit train/val split (20 train, 20 val)
  python scripts/pipeline.py model-routing \\
      --dataset nq_validation_qwen \\
      --output-dir /tmp/so_pipeline/nq \\
      --test-dataset nq_test_qwen \\
      --exploration-samples 40 \\
      --train-samples 20 \\
      --val-samples 20

  # Generate exploration only
  python scripts/pipeline.py model-routing \\
      --dataset nq_validation_qwen \\
      --output-dir /tmp/so_pipeline/nq \\
      --phases explore

  # Use existing exploration data, run learn + select (20 train, 20 val)
  python scripts/pipeline.py model-routing \\
      --dataset nq_validation_qwen \\
      --exploration-data /path/to/inference_results.jsonl \\
      --output-dir /tmp/so_pipeline/nq \\
      --phases learn,select \\
      --train-samples 20 \\
      --val-samples 20

  # Use existing exploration data, run learn + select + test
  python scripts/pipeline.py model-routing \\
      --dataset nq_validation_qwen \\
      --exploration-data /path/to/inference_results.jsonl \\
      --output-dir /tmp/so_pipeline/nq \\
      --test-dataset nq_test_qwen \\
      --phases learn,select,test

  # Learn only (no LLM, manual skills + oracle evaluation)
  python scripts/pipeline.py model-routing \\
      --dataset nq_validation_qwen \\
      --exploration-data /path/to/inference_results.jsonl \\
      --output-dir /tmp/so_pipeline/nq \\
      --phases learn,select \\
      --no-llm

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add so-internal to path and load .env
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from config import resolve_model
from config.pipeline import (
    DATA_DIR,
    DEFAULT_MODEL_CONFIG,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POOL_MODELS,
    RSL_RESULTS_DIR,
)
from skillorchestra.core.handbook import SkillHandbook
from skillorchestra.core.traces import ExplorationBundle
from skillorchestra.converters.from_ar import load_rsl_results, rsl_stats, find_rsl_results
from skillorchestra.converters.to_ar import save_as_rsl
from skillorchestra.selection.candidates import (
    CandidateHandbook,
    generate_depth_candidates,
    compute_mode_max_depths,
)
from skillorchestra.selection.pareto import (
    EvaluationResult,
    evaluate_candidate_oracle,
    select_pareto_optimal,
    select_pareto_optimal_live,
    find_pareto_frontier,
)
from skillorchestra.selection.store import HandbookStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# File handler for pipeline.log (set by _setup_log_file when output_dir is known)
_log_file_handler: Optional[logging.FileHandler] = None


def _setup_log_file(output_dir: Path) -> None:
    """Add a FileHandler to capture all logs to output_dir/pipeline.log."""
    global _log_file_handler
    if _log_file_handler is not None:
        return
    log_path = Path(output_dir) / "pipeline.log"
    try:
        # Write run separator first (before handler adds its first line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*70}\nPipeline run {datetime.now().isoformat()}\n{'='*70}\n")
        _log_file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        _log_file_handler.setLevel(logging.DEBUG)
        _log_file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        logging.getLogger().addHandler(_log_file_handler)
        logger.info(f"Logging to {log_path}")
    except OSError as e:
        logger.warning(f"Could not create log file {log_path}: {e}")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Unified configuration for the full pipeline."""

    task_type: str = "model-routing"
    output_dir: str = ""  # default: output/ in repo
    phases: List[str] = field(default_factory=lambda: ["explore", "learn", "select", "test"])

    # Exploration
    dataset: str = ""  # e.g., "nq_validation_qwen"
    exploration_data: Optional[str] = None  # path to existing exploration data
    samples_path: Optional[str] = None
    max_train_samples: Optional[int] = None
    exploration_epochs: int = 1
    exploration_samples: Optional[int] = None
    exploration_stages: Optional[List[str]] = None
    exploration_models: Optional[List[str]] = None

    # Model routing specific
    pool_models: List[str] = field(default_factory=lambda: DEFAULT_POOL_MODELS.copy())
    router_model: str = "qwen2.5-3b-instruct"
    distributed_config: Optional[str] = None

    model_config: str = str(DEFAULT_MODEL_CONFIG)
    concurrency: int = 20
    tool_concurrency: int = 10
    max_rounds: int = 20
    routing_strategy: str = "weighted_avg"

    # Learning
    use_llm: bool = True
    llm_model: str = "deepseek/deepseek-v4-pro"
    skill_id_model: Optional[str] = None  # Skill identification. If None, use llm_model.
    val_ratio: float = 0.3
    min_val_samples: int = 20  # Minimum validation samples for selection
    max_val_samples: Optional[int] = None  # Cap validation size (e.g. 20 for consistency)
    train_samples: Optional[int] = None  # If set, use first N for learning (overrides ratio)
    val_samples: Optional[int] = None  # If set, use next N for selection (after train_samples)
    validation_samples: Optional[str] = None  # Path to JSONL for exact same validation set
    max_refinement_rounds: int = 3
    max_merge_credits: int = 50  # Max merge candidates to apply per refinement phase
    max_split_credits: int = 1  # Max split candidates to apply per refinement phase
    max_discovery_bundles: Optional[int] = None

    # Selection
    lambda_cost: float = 0.0
    selection_candidates: Optional[List[str]] = None  # Limit to these candidate names (for debugging)

    # Testing
    test_dataset: Optional[str] = None  # model routing: e.g. "nq_test_qwen"
    test_samples: Optional[str] = None  # Path to test samples JSONL
    test_max_samples: Optional[int] = None
    handbook_path: Optional[str] = None  # Override: path to handbook JSON (or dir with handbook.json) for test phase

    # Resuming / existing run
    run_dir: Optional[str] = None  # Use existing run dir (no new timestamp)
    use_existing_candidates: bool = False  # Load candidates from store instead of generating
    dataset_from_cli: bool = False  # True if user passed --dataset (overrides saved when using run_dir)

    @property
    def experiment_name(self) -> str:
        ds = self.dataset
        if "/" in ds:
            ds = Path(ds).stem
        ds = ds.replace("/", "_")
        return f"{self.task_type}_{ds}" if ds else f"{self.task_type}_default"


# ===========================================================================
# Phase 0: Exploration
# ===========================================================================

def phase_explore_model_routing(config: PipelineConfig) -> Path:
    """Generate exploration data by running all pool models on a dataset.

    Calls model_routing/explore.py which loads the dataset from HuggingFace,
    queries every pool model, evaluates correctness, and saves raw
    per-query results as inference_results.jsonl.
    """
    logger.info("=" * 60)
    logger.info("Phase 0: Exploration (Model Routing)")
    logger.info("=" * 60)

    if config.exploration_data:
        p = Path(config.exploration_data)
        if p.exists():
            logger.info(f"Using existing exploration data: {p}")
            return p
        logger.warning(f"Exploration data not found: {p}, will generate")

    explore_dir = (Path(config.output_dir) / "exploration").resolve()
    explore_dir.mkdir(parents=True, exist_ok=True)

    # Use conda so_env python (guarantees all deps like sympy, openai are available)
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    _python_exe = os.path.join(_conda_prefix, "bin", "python") if _conda_prefix else sys.executable
    if not os.path.exists(_python_exe):
        _python_exe = sys.executable

    cmd = [
        _python_exe, str(Path(__file__).parent.parent / "model_routing" / "explore.py"),
        "--dataset", config.dataset,
        "--output-dir", str(explore_dir),
    ]
    if config.exploration_samples:
        cmd += ["--max-samples", str(config.exploration_samples)]
    if config.distributed_config:
        cmd += ["--distributed-config", config.distributed_config]
    if config.pool_models:
        cmd += ["--pool-models", ",".join(config.pool_models)]

    logger.info(f"Running exploration: {' '.join(cmd)}")
    # Run from repo_root so paths and imports resolve correctly
    repo_root = Path(__file__).resolve().parent.parent
    # Pass explicit environment: inherit parent but unset conflicting shell vars
    # to ensure .env settings (OPENAI_BASE_URL / OPENAI_API_KEY) take effect
    child_env = os.environ.copy()
    child_env.pop("base_url", None)
    child_env.pop("api_key", None)
    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        capture_output=True, text=True, timeout=14400,
        env=child_env,
    )

    if result.returncode != 0:
        logger.error(f"Exploration failed:\n{result.stdout[-1000:]}\n{result.stderr[-1000:]}")
        raise RuntimeError(f"Exploration failed with exit code {result.returncode}")

    # Log subprocess output for debugging (even on success)
    if result.stderr.strip():
        for line in result.stderr.strip().split("\n"):
            line = line.strip()
            if line:
                logger.debug(f"[explore] {line}")

    results_file = explore_dir / "inference_results.jsonl"
    if results_file.exists():
        logger.info(f"Exploration complete: {results_file}")
        return results_file

    raise FileNotFoundError(f"No inference_results.jsonl found in {explore_dir}")



# ===========================================================================
# Phase 1: Learning
# ===========================================================================

def phase_learn(
    config: PipelineConfig,
    bundles: List[ExplorationBundle],
    store: HandbookStore,
) -> SkillHandbook:
    """Learn a Skill Handbook from exploration data."""
    logger.info("=" * 60)
    logger.info("Phase 1: Skill Handbook Learning")
    logger.info(f"  Bundles: {len(bundles)}, Use LLM: {config.use_llm}")
    logger.info("=" * 60)

    if config.use_llm:
        return _learn_with_llm(config, bundles, store)
    else:
        return _learn_manual(config, bundles, store)


def _learn_with_llm(
    config: PipelineConfig,
    bundles: List[ExplorationBundle],
    store: HandbookStore,
) -> SkillHandbook:
    """Learn using the full LLM-powered learning pipeline."""
    from skillorchestra.llm.client import LLMClient
    from skillorchestra.learning.pipeline import LearningPipeline, LearningConfig

    openrouter_url = os.environ.get("OPENROUTER_BASE_URL", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if openrouter_url and openrouter_key:
        llm = LLMClient(model=config.llm_model, provider="custom", base_url=openrouter_url, api_key=openrouter_key)
    else:
        llm = LLMClient(model=config.llm_model)

    lconfig = LearningConfig(
        validation_ratio=0.5,
        max_refinement_rounds=config.max_refinement_rounds,
        max_merge_credits=config.max_merge_credits,
        max_split_credits=config.max_split_credits,
        max_discovery_bundles=config.max_discovery_bundles,
        learning_llm_model=config.llm_model,
        skill_id_model=config.skill_id_model,
        llm_base_url=openrouter_url,
        llm_api_key=openrouter_key,
        output_dir=str(Path(config.output_dir) / "learning"),
        experiment_name=config.experiment_name,
    )

    pipeline = LearningPipeline(llm=llm, config=lconfig, store=store)
    result = pipeline.run(bundles)

    logger.info(f"Learning complete: {result.handbook.num_skills} skills")
    logger.info(f"  Stats: {json.dumps({k: v for k, v in result.stats.items() if k != 'end_time' and k != 'start_time'}, indent=2)[:500]}")

    return result.handbook


def _learn_manual(
    config: PipelineConfig,
    bundles: List[ExplorationBundle],
    store: HandbookStore,
) -> SkillHandbook:
    """Build a handbook with manual skills (no LLM calls) for testing."""
    from skillorchestra.core.types import Skill, AgentProfile, BetaCompetence, CostStats

    handbook = SkillHandbook()

    modes = set()
    agent_success: Dict[str, Dict[str, Dict[str, int]]] = {}  # mode -> agent -> {correct, total}
    SKIP_MODES = {"reference"}

    for bundle in bundles:
        for trace in bundle.trajectories:
            mode = trace.varied_mode
            agent = trace.varied_agent_id
            if not mode or not agent or mode in SKIP_MODES:
                continue
            modes.add(mode)
            agent_success.setdefault(mode, {}).setdefault(agent, {"correct": 0, "total": 0})
            agent_success[mode][agent]["total"] += 1
            if trace.task_success:
                agent_success[mode][agent]["correct"] += 1

    for mode in sorted(modes):
        handbook.add_mode(mode, description=f"Operational mode: {mode}")

        skill = Skill(
            skill_id=f"{mode}.general",
            name=f"General {mode}",
            description=f"General capability for {mode} tasks",
            indicators=[],
            examples=[],
            mode=mode,
        )
        handbook.add_skill(skill)

        for agent_id, counts in sorted(agent_success.get(mode, {}).items()):
            profile = handbook.get_or_create_agent_profile(
                agent_id=agent_id, mode=mode, model_name=resolve_model(agent_id),
            )
            bc = BetaCompetence(
                alpha=1 + counts["correct"],
                beta=1 + counts["total"] - counts["correct"],
            )
            profile.skill_competence[skill.skill_id] = bc
            profile.cost_stats = CostStats(total_executions=counts["total"])

    exp_name = config.experiment_name
    store.save_learned(handbook, exp_name)

    logger.info(f"Manual handbook: {handbook.num_skills} skills, {len(handbook.agent_profiles)} agents")
    for mode in sorted(modes):
        agents = handbook.get_agents_for_mode(mode)
        for a in agents:
            bc = a.skill_competence.get(f"{mode}.general")
            if bc:
                logger.info(f"  [{mode}] {a.agent_id}: {bc.mean:.1%} ({bc.total_observations} obs)")

    return handbook


# ===========================================================================
# Phase 2: Selection
# ===========================================================================

def _evaluate_candidates_live_model_routing(
    candidates: List[CandidateHandbook],
    val_bundles: List[ExplorationBundle],
    config: PipelineConfig,
    work_dir: Path,
) -> List[EvaluationResult]:
    """Run real skill routing on validation data for each candidate.

    For each candidate: save as RSL, run test_skill_routing on val samples,
    parse EM and cost from results.
    """
    val_jsonl = work_dir / "validation_samples.jsonl"
    if config.validation_samples:
        import shutil
        shutil.copy(config.validation_samples, val_jsonl)
        n = sum(1 for _ in open(val_jsonl))
        logger.info(f"Using {config.validation_samples} ({n} samples) for validation")
    else:
        with open(val_jsonl, "w") as f:
            for i, b in enumerate(val_bundles):
                f.write(
                    json.dumps(
                        {
                            "id": b.query_id or f"val_{i}",
                            "question": b.query,
                            "ground_truths": b.ground_truths,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        logger.info(f"Wrote {len(val_bundles)} validation samples to {val_jsonl}")

    repo_root = Path(__file__).resolve().parent.parent
    results: List[EvaluationResult] = []

    for candidate in candidates:
        hb_path = work_dir / f"handbook_{candidate.name}.json"
        save_as_rsl(candidate.handbook, hb_path)

        out_dir = work_dir / f"eval_{candidate.name}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use same settings as test_skill_routing (no overrides - same logic, hierarchy, etc.)
        cmd = [
            sys.executable,
            str(repo_root / "model_routing" / "test_skill_routing.py"),
            "--handbook", str(hb_path),
            "--input-file", str(val_jsonl),
            "--output-dir", str(out_dir),
            "--router-model", config.router_model,
            "--routing-strategy", config.routing_strategy,
        ]

        logger.info(f"Running live eval for {candidate.name}...")
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=7200,
            )
            if proc.returncode != 0:
                logger.warning(
                    f"test_skill_routing failed for {candidate.name}: {proc.stderr[:500]}"
                )
                results.append(
                    EvaluationResult(
                        name=candidate.name,
                        accuracy=0.0,
                        avg_cost=0.0,
                        num_skills=candidate.handbook.num_skills,
                        granularity=candidate.granularity,
                        score=0.0,
                        details={"error": proc.stderr[:500]},
                    )
                )
                continue

            inf_path = out_dir / "inference_results.jsonl"
            em = 0.0
            total_cost = 0.0
            n = 0
            if inf_path.exists():
                with open(inf_path) as f:
                    for line in f:
                        r = json.loads(line)
                        em += r.get("exact_match", 0.0)
                        total_cost += r.get("costs", {}).get("total", 0.0)
                        n += 1
            em = em / n if n else 0.0
            avg_cost = total_cost / n if n else 0.0

            score = em - config.lambda_cost * avg_cost
            results.append(
                EvaluationResult(
                    name=candidate.name,
                    accuracy=em,
                    avg_cost=avg_cost,
                    num_skills=candidate.handbook.num_skills,
                    granularity=candidate.granularity,
                    score=score,
                )
            )
            logger.info(
                f"  {candidate.name}: acc={em:.1%}, cost=${avg_cost:.4f}, score={score:.4f}"
            )
        except subprocess.TimeoutExpired:
            logger.warning(f"Timeout for {candidate.name}")
            results.append(
                EvaluationResult(
                    name=candidate.name,
                    accuracy=0.0,
                    avg_cost=0.0,
                    num_skills=candidate.handbook.num_skills,
                    granularity=candidate.granularity,
                    score=0.0,
                    details={"error": "Timeout"},
                )
            )

    return results


def phase_select(
    config: PipelineConfig,
    handbook: SkillHandbook,
    val_bundles: List[ExplorationBundle],
    store: HandbookStore,
) -> CandidateHandbook:
    """Generate candidates and select the Pareto-optimal handbook.

    For model-routing: always runs real skill routing on validation data
    (actual LLM calls). Oracle is saved as reference only, not for selection.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Handbook Selection")
    logger.info("=" * 60)

    exp_name = config.experiment_name

    if config.use_existing_candidates:
        cand_names = store.list_candidates(exp_name)
        if cand_names:
            candidates = [store.load_candidate(n, exp_name) for n in cand_names]
            logger.info(f"Loaded {len(candidates)} existing candidates from store")
        else:
            candidates = []
    else:
        candidates = []

    if not candidates:
        candidates = generate_depth_candidates(handbook)
        if not candidates:
            logger.warning("No depth-based candidates, using full handbook")
            candidates = [CandidateHandbook(
                name="full",
                handbook=handbook.subgraph(),
                granularity="full",
            )]
        store.save_all_candidates(candidates, exp_name)
        logger.info(f"Generated {len(candidates)} candidates")

    if config.selection_candidates:
        want = set(config.selection_candidates)
        all_names = {c.name for c in candidates}
        candidates = [c for c in candidates if c.name in want]
        missing = want - all_names
        if missing:
            logger.warning(f"Candidates not found: {missing}. Available: {sorted(all_names)[:20]}{'...' if len(all_names) > 20 else ''}")
        if not candidates:
            raise RuntimeError(
                f"No candidates match --candidates {config.selection_candidates}. "
                f"Available: {sorted(all_names)}"
            )
        logger.info(f"Limited to {len(candidates)} candidates: {[c.name for c in candidates]}")

    if config.task_type == "model-routing" and len(val_bundles) >= 1:
        logger.info(
            f"Running live evaluation on {len(val_bundles)} validation samples "
            "(real routing, actual LLM calls)"
        )
        work_dir = Path(config.output_dir) / exp_name / "selection_live"
        work_dir.mkdir(parents=True, exist_ok=True)
        all_results = _evaluate_candidates_live_model_routing(
            candidates, val_bundles, config, work_dir,
        )
        best, all_results = select_pareto_optimal_live(
            candidates, all_results, lambda_cost=config.lambda_cost,
        )
        # Optionally log oracle as reference (not used for selection)
        oracle_results = [
            evaluate_candidate_oracle(c, val_bundles, config.lambda_cost)
            for c in candidates
        ]
        store.save_evaluation_results(
            [r.to_dict() for r in oracle_results], exp_name, "oracle_reference.json",
        )
        logger.debug(f"Oracle reference: {[f'{r.name}={r.accuracy:.1%}' for r in oracle_results]}")
    else:
        raise RuntimeError(
            "Selection requires validation data (val or train/val split). "
            "Use --train-samples and --val-samples to provide data for selection."
        )

    store.save_evaluation_results(
        [r.to_dict() for r in all_results], exp_name, "selection_results.json",
    )

    frontier = find_pareto_frontier(all_results)
    logger.info(f"Pareto frontier: {[r.name for r in frontier]}")

    best_result = next(r for r in all_results if r.name == best.name)
    store.save_selected(
        best.handbook, "default", exp_name,
        eval_result={
            "name": best.name,
            "accuracy": best_result.accuracy,
            "avg_cost": best_result.avg_cost,
        },
    )

    logger.info(
        f"Selected: {best.name} (accuracy={best_result.accuracy:.1%}, "
        f"cost=${best_result.avg_cost:.4f})"
    )
    return best


# ===========================================================================
# Phase 3: Testing
# ===========================================================================

def phase_test_model_routing(
    config: PipelineConfig,
    handbook: SkillHandbook,
    store: HandbookStore,
) -> Dict[str, Any]:
    """Test the selected handbook on a test dataset using model_routing/test_skill_routing.py."""
    logger.info("=" * 60)
    logger.info("Phase 3: Testing (Model Routing)")
    logger.info("=" * 60)

    def infer_test_dataset(train_ds: str) -> Optional[str]:
        # QA mapping: *_validation_qwen -> *_test_qwen
        if "_validation_" in train_ds:
            return train_ds.replace("_validation_", "_test_")
        # Math mappings
        if train_ds == "math500-validation":
            return "math500-test"
        if train_ds == "amc-validation-22":
            return "amc-test-23"
        if "-validation" in train_ds:
            return train_ds.replace("-validation", "-test")
        return None

    test_dataset = config.test_dataset
    if not test_dataset:
        train_ds = config.dataset
        test_dataset = infer_test_dataset(train_ds)
        if not test_dataset:
            logger.warning("No test dataset specified and cannot infer from training dataset")
            return {"status": "skipped", "reason": "no test dataset"}

    logger.info(f"Test dataset: {test_dataset}")

    handbook_path = Path(config.output_dir).resolve() / "test" / "rsl_handbook.json"
    save_as_rsl(handbook, handbook_path)

    test_output = Path(config.output_dir).resolve() / "test" / "results"
    test_output.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable, str(repo_root / "model_routing" / "test_skill_routing.py"),
        "--handbook", str(handbook_path),
        "--dataset", test_dataset,
        "--output-dir", str(test_output),
        "--router-model", config.router_model,
        "--routing-strategy", config.routing_strategy,
        "--always-use-original-query",
    ]
    if config.test_max_samples:
        cmd += ["--max-samples", str(config.test_max_samples)]
    if config.distributed_config:
        cmd += ["--distributed-config", config.distributed_config]

    logger.info(f"Running test: {' '.join(cmd)}")

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    result = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True, text=True, timeout=14400,
    )

    if result.returncode != 0:
        logger.error(f"Test failed:\nstdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}")
        return {"status": "failed", "returncode": result.returncode, "stderr": result.stderr[-500:]}

    # Collect all test artifacts into the experiment's evaluation/ directory
    eval_dir = store.experiment_dir(config.experiment_name) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Save the RSL handbook used for testing
    import shutil
    rsl_dest = eval_dir / "rsl_handbook.json"
    shutil.copy2(handbook_path, rsl_dest)
    logger.info(f"Saved test handbook to {rsl_dest}")

    # Find and save full inference results
    inference_file = _find_latest_file(test_output, "inference_results.jsonl")
    if inference_file:
        inf_dest = eval_dir / "inference_results.jsonl"
        shutil.copy2(inference_file, inf_dest)
        logger.info(f"Saved inference results ({inference_file.stat().st_size / 1024:.1f} KB) to {inf_dest}")

    summary = _find_and_load_json(test_output, "summary.json")
    if summary:
        logger.info(f"Test results: {json.dumps(summary, indent=2)[:500]}")
        store.save_evaluation_results(
            [summary], config.experiment_name, "test_results.json",
        )
        return {"status": "success", "summary": summary}

    return {"status": "completed", "output_dir": str(test_output)}


# ===========================================================================
# Handbook loading for test phase
# ===========================================================================


def _load_handbook_for_test(path: str) -> str:
    """Resolve handbook path for test phase. Returns path to handbook JSON.

    - If path is a directory: look for handbook.json or first .json file.
    - If JSON has model_profiles or agent_profiles: return path as-is (RSL format).
    """
    p = Path(path).resolve()
    if p.is_dir():
        candidates = [p / "handbook.json"] + list(p.glob("*.json"))
        for c in candidates:
            if c.exists():
                p = c
                break
        else:
            raise FileNotFoundError(f"No handbook JSON found in directory: {path}")
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Handbook not found: {path}")

    with open(p) as f:
        data = json.load(f)

    if "model_profiles" in data or "agent_profiles" in data:
        return str(p)

    raise ValueError(
        f"Handbook at {p} has neither model_profiles nor agent_profiles"
    )


# ===========================================================================
# Main pipeline orchestration
# ===========================================================================

def run_pipeline(config: PipelineConfig) -> Dict[str, Any]:
    """Run the full pipeline according to config.phases."""

    if config.run_dir:
        output_dir = Path(config.run_dir).resolve()
        if not output_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {output_dir}")
        config.output_dir = str(output_dir)
        timestamp = output_dir.name
        # Load config from run to get model_config, routing_strategy, etc.
        cfg_path = output_dir / "pipeline_config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                saved = json.load(f)
            # Restore run settings; only restore dataset if user did NOT pass --dataset
            for k in ("model_config", "routing_strategy",
                      "max_rounds", "concurrency", "pool_models"):
                if k in saved and saved[k] is not None:
                    setattr(config, k, saved[k])

            if config.model_config and not Path(config.model_config).exists():
                fallback = str(DEFAULT_MODEL_CONFIG)
                if Path(fallback).exists():
                    logger.info(f"model_config {config.model_config} not found; using {fallback}")
                    config.model_config = fallback

            if not config.dataset_from_cli and "dataset" in saved and saved["dataset"]:
                config.dataset = saved["dataset"]
                logger.info(f"Using dataset from run: {saved['dataset']} (experiment_name={config.experiment_name})")
            elif config.dataset:
                logger.info(f"Using dataset: {config.dataset} (experiment_name={config.experiment_name})")
    else:
        base_output_dir = Path(config.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        config.output_dir = str(output_dir)

    _repo_root = Path(__file__).resolve().parent.parent
    # Resolve relative paths to absolute
    if config.model_config:
        mp = Path(config.model_config)
        if not mp.is_absolute():
            cand = (_repo_root / mp).resolve()
            if cand.is_file():
                config.model_config = str(cand)

    # If resuming (e.g. --phases test), copy learned/selected from latest run
    base_output_dir = output_dir.parent
    latest_link = base_output_dir / "latest"
    if not config.run_dir and latest_link.exists() and latest_link.resolve() != output_dir.resolve():
        prev = latest_link.resolve()
        for subdir in ["model-routing_*"]:
            import glob as _glob
            for src in _glob.glob(str(prev / subdir)):
                dst = output_dir / Path(src).name
                if not dst.exists():
                    import shutil
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    logger.info(f"Carried forward {Path(src).name} from previous run")

    _setup_log_file(output_dir)

    store = HandbookStore(output_dir)
    results: Dict[str, Any] = {
        "task_type": config.task_type,
        "phases": config.phases,
        "started_at": datetime.now().isoformat(),
        "run_id": timestamp,
    }

    _save_config(config, output_dir / "pipeline_config.json")

    needs_bundles = ("explore" in config.phases
                     or "learn" in config.phases
                     or "select" in config.phases)

    train_bundles: List[ExplorationBundle] = []
    val_bundles: List[ExplorationBundle] = []

    if needs_bundles:
        # -- Phase 0: Exploration --
        exploration_path: Optional[Path] = None
        if "explore" in config.phases:
            exploration_path = phase_explore_model_routing(config)
            results["exploration_path"] = str(exploration_path)
        elif config.exploration_data:
            exploration_path = Path(config.exploration_data)
        else:
            existing = find_rsl_results(RSL_RESULTS_DIR, config.dataset)
            if existing:
                exploration_path = existing

        bundles = _load_bundles(config, exploration_path)
        if not bundles:
            raise RuntimeError(
                "No exploration data found. Run with --phases explore or "
                "provide --exploration-data"
            )

        logger.info(f"Loaded {len(bundles)} exploration bundles")
        results["num_bundles"] = len(bundles)

        if config.train_samples is not None or config.val_samples is not None:
            # Explicit split: first train_samples for learning, next val_samples for selection
            if config.train_samples is not None and config.val_samples is not None:
                train_bundles = bundles[: config.train_samples]
                val_bundles = bundles[config.train_samples : config.train_samples + config.val_samples]
            elif config.train_samples is not None:
                train_bundles = bundles[: config.train_samples]
                val_bundles = bundles[config.train_samples:]
            else:
                n = config.val_samples or 0
                val_bundles = bundles[-n:] if n > 0 else []
                train_bundles = bundles[:-n] if n > 0 else bundles
            logger.info(
                f"Split (explicit): {len(train_bundles)} train, {len(val_bundles)} validation"
            )
        else:
            n_val_ratio = int(len(bundles) * config.val_ratio)
            n_val = max(config.min_val_samples, n_val_ratio, 1)
            n_val = min(n_val, len(bundles) - 1)  # keep at least 1 for train
            if config.max_val_samples is not None:
                n_val = min(n_val, config.max_val_samples)
            train_bundles = bundles[: len(bundles) - n_val]
            val_bundles = bundles[len(bundles) - n_val :]
            logger.info(
                f"Split: {len(train_bundles)} train, {len(val_bundles)} validation "
                f"(min_val={config.min_val_samples})"
            )

        oracle_train = sum(b.oracle_accuracy for b in train_bundles) / len(train_bundles) if train_bundles else 0
        oracle_val = sum(b.oracle_accuracy for b in val_bundles) / len(val_bundles) if val_bundles else 0
        logger.info(f"Oracle accuracy: train={oracle_train:.1%}, val={oracle_val:.1%}")
        results["oracle_train"] = oracle_train
        results["oracle_val"] = oracle_val

        # Save raw exploration data and per-query training details
        _save_exploration_artifacts(
            store, config.experiment_name, exploration_path,
            train_bundles, val_bundles,
        )

    # -- Phase 1: Learning --
    # Skip loading when only testing with --handbook (use passed handbook instead)
    test_only_with_handbook = (
        "test" in config.phases
        and config.handbook_path
        and "learn" not in config.phases
        and "select" not in config.phases
    )
    handbook: Optional[SkillHandbook] = None
    selected: Optional[CandidateHandbook] = None

    # Skip learning/selection/testing when only exploration is requested
    _skip_post_explore = (
        "learn" not in config.phases
        and "select" not in config.phases
        and "test" not in config.phases
    )
    if _skip_post_explore:
        logger.info("Skipping learn/select/test — explore-only run")
        return results
    if not test_only_with_handbook:
        if "learn" in config.phases:
            handbook = phase_learn(config, train_bundles + val_bundles, store)
            results["skills_learned"] = handbook.num_skills
            results["agents_profiled"] = len(handbook.agent_profiles)
        else:
            try:
                handbook = store.load_learned(config.experiment_name)
                logger.info(f"Loaded existing handbook: {handbook.num_skills} skills")
            except FileNotFoundError:
                if "learn" not in config.phases:
                    raise RuntimeError(
                        f"No learned handbook found for experiment '{config.experiment_name}'. "
                        "Run with --phases learn (or learn,select) to create one."
                    )
                if needs_bundles:
                    logger.warning("No learned handbook found. Running manual learning.")
                    handbook = phase_learn(config, train_bundles + val_bundles, store)
                else:
                    raise RuntimeError(
                        "No learned handbook found. Run with --phases learn first."
                    )

        # -- Phase 2: Selection --
        if "select" in config.phases:
            selected = phase_select(config, handbook, val_bundles, store)
            results["selected_candidate"] = selected.name
        else:
            try:
                selected_hb = store.load_selected("default", config.experiment_name)
                selected = CandidateHandbook(
                    name="loaded_selected", handbook=selected_hb, granularity="unknown",
                )
            except FileNotFoundError:
                logger.info("No selected handbook, using full learned handbook")
                selected = CandidateHandbook(
                    name="full", handbook=handbook.subgraph(), granularity="full",
                )

    # -- Phase 3: Testing --
    if "test" in config.phases:
        test_handbook = selected.handbook
        if config.handbook_path:
            test_handbook = _load_handbook_for_test(config.handbook_path)
            logger.info(f"Using handbook from --handbook: {config.handbook_path}")
        test_results = phase_test_model_routing(config, test_handbook, store)
        results["test"] = test_results

    results["completed_at"] = datetime.now().isoformat()

    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Pipeline summary saved to {summary_path}")

    # Update "latest" symlink
    base_output_dir = output_dir.parent
    latest_link = base_output_dir / "latest"
    latest_link.unlink(missing_ok=True)
    latest_link.symlink_to(output_dir.name)
    logger.info(f"Updated latest -> {output_dir.name}")

    # Copy key results to base_output_dir/results/ for easy access
    _copy_results_to_output(base_output_dir, output_dir, config.experiment_name)

    return results


def _copy_results_to_output(
    base_output_dir: Path,
    run_dir: Path,
    experiment_name: str,
) -> None:
    """Copy learned, selected, handbooks, and summary to base_output_dir/results/."""
    results_dir = base_output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    exp_dir = run_dir / experiment_name

    def _copy_dir(src: Path, dst_name: str) -> None:
        dst = results_dir / dst_name
        if src.exists():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            logger.info(f"Copied {dst_name}/ -> results/{dst_name}/")

    def _copy_file(src: Path, dst_name: str) -> None:
        if src.exists():
            dst = results_dir / dst_name
            shutil.copy2(src, dst)
            logger.info(f"Copied {dst_name} -> results/")

    if exp_dir.exists():
        _copy_dir(exp_dir / "learned", "learned")
        _copy_dir(exp_dir / "selected", "selected")
        _copy_dir(exp_dir / "snapshots", "snapshots")
        _copy_dir(exp_dir / "evaluation", "evaluation")
        if (run_dir / "learning").exists():
            _copy_dir(run_dir / "learning", "learning")
    _copy_file(run_dir / "pipeline_summary.json", "pipeline_summary.json")
    _copy_file(run_dir / "pipeline_config.json", "pipeline_config.json")
    _copy_file(run_dir / "pipeline.log", "pipeline.log")
    logger.info(f"Results available at {results_dir}")


# ===========================================================================
# Helpers
# ===========================================================================

def _save_exploration_artifacts(
    store: HandbookStore,
    experiment_name: str,
    exploration_path: Optional[Path],
    train_bundles: List[ExplorationBundle],
    val_bundles: List[ExplorationBundle],
) -> None:
    """Save per-query exploration results (all model answers + correctness)."""
    eval_dir = store.experiment_dir(experiment_name) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    def _bundle_to_dict(bundle: ExplorationBundle, split: str) -> Dict[str, Any]:
        model_results = {}
        for trace in bundle.trajectories:
            model_results[trace.varied_agent_id] = {
                "mode": trace.varied_mode,
                "success": trace.task_success,
                "answer": trace.final_answer,
                "cost_usd": trace.total_cost_usd,
            }
        successful_models = [m for m, r in model_results.items() if r["success"]]
        return {
            "query_id": bundle.query_id,
            "query": bundle.query,
            "ground_truths": bundle.ground_truths,
            "split": split,
            "oracle_correct": bundle.any_successful,
            "num_models": len(model_results),
            "num_correct": len(successful_models),
            "successful_models": successful_models,
            "model_results": model_results,
        }

    records = []
    for b in train_bundles:
        records.append(_bundle_to_dict(b, "train"))
    for b in val_bundles:
        records.append(_bundle_to_dict(b, "val"))

    path = eval_dir / "exploration_results.jsonl"
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Compute per-model accuracy
    all_bundles = train_bundles + val_bundles
    model_correct: Dict[str, int] = {}
    model_total: Dict[str, int] = {}
    for b in all_bundles:
        for t in b.trajectories:
            model_correct.setdefault(t.varied_agent_id, 0)
            model_total.setdefault(t.varied_agent_id, 0)
            model_total[t.varied_agent_id] += 1
            if t.task_success:
                model_correct[t.varied_agent_id] += 1

    summary = {
        "total_queries": len(records),
        "train_queries": len(train_bundles),
        "val_queries": len(val_bundles),
        "oracle_accuracy_train": sum(1 for b in train_bundles if b.any_successful) / len(train_bundles) if train_bundles else 0,
        "oracle_accuracy_val": sum(1 for b in val_bundles if b.any_successful) / len(val_bundles) if val_bundles else 0,
        "model_accuracies": {
            m: {"correct": model_correct[m], "total": model_total[m],
                "accuracy": round(model_correct[m] / model_total[m], 4)}
            for m in sorted(model_total.keys())
        },
    }
    with open(eval_dir / "exploration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Saved exploration results: {len(records)} queries "
        f"({len(train_bundles)} train, {len(val_bundles)} val) to {path}"
    )


def _load_bundles(
    config: PipelineConfig,
    exploration_path: Optional[Path],
) -> List[ExplorationBundle]:
    """Load exploration data as ExplorationBundles."""
    if exploration_path is None:
        return []

    exploration_path = Path(exploration_path)

    if exploration_path.is_file() and exploration_path.suffix == ".jsonl":
        bundles = load_rsl_results(exploration_path, config.max_train_samples)
        stats = rsl_stats(bundles)
        logger.info(f"RSL stats: {json.dumps(stats, indent=2)}")
        return bundles
    existing = find_rsl_results(exploration_path, config.dataset)
    if existing:
        return load_rsl_results(existing, config.max_train_samples)
    return []


def _find_latest_file(directory: Path, filename: str) -> Optional[Path]:
    """Recursively find the most recent file with the given name."""
    candidates = list(directory.rglob(filename))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_and_load_json(directory: Path, filename: str) -> Optional[Dict]:
    """Find and load a JSON file."""
    path = _find_latest_file(directory, filename)
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_config(config: PipelineConfig, path: Path) -> None:
    """Save pipeline configuration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in config.__dict__.items()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full SkillOrchestra Pipeline: explore → learn → select → test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="task_type", required=True)

    # -- Model Routing --
    mr = subparsers.add_parser(
        "model-routing",
        help="QA benchmark model routing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(mr)
    mr.add_argument("--dataset", type=str, required=True,
                     help="Dataset name (e.g., nq_validation_qwen, triviaqa_test_qwen)")
    mr.add_argument("--test-dataset", type=str, default=None,
                     help="Test dataset (default: inferred from training dataset)")
    mr.add_argument("--router-model", type=str, default="qwen2.5-3b-instruct",
                     help="Router model for testing")
    mr.add_argument("--pool-models", type=str, default=None,
                     help="Comma-separated pool models")
    mr.add_argument("--distributed-config", type=str, default=None,
                     help="Path to JSON with per-model host config")
    mr.add_argument("--routing-strategy", type=str, default="weighted_avg",
                     choices=["router_decides", "weighted_avg", "analyze_model_decide",
                              "weakest_skill", "strongest_skill"],
                     help="Routing strategy (default: weighted_avg, matches test_skill_routing)")

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to both subcommands."""
    parser.add_argument("--output-dir", type=str, default="",
                         help="Output directory (default: output/<dataset_name>)")
    parser.add_argument("--phases", type=str, default="learn,select",
                         help="Comma-separated phases to run: explore,learn,select,test")
    parser.add_argument("--exploration-data", type=str, default=None,
                         help="Path to existing exploration data (skip explore phase)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                         help="Max training samples to use")
    parser.add_argument("--exploration-samples", type=int, default=None,
                         help="Max samples for exploration phase")
    parser.add_argument("--exploration-epochs", type=int, default=1,
                         help="Epochs for exploration")

    parser.add_argument("--no-llm", action="store_true",
                         help="Skip LLM calls, use manual skills for testing")
    parser.add_argument("--llm-model", type=str, default="deepseek/deepseek-v4-pro",
                         help="LLM model for learning")
    parser.add_argument("--skill-id-model", type=str, default=None,
                         help="LLM for skill identification (default: same as --llm-model). Use gpt-5 for best competence estimates.")
    parser.add_argument("--val-ratio", type=float, default=0.3,
                         help="Validation ratio for train/val split")
    parser.add_argument("--train-samples", type=int, default=None,
                        help="Use first N samples for learning (e.g. 20). With --val-samples, explicit split.")
    parser.add_argument("--val-samples", type=int, default=None,
                        help="Use next N samples for selection (e.g. 20). With --train-samples, uses samples after train.")
    parser.add_argument("--min-val-samples", type=int, default=20,
                         help="Minimum validation samples for handbook selection")
    parser.add_argument("--max-val-samples", type=int, default=None,
                         help="Cap validation size (e.g. 20 for consistent comparison)")
    parser.add_argument("--validation-samples", type=str, default=None,
                         help="Path to validation_samples.jsonl for exact same validation set")
    parser.add_argument("--max-refinement-rounds", type=int, default=3)
    parser.add_argument("--max-merge-credits", type=int, default=50,
                        help="Max merge candidates to apply per refinement phase")
    parser.add_argument("--max-split-credits", type=int, default=1,
                        help="Max split candidates to apply per refinement phase")
    parser.add_argument("--max-discovery-bundles", type=int, default=None,
                         help="Max bundles for skill discovery")

    parser.add_argument("--lambda-cost", type=float, default=0.0,
                         help="Cost penalty for Pareto selection")
    parser.add_argument("--candidates", type=str, default=None,
                         help="Comma-separated candidate names for selection (e.g. answer0_code0_search0,answer0_code2_search2). Debug only.")
    parser.add_argument("--test-max-samples", type=int, default=None,
                         help="Max samples for test phase")
    parser.add_argument("--run-dir", type=str, default=None,
                         help="Use existing run dir (e.g. output/nq/20260316_061846) instead of creating new")
    parser.add_argument("--use-existing-candidates", action="store_true",
                         help="Load candidates handbooks from store instead of regenerating (for selection-only runs)")
    parser.add_argument("-v", "--verbose", action="store_true")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default output: output/<dataset_name> (e.g. output/nq_validation_qwen)
    ds = args.dataset or ""
    if "/" in ds:
        ds = Path(ds).stem
    ds = ds.replace("/", "__").replace(".", "__") or "default"
    default_output = str(DEFAULT_OUTPUT_DIR / ds)
    output_dir = args.output_dir or default_output
    run_dir = getattr(args, "run_dir", None)
    dataset_from_cli = "--dataset" in sys.argv
    config = PipelineConfig(
        task_type=args.task_type,
        output_dir=output_dir,
        run_dir=run_dir,
        dataset_from_cli=dataset_from_cli,
        use_existing_candidates=getattr(args, "use_existing_candidates", False),
        phases=[p.strip() for p in args.phases.split(",")],
        dataset=args.dataset,
        exploration_data=args.exploration_data,
        max_train_samples=args.max_train_samples,
        exploration_epochs=args.exploration_epochs,
        exploration_samples=args.exploration_samples,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
        skill_id_model=args.skill_id_model,
        val_ratio=args.val_ratio,
        min_val_samples=args.min_val_samples,
        max_val_samples=args.max_val_samples,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        validation_samples=args.validation_samples,
        max_refinement_rounds=args.max_refinement_rounds,
        max_merge_credits=args.max_merge_credits,
        max_split_credits=args.max_split_credits,
        max_discovery_bundles=args.max_discovery_bundles,
        lambda_cost=args.lambda_cost,
        selection_candidates=(
            [s.strip() for s in args.candidates.split(",")] if getattr(args, "candidates", None) else None
        ),
        test_max_samples=args.test_max_samples,
    )

    config.test_dataset = args.test_dataset
    config.router_model = args.router_model
    config.distributed_config = args.distributed_config
    config.routing_strategy = args.routing_strategy
    if args.pool_models:
        config.pool_models = args.pool_models.split(",")

    # Default: first 30 samples for exploration when running explore phase
    if "explore" in config.phases and config.exploration_samples is None:
        config.exploration_samples = 30

    logger.info("=" * 70)
    logger.info("SkillOrchestra Full Pipeline")
    logger.info(f"  Task:    {config.task_type}")
    logger.info(f"  Dataset: {config.dataset}")
    logger.info(f"  Phases:  {config.phases}")
    logger.info(f"  Output:  {config.output_dir}")
    logger.info(f"  LLM:     {'enabled (' + config.llm_model + ')' if config.use_llm else 'disabled (manual)'}")
    logger.info("=" * 70)

    try:
        results = run_pipeline(config)
        logger.info("Pipeline completed successfully!")
        return results
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
