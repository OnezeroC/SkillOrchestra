# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkillOrchestra is a framework for skill-aware orchestration of compound AI systems. Instead of learning routing policies end-to-end, it learns fine-grained **skills** from execution experience and models agent-specific competence and cost under those skills. At deployment, the orchestrator infers skill demands of each interaction and selects agents that best satisfy them under a performance-cost trade-off.

Two task types are supported:
- **Model routing** — Route queries to the best pool model (QA benchmarks like NQ, TriviaQA, Math)
- **Agent orchestration (FRAMES)** — Multi-stage agent orchestration with search/code/answer stages

## Environment & Setup

Three conda environments are used:

```bash
./scripts/setup/run.sh                        # Install all environments
./scripts/setup/env.sh --pipeline              # so_env: pipeline + exploration + learning
./scripts/setup/env.sh --sglang                # sglang_env: model serving via SGLang
./scripts/setup/retriever.sh                   # retriever: Qwen3-Embedding + FAISS
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or `OPENAI_GATEWAY_KEY`), `INDEX_DIR` (for retrieval), and optionally `HF_HOME`.

## Serving Models

Models must be running before exploration/selection/testing:

```bash
conda activate sglang_env
./scripts/serve/serve_routing.sh        # Model routing pool
./scripts/serve/serve_orchestration.sh  # Agent orchestration + retriever
```

Edit `config/eval_config.json` to match model host/ports if not using localhost.

## Running the Pipeline

Always activate `so_env` first:

```bash
conda activate so_env
```

### Model Routing

```bash
# Full pipeline
python scripts/pipeline.py model-routing \
    --dataset nq_validation_qwen \
    --output-dir output/nq \
    --test-dataset nq_test_qwen \
    --phases explore,learn,select,test \
    --exploration-samples 40 --train-samples 20 --val-samples 20

# Skip exploration (use existing data)
python scripts/pipeline.py model-routing \
    --dataset nq_validation_qwen \
    --exploration-data /path/to/inference_results.jsonl \
    --output-dir output/nq \
    --phases learn,select,test

# No-LLM mode (manual skills for testing)
python scripts/pipeline.py model-routing ... --no-llm
```

### Agent Orchestration (FRAMES)

```bash
python scripts/pipeline.py frames \
    --output-dir output/frames \
    --eval-script orchestration/eval_frames.py \
    --test-samples data/frames_test.jsonl \
    --phases explore,learn,select,test
```

### Key flags

- `--phases explore,learn,select,test` — run specific phases (comma-separated)
- `--exploration-samples N` — max samples for exploration (default: 30)
- `--train-samples N --val-samples N` — explicit train/val split
- `--llm-model MODEL` — LLM for discovery/refinement (default: `deepseek-ai/deepseek-v4-pro`)
- `--skill-id-model MODEL` — separate LLM for skill identification
- `--lambda-cost FLOAT` — cost penalty for Pareto selection trade-off
- `--routing-strategy` — `weighted_avg`, `router_decides`, `analyze_model_decide`, `weakest_skill`, `strongest_skill`
- `--max-refinement-rounds N` — refinement iterations (default: 3)
- `--run-dir DIR` — resume an existing run
- `--handbook PATH` — use a specific handbook JSON for testing

## Architecture

### Four-Phase Pipeline

```
explore → learn → select → test
```

1. **Explore** — Run all pool models on dataset samples, collect execution traces
2. **Learn** — LLM-powered skill discovery from traces (contrastive pairs) + agent profiling + iterative refinement (split/merge skill definitions)
3. **Select** — Generate candidate handbooks at different skill granularity levels, evaluate each on validation data (live routing), select Pareto-optimal by (accuracy, cost)
4. **Test** — Evaluate the selected handbook on a held-out test set

### Package Structure (`skillorchestra/`)

- **`core/`** — Central data model: `Skill`, `AgentProfile`, `BetaCompetence` (Bayesian competence estimation), `CostStats`, `SkillHandbook` (the full learned artifact with skills, agent profiles, mode metadata, routing logic), and `ExecutionTrace`/`ExplorationBundle` (trace data model)
- **`learning/`** — Learning pipeline orchestration (`pipeline.py`), skill discovery via LLM taxonomy (`discoverer.py`), agent profiling (`profiler.py`), and refinement with split/merge operations (`refiner.py`, `failure_refiner.py`)
- **`selection/`** — Candidate handbook generation at varying skill granularity (`candidates.py`), Pareto-optimal selection from evaluation results (`pareto.py`), live evaluation by running real routing (`live_eval.py`), and versioned storage (`store.py`)
- **`routing/`** — Deployment-time orchestrator (`orchestrator.py`), SGLang pool model service client (`pool_service.py`), and API provider abstraction (`api_provider.py`)
- **`converters/`** — Format conversion between SkillHandbook and downstream formats (RSL for model routing, StageRouter for agent orchestration)
- **`llm/`** — LLM client with multi-provider support (OpenAI, Salesforce gateway, custom OpenAI-compatible), structured output parsing, retry logic
- **`prompts/`** — Prompt templates for skill discovery, model routing analysis, and evaluation
- **`eval/`** — Exact match and F1 scoring utilities
- **`adapters/`** — Adapters for external router formats

### Key Data Flow

```
ExecutionTrace (per-query, per-agent) 
    → ExplorationBundle (all agents for one query)
        → SkillHandbook (skills + agent profiles + mode metadata)
            → CandidateHandbook (subgraphs at different granularity)
                → Selected handbook → RSL/StageRouter JSON → test
```

### Data Types (from `core/types.py`)

- **`Skill`** — A reusable capability with skill_id, description, indicators (keywords for matching), examples, and provenance tracking
- **`BetaCompetence`** — Bayesian Beta(α,β) distribution for agent competence on a skill; `alpha-1` = successes, `beta-1` = failures
- **`AgentProfile`** — Per-agent record with skill_competence map, cost stats, strengths/weaknesses
- **`SkillHandbook`** — The central artifact: mode registry, skill registry, agent profiles, mode-skill index. Provides `select_agent()` with hierarchical tie-breaking (skill → category → mode → cost)

### Configuration Files

- **`config/pool_config.json`** — Model pool: model paths, ports, pricing, host env vars. Edit to add/remove models.
- **`config/eval_config.json`** — Model endpoint IP:port mapping for orchestration (vLLM/SGLang servers)
- **`config/models.py`** — Agent ID → model name resolution
- **`config/pipeline.py`** — Default paths, model lists, pool configurations
- **`.env`** — API keys, cache paths, index directory (from `.env.example`)

### Model Routing-specific (`model_routing/`)

- `explore.py` — Run all pool models on a dataset, collect answers + correctness
- `test_skill_routing.py` — Skill-based routing inference: LLM router identifies active skills → weighted competence scoring → agent selection → pool model call → EM evaluation
- `load_qa_dataset.py` — Dataset loading from HuggingFace

### FRAMES-specific

- `orchestration/eval_frames.py` — Multi-stage agent evaluation (search → code → answer) with configurable models per stage and skill-based routing via `--handbook`
- `orchestration/LLM_CALL.py` — Shared LLM call utilities for orchestration

## Learning Pipeline Details

The learning phase (`learning/pipeline.py`) runs:

1. **Skill Discovery** (`discoverer.py`): LLM generates a hierarchical taxonomy of skills from contrastive execution pairs (successful vs. failed traces per mode). Uses progressive bundle reduction if context length is exceeded.
2. **Agent Profiling** (`profiler.py`): For each agent, identify which skills are demonstrated in each trace and update BetaCompetence estimates. Distills mode-level routing insights from patterns.
3. **Refinement** (`refiner.py`): Data-driven split/merge of skills — split high-variance skills into finer sub-skills, merge skills with indistinguishable performance profiles.

Candidates are generated by varying per-mode skill depth (cross product of depth levels), then evaluated via live routing on validation samples. The Pareto frontier maximizes accuracy while minimizing cost.
