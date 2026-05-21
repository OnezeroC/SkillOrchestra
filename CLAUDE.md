# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SkillOrchestra is a framework for skill-aware model routing. Instead of learning routing policies end-to-end, it learns fine-grained **skills** from execution experience and models model-specific competence and cost under those skills. At deployment, the router infers skill demands of each query and selects the best pool model under a performance-cost trade-off.

## Environment & Setup

Two conda environments are used:

```bash
./scripts/setup/run.sh                        # Install all environments
./scripts/setup/env.sh --pipeline              # so_env: pipeline + exploration + learning
./scripts/setup/env.sh --sglang                # sglang_env: model serving via SGLang
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (or `OPENAI_GATEWAY_KEY`), and optionally `HF_HOME`.

## Serving Models

Models must be running before exploration/selection/testing:

```bash
conda activate sglang_env
./scripts/serve/serve_routing.sh        # Model routing pool
```

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
- **`selection/`** — Candidate handbook generation at varying skill granularity (`candidates.py`), Pareto-optimal selection from evaluation results (`pareto.py`), and versioned storage (`store.py`)
- **`routing/`** — Deployment-time router (`pool_service.py`), and API provider abstraction (`api_provider.py`)
- **`converters/`** — Format conversion between SkillHandbook and downstream formats (RSL for model routing)
- **`llm/`** — LLM client with multi-provider support (OpenAI, Salesforce gateway, custom OpenAI-compatible), structured output parsing, retry logic
- **`prompts/`** — Prompt templates for skill discovery, model routing analysis, and evaluation
- **`eval/`** — Exact match and F1 scoring utilities

### Key Data Flow

```
ExecutionTrace (per-query, per-model)
    → ExplorationBundle (all models for one query)
        → SkillHandbook (skills + agent profiles + mode metadata)
            → CandidateHandbook (subgraphs at different granularity)
                → Selected handbook → RSL JSON → test
```

### Data Types (from `core/types.py`)

- **`Skill`** — A reusable capability with skill_id, description, indicators (keywords for matching), examples, and provenance tracking
- **`BetaCompetence`** — Bayesian Beta(alpha,beta) distribution for model competence on a skill; `alpha-1` = successes, `beta-1` = failures
- **`AgentProfile`** — Per-model record with skill_competence map, cost stats, strengths/weaknesses
- **`SkillHandbook`** — The central artifact: mode registry, skill registry, agent profiles, mode-skill index. Provides `select_agent()` with hierarchical tie-breaking (skill -> category -> mode -> cost)

### Configuration Files

- **`config/pool_config.json`** — Model pool: model paths, ports, pricing, host env vars. Edit to add/remove models.
- **`config/api_models.json`** — API model registry: providers (base URL, endpoint), models (identifier, provider, model name), and pricing. Edit to add API-based models.
- **`config/pipeline.py`** — Default paths, model lists, pool configurations.
- **`.env`** — API keys, cache paths (from `.env.example`).

### Model Routing-specific (`model_routing/`)

- `explore.py` — Run all pool models on a dataset, collect answers + correctness
- `test_skill_routing.py` — Skill-based routing inference: LLM router identifies active skills -> weighted competence scoring -> agent selection -> pool model call -> EM evaluation
- `load_qa_dataset.py` — Dataset loading from HuggingFace

## Model Calling Logic

The router uses **API first, then SGLang fallback**:

```
test_skill_routing.py (per question)
  |- call_router()
  |    |- 1st: local SGLang (health check -> /generate)
  |    |- fallback: API (config/api_models.json)
  |
  |- parse_skill_analysis() -> route_by_weighted_avg()
  |
  |- call_pool_model_unified()
       |- 1st: API (APIPoolProvider -> config/api_models.json)
       |- fallback: local SGLang (config/pool_config.json)
```

To add a new API model, edit `config/api_models.json` (`providers` + `models` + `pricing`) and set the API key in `.env`. No Python code changes needed.

## Learning Pipeline Details

The learning phase (`learning/pipeline.py`) runs:

1. **Skill Discovery** (`discoverer.py`): LLM generates a hierarchical taxonomy of skills from contrastive execution pairs (successful vs. failed traces per mode). Uses progressive bundle reduction if context length is exceeded.
2. **Agent Profiling** (`profiler.py`): For each agent, identify which skills are demonstrated in each trace and update BetaCompetence estimates. Distills mode-level routing insights from patterns.
3. **Refinement** (`refiner.py`): Data-driven split/merge of skills — split high-variance skills into finer sub-skills, merge skills with indistinguishable performance profiles.

Candidates are generated by varying per-mode skill depth (cross product of depth levels), then evaluated via live routing on validation samples. The Pareto frontier maximizes accuracy while minimizing cost.
