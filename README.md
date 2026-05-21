# SkillOrchestra

This repository contains the code for [SkillOrchestra](https://arxiv.org/abs/2602.19672), a framework for skill-aware model routing. Instead of learning a routing policy end-to-end, SkillOrchestra learns fine-grained skills from execution experience and models model-specific competence and cost under those skills. At deployment, the router infers the skill demands of each query and selects the best pool model under an explicit performance-cost trade-off.

---

## Setup

### 1. Environment

```bash
# All environments (pipeline + SGLang + retriever)
./scripts/setup/run.sh

# Or individually:
./scripts/setup/env.sh --pipeline   # so_env: pipeline, exploration, learning, selection, testing
./scripts/setup/env.sh --sglang     # sglang_env: model serving via SGLang
```

### 2. Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Set at minimum:
- `OPENAI_API_KEY` or `OPENAI_GATEWAY_KEY` — for LLM calls during learning and routing

---

## Model Calling Logic

The router uses a **API first, then SGLang fallback** strategy when calling pool models at inference time:

```
test_skill_routing.py (per question)
  ├─ call_router()
  │   ├─ 1st: local SGLang (health check OK → /generate)
  │   └─ fallback: API (_get_router_api_provider → config/api_models.json)
  │
  ├─ parse_skill_analysis() → route_by_weighted_avg()
  │
  └─ call_pool_model_unified()
        ├─ 1st: API (_detect_backend checks OPENAI_BASE_URL or config/api_models.json
        │         → APIPoolProvider reads config/api_models.json by model_key)
        └─ fallback: local SGLang (config/pool_config.json → MODEL_CONFIGS[model_key])
```

For each pool model, the system first tries the API provider path. If no API configuration is found for that model, it falls back to the local SGLang server.

## Configuration

All configuration is done by editing JSON files — no Python code changes needed.

### API providers and router models (`config/api_models.json`)

Edit this file to configure API-based models:

- **Add an API provider**: Add an entry to `providers` with the provider name, base URL, and endpoint path. Set the API key in `.env` using the `env_var` field.
- **Register a model**: Add an entry to `models` with the model identifier (used as `--router-model`), provider reference, model name string, and pricing.
- **Set pricing**: Add per-model pricing in the `pricing` section (prompt/completion cost per million tokens).

Example: to add a new router model, add to `models` and reference an existing or new `providers` entry. Then pass `--router-model <key>` at runtime.

### Pool models (`config/pool_config.json`)

Edit this file to configure the model pool for local SGLang serving:

- **Add/remove pool models**: Edit the `models` list with model paths, ports, and GPU counts.
- **Register model keys**: Add the model key to `pool_model_keys`.
- **Set pricing**: Add per-model pricing in the `pricing` section.

### Runtime flags

| Flag | Purpose |
|------|---------|
| `--router-model <key>` | Switch the router LLM (key from `config/api_models.json` → `models`) |
| `--pool-models <keys>` | Override which pool models to use (comma-separated) |
| `--lambda-cost FLOAT` | Cost penalty for performance-cost trade-off in routing |
| `--routing-strategy` | Routing behavior: `weighted_avg`, `analyze_model_decide`, `weakest_skill`, `strongest_skill` |

---

## Serving Models

Models must be running before the pipeline can explore, select, or test.

```bash
conda activate sglang_env
./scripts/serve/serve_routing.sh
```

---

## Running the Full Pipeline

Activate the pipeline environment and run:

```bash
conda activate so_env
```

### Model Routing

```bash
# Full pipeline: explore → learn → select → test
python scripts/pipeline.py model-routing \
    --dataset nq_validation_qwen \
    --output-dir output/nq \
    --test-dataset nq_test_qwen \
    --phases explore,learn,select,test \
    --exploration-samples 40 \
    --train-samples 20 \
    --val-samples 20

# Use existing exploration data (skip explore)
python scripts/pipeline.py model-routing \
    --dataset nq_validation_qwen \
    --exploration-data /path/to/inference_results.jsonl \
    --output-dir output/nq \
    --phases learn,select,test \
    --test-dataset nq_test_qwen

# No-LLM mode (manual skills for testing)
python scripts/pipeline.py model-routing ... --no-llm
```

### Phases

- `explore` — Run all pool models on the dataset; collect execution traces
- `learn` — Learn a skill handbook from traces (skills, competence, cost) with refinement
- `select` — Generate candidate handbooks and select Pareto-optimal with live validation
- `test` — Evaluate the selected handbook on the test set

Use `--phases explore`, `--phases learn,select`, etc. to run subsets.

---

## Customization

- **Model pool:** Edit `config/pool_config.json` — add/remove models, set ports, pricing, and host env vars. Use `--pool-models` to override at runtime.
- **API models:** Edit `config/api_models.json` — add providers, models, and pricing. Set API keys in `.env`.
- **Learning:** Use `--llm-model`, `--skill-id-model`, `--max-refinement-rounds`, `--max-merge-credits`, etc. to tune the learning pipeline.
- **Deployment:** Use `--lambda-cost` for performance-cost trade-off; `--routing-strategy` for routing behavior (e.g. `weighted_avg`, `analyze_model_decide`).

---

## License

[Apache 2.0](LICENSE)

---

## Citation

If you find this work helpful, please consider giving a star and citing our paper:

```bibtex
@misc{wang2026skillorchestra,
      title={SkillOrchestra: Learning to Route Agents via Skill Transfer}, 
      author={Jiayu Wang and Yifei Ming and Zixuan Ke and Shafiq Joty and Aws Albarghouthi and Frederic Sala},
      year={2026},
      eprint={2602.19672},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.19672}, 
}
```

---

## Contact

We are here to help! Any questions? Please open an issue and cc [Jiayu Wang](mailto:milawang@cs.wisc.edu) (milawang@cs.wisc.edu).
