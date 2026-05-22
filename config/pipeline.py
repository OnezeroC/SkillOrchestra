"""
Pipeline paths and default model configuration.

Used by scripts/pipeline.py for model-routing pipelines.
"""

from pathlib import Path

_CONFIG_DIR = Path(__file__).resolve().parent
_REPO_DIR = _CONFIG_DIR.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = _REPO_DIR / "data"
CONFIGS_DIR = _REPO_DIR / "config"
DEFAULT_OUTPUT_DIR = _REPO_DIR / "output"

DEFAULT_MODEL_CONFIG = CONFIGS_DIR / "eval_config.json"

# Model routing: RSL results (when using existing exploration)
RSL_RESULTS_DIR = DATA_DIR / "rsl_results"

# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------

# Pool models for model routing
DEFAULT_POOL_MODELS = [
    "qwen2.5-7b-instruct",
    "llama3.1-8b-instruct",
    "llama3.1-70b-instruct",
    "mistral-7b-instruct",
    "mixtral-8x22b-instruct",
    "gemma2-27b-it",
]
