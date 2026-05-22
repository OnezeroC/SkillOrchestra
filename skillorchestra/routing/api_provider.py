"""
API-based model pool provider for SkillOrchestra.

This module replaces the local SGLang model serving (direct HTTP to localhost:port)
with external API calls via OpenAI-compatible endpoints (OpenRouter, NVIDIA NIM, etc.).

Migrated from: Router-R1/router_r1/llm_agent/route_service.py

Key design:
  - Unified model registry in config/api_models.json (single source of truth)
  - Multi-provider: one APIPoolProvider manages NVIDIA, OpenRouter, MiniMax, etc.
  - Each model specifies its provider; clients are lazy-created per provider
  - Legacy single-provider mode preserved for backward compatibility
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Re-use PoolCallResult and PoolModelCost from pool_service for compatibility
# ---------------------------------------------------------------------------

from skillorchestra.routing.pool_service import PoolCallResult, PoolModelCost

# ---------------------------------------------------------------------------
# Default config path
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "api_models.json"

# ---------------------------------------------------------------------------
# Legacy model maps (fallback when config file is absent)
# ---------------------------------------------------------------------------

OPENROUTER_MODEL_MAP: Dict[str, Tuple[str, float, int, float]] = {
    "qwen2.5-7b-instruct":       ("qwen/qwen2.5-7b-instruct", 1.0, 512, 1.0),
    "qwen2.5-3b-instruct":       ("qwen/qwen2.5-3b-instruct", 1.0, 512, 1.0),
    "llama3.1-8b-instruct":      ("meta-llama/llama-3.1-8b-instruct", 1.0, 512, 1.0),
    "llama3.1-70b-instruct":     ("meta-llama/llama-3.1-70b-instruct", 1.0, 512, 1.0),
    "mistral-7b-instruct":       ("mistralai/mistral-7b-instruct-v0.3", 1.0, 512, 1.0),
    "mixtral-8x22b-instruct":    ("mistralai/mixtral-8x22b-instruct-v0.1", 1.0, 512, 1.0),
    "gemma2-27b-it":             ("google/gemma-2-27b-it", 1.0, 512, 1.0),
    "deepseek-v4-pro":           ("deepseek/deepseek-v4-pro", 0.6, 8192, 1.0),
}

NVIDIA_MODEL_MAP: Dict[str, Tuple[str, float, int, float]] = {
    "qwen2.5-3b-instruct":       ("meta/llama-3.1-8b-instruct", 1.0, 512, 1.0),
    "qwen3-next-80b-a3b-instruct": ("qwen/qwen3-next-80b-a3b-instruct", 1.0, 512, 1.0),
    "llama3.1-8b-instruct":      ("meta/llama-3.1-8b-instruct", 1.0, 512, 1.0),
    "mistral-7b-instruct":       ("mistralai/mistral-nemotron", 1.0, 512, 1.0),
    "mixtral-8x22b-instruct":    ("mistralai/mixtral-8x22b-instruct-v0.1", 1.0, 512, 1.0),
    "gemma2-27b-it":             ("mistralai/mistral-small-4-119b-2603", 1.0, 512, 1.0),
}

API_PRICE_1M_TOKENS_OUTPUT: Dict[str, float] = {
    "qwen/qwen2.5-7b-instruct": 0.30,
    "qwen/qwen2.5-3b-instruct": 0.13,
    "qwen/qwen3.5-122b-a10b": 0.35,
    "qwen/qwen3-next-80b-a3b-instruct": 0.30,
    "qwen/qwen2.5-coder-32b-instruct": 0.86,
    "meta-llama/llama-3.1-8b-instruct": 0.18,
    "meta-llama/llama-3.1-70b-instruct": 0.88,
    "mistralai/mistral-7b-instruct-v0.3": 0.20,
    "mistralai/mistral-nemotron": 0.35,
    "mistralai/mistral-small-4-119b-2603": 0.30,
    "mistralai/mixtral-8x22b-instruct-v0.1": 1.20,
    "google/gemma-2-27b-it": 0.80,
    "google/gemma-3-27b-it": 0.80,
    "deepseek/deepseek-v4-pro": 2.00,
    "meta/llama-3.1-8b-instruct": 0.18,
    "meta/llama-3.1-70b-instruct": 0.88,
    "nvidia/llama3-chatqa-1.5-8b": 0.18,
    "mistralai/mixtral-8x22b-instruct-v0.1": 1.20,
    "writer/palmyra-creative-122b": 1.80,
}

# Prompt templates
API_PROMPT_TEMPLATE = """\
You are a helpful assistant. \
You are participating in a multi-agent reasoning process, where a base model delegates sub-questions to specialized models like you. \

Your task is to do your **absolute best** to either: \n
    + Answer the question directly, if possible, and provide a brief explanation; or \n
    + Offer helpful and relevant context, background knowledge, or insights related to the question, even if you cannot fully answer it. \

If you are completely unable to answer the question or provide any relevant or helpful information, you must: \n
    + Clearly state that you are unable to assist with this question, and \n
    + Explicitly instruct the base model to consult other LLMs for further assistance. \

**Important Constraints**: \n
    + Keep your response clear, concise, and informative (preferably under 512 tokens). Your response will help guide the base model's reasoning and next steps. \\n
    + Stay strictly on-topic. Do not include irrelevant or generic content. \

\n\nHere is the sub-question for you to assist with: {query}\n
"""

SIMPLE_PROMPT_TEMPLATE = "You are a helpful AI assistant. Please answer the question below:\n\nQuestion: {query}"


# ---------------------------------------------------------------------------
# APIPoolProvider
# ---------------------------------------------------------------------------

class APIPoolProvider:
    """API-based model pool provider with multi-provider support.

    Two modes:
    1. Config mode (recommended): reads config/api_models.json for all model/provider
       definitions. Each model specifies its provider; clients are lazy-created per provider.
    2. Legacy mode: single base_url/api_key/model_map, for backward compatibility.

    Config mode usage:
        provider = APIPoolProvider()  # auto-loads config/api_models.json

    Legacy mode usage:
        provider = APIPoolProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-...",
            model_map="openrouter",
        )
    """

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        *,
        config_path: Optional[str] = None,
        model_map: str | Dict[str, Tuple[str, float, int, float]] = "openrouter",
        max_retries: int = 3,
        timeout: int = 180,
        max_trials: int = 8,
        time_gap: float = 10.0,
        prompt_template: str = SIMPLE_PROMPT_TEMPLATE,
    ):
        if not _OPENAI_AVAILABLE:
            raise ImportError("openai library required: pip install openai")

        self.max_retries = max_retries
        self.timeout = timeout
        self.max_trials = max_trials
        self.time_gap = time_gap
        self.prompt_template = prompt_template

        # Determine mode
        self._legacy_mode = bool(base_url)
        self._legacy_base_url = ""
        self._legacy_api_key = ""
        self._legacy_model_map: Dict[str, Tuple[str, float, int, float]] = {}
        self._legacy_client: Optional[OpenAI] = None

        # Config-mode state
        self._providers: Dict[str, Dict[str, str]] = {}   # name → {base_url, api_key}
        self._models: Dict[str, Dict[str, Any]] = {}       # model_key → {provider, model_name, ...}
        self._pricing: Dict[str, Dict[str, float]] = {}    # api_model_name → {input, output}
        self._clients: Dict[str, OpenAI] = {}               # provider_name → client

        if self._legacy_mode:
            self._init_legacy(base_url, api_key, model_map)
        else:
            self._init_from_config(config_path)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_legacy(
        self,
        base_url: str,
        api_key: str,
        model_map: str | Dict[str, Tuple[str, float, int, float]],
    ) -> None:
        """Legacy single-provider setup (backward compatible)."""
        self._legacy_base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", os.environ.get("base_url", "")
        )
        self._legacy_api_key = api_key or os.environ.get(
            "OPENAI_API_KEY", os.environ.get("api_key", "")
        )

        if not self._legacy_base_url:
            raise ValueError(
                "base_url is required. Set it via parameter, "
                "OPENAI_BASE_URL env var, or base_url env var."
            )
        if not self._legacy_api_key:
            raise ValueError(
                "api_key is required. Set it via parameter, "
                "OPENAI_API_KEY env var, or api_key env var."
            )

        if isinstance(model_map, str):
            if model_map == "openrouter":
                self._legacy_model_map = OPENROUTER_MODEL_MAP
            elif model_map == "nvidia":
                self._legacy_model_map = NVIDIA_MODEL_MAP
            elif model_map == "minimax":
                self._legacy_model_map = {}
            else:
                raise ValueError(f"Unknown model_map preset: {model_map}")
        else:
            self._legacy_model_map = model_map

    def _init_from_config(self, config_path: Optional[str] = None) -> None:
        """Load model registry from config/api_models.json."""
        path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

        if not path.exists():
            logger.warning(
                "Config file %s not found, falling back to legacy env vars. "
                "Create config/api_models.json for multi-provider support.",
                path,
            )
            # Fall back to legacy mode with env vars
            self._legacy_mode = True
            self._init_legacy("", "", "openrouter")
            return

        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load providers
        for name, pconf in config.get("providers", {}).items():
            base_url = os.environ.get(f"{name.upper()}_BASE_URL", pconf["base_url"])
            api_key_env = pconf.get("api_key_env", f"{name.upper()}_API_KEY")
            api_key = os.environ.get(api_key_env, "")
            if not api_key:
                # Backward compat: fall back to OPENAI_API_KEY / api_key
                api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("api_key", ""))
            if not api_key:
                logger.warning("No API key found for provider '%s' (env: %s)", name, api_key_env)
            self._providers[name] = {"base_url": base_url, "api_key": api_key}

        # Load models
        for mkey, mconf in config.get("models", {}).items():
            provider = mconf.get("provider", "")
            if provider not in self._providers:
                logger.warning("Model '%s' references unknown provider '%s'", mkey, provider)
            self._models[mkey] = {
                "provider": provider,
                "model_name": mconf["model_name"],
                "temperature": mconf.get("temperature", 1.0),
                "max_tokens": mconf.get("max_tokens", 512),
                "top_p": mconf.get("top_p", 1.0),
            }

        # Load pricing
        self._pricing = config.get("pricing", {})

        logger.info(
            "Loaded API model registry: %d providers, %d models from %s",
            len(self._providers), len(self._models), path,
        )

    # ------------------------------------------------------------------
    # Client management
    # ------------------------------------------------------------------

    def _get_client(self, provider_name: str) -> OpenAI:
        """Lazy-create and cache OpenAI client for a provider."""
        if provider_name not in self._clients:
            p = self._providers.get(provider_name)
            if p is None:
                raise ValueError(
                    f"Unknown provider '{provider_name}'. "
                    f"Available: {list(self._providers.keys())}"
                )
            if not p["api_key"]:
                raise ValueError(
                    f"No API key for provider '{provider_name}'. "
                    f"Set the required environment variable."
                )
            self._clients[provider_name] = OpenAI(
                base_url=p["base_url"],
                api_key=p["api_key"],
                max_retries=self.max_retries,
                timeout=self.timeout,
            )
        return self._clients[provider_name]

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_model(self, model_key: str) -> Tuple[str, str, float, int, float]:
        """Resolve a SkillOrchestra model key to API parameters.

        Config mode returns:  (provider_name, api_model_name, temperature, max_tokens, top_p)
        Legacy mode returns:  ("", api_model_name, temperature, max_tokens, top_p)
        """
        # Config mode: look up in loaded model registry
        if not self._legacy_mode:
            m = self._models.get(model_key)
            if m is not None:
                return (
                    m["provider"],
                    m["model_name"],
                    m["temperature"],
                    m["max_tokens"],
                    m["top_p"],
                )
            # Fallback for unknown keys: try prefix matching for known providers
            key_lower = model_key.lower()
            for prefix, api_prefix in [
                ("qwen", "qwen/"),
                ("llama", "meta-llama/"),
                ("mistral", "mistralai/"),
                ("mixtral", "mistralai/"),
                ("gemma", "google/"),
                ("minimax", ""),
            ]:
                if key_lower.startswith(prefix):
                    provider = "nvidia" if "nvidia" in str(self._providers) else "openrouter"
                    return (provider, f"{api_prefix}{model_key}", 1.0, 512, 1.0)
            # Last resort
            provider = next(iter(self._providers), "") if self._providers else ""
            logger.warning("No mapping found for %s, using as-is with provider '%s'", model_key, provider)
            return (provider, model_key, 1.0, 512, 1.0)

        # Legacy mode
        if model_key in self._legacy_model_map:
            api_name, tau, mt, tp = self._legacy_model_map[model_key]
            return ("", api_name, tau, mt, tp)

        # Legacy fallback
        key_lower = model_key.lower()
        for prefix, api_prefix in [
            ("qwen", "qwen/"),
            ("llama", "meta-llama/"),
            ("mistral", "mistralai/"),
            ("mixtral", "mistralai/"),
            ("gemma", "google/"),
        ]:
            if key_lower.startswith(prefix):
                return ("", f"{api_prefix}{model_key}", 1.0, 512, 1.0)

        logger.warning(f"No mapping found for {model_key}, using as-is")
        return ("", model_key, 1.0, 512, 1.0)

    # ------------------------------------------------------------------
    # Single model call
    # ------------------------------------------------------------------

    def call_pool_model(
        self,
        model_key: str,
        query: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = 42,
        prompt_template: Optional[str] = None,
    ) -> PoolCallResult:
        """Call a single pool model via external API.

        Args:
            model_key: SkillOrchestra model key (e.g. "llama3.1-8b-instruct")
            query: The question/prompt text
            max_tokens: Override default max_tokens
            temperature: Override default temperature
            top_p: Override default top_p
            seed: Random seed (default 42)
            prompt_template: Override the prompt template

        Returns:
            PoolCallResult with response text and cost info
        """
        provider_name, api_model, default_tau, default_max_tokens, default_top_p = \
            self._resolve_model(model_key)

        temperature = temperature if temperature is not None else default_tau
        max_tokens = max_tokens if max_tokens is not None else default_max_tokens
        top_p = top_p if top_p is not None else default_top_p
        template = prompt_template or self.prompt_template

        if "{query}" in template:
            user_content = template.format(query=query)
        else:
            user_content = query

        # Get the right client
        if self._legacy_mode:
            client = self._legacy_client
            if client is None:
                self._legacy_client = OpenAI(
                    base_url=self._legacy_base_url,
                    api_key=self._legacy_api_key,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
                client = self._legacy_client
        else:
            client = self._get_client(provider_name)

        trials = self.max_trials
        attempt = 0
        completion = None

        while trials > 0:
            trials -= 1
            attempt += 1
            try:
                completion = client.chat.completions.create(
                    model=api_model,
                    messages=[{"role": "user", "content": user_content}],
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    max_tokens=max_tokens,
                )
                break
            except Exception as e:
                err_msg = str(e).strip()
                logger.warning(
                    f"[{model_key}] API error attempt {attempt}/{self.max_trials}: {err_msg[:200]}"
                )
                if trials == 0:
                    break
                # Exponential backoff: time_gap * 2^(attempt-1), capped at 120s
                delay = min(self.time_gap * (2 ** (attempt - 1)), 120.0)
                logger.info(f"[{model_key}] retrying in {delay:.0f}s ({trials} trials left)")
                time.sleep(delay)

        if completion is None or completion.choices is None:
            return PoolCallResult(
                model_key=model_key,
                response="API Request Error",
                cost=PoolModelCost(model_key=model_key),
                success=False,
                error=f"max_trials_exceeded after {self.max_trials} attempts",
            )

        response_text = completion.choices[0].message.content or ""
        usage = completion.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        # Calculate cost (prefer config pricing, fall back to legacy dict)
        if not self._legacy_mode and api_model in self._pricing:
            input_price = self._pricing[api_model].get("input", 0.30)
            output_price = self._pricing[api_model].get("output", 0.30)
        else:
            output_price = API_PRICE_1M_TOKENS_OUTPUT.get(api_model, 0.30)
            input_price = output_price / 8.0  # conservative estimate

        cost = PoolModelCost(
            model_key=model_key,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_cost=prompt_tokens * input_price / 1_000_000,
            output_cost=completion_tokens * output_price / 1_000_000,
        )

        return PoolCallResult(
            model_key=model_key,
            response=response_text,
            cost=cost,
            success=True,
        )

    # ------------------------------------------------------------------
    # Parallel multi-model calls
    # ------------------------------------------------------------------

    def call_pool_models_parallel(
        self,
        model_keys: List[str],
        query: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = 42,
        max_workers: int = 15,
        prompt_template: Optional[str] = None,
    ) -> Dict[str, PoolCallResult]:
        """Call multiple pool models in parallel on the same query."""
        results: Dict[str, PoolCallResult] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.call_pool_model,
                    mk, query,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    prompt_template=prompt_template,
                ): mk
                for mk in model_keys
            }
            for fut in as_completed(futures):
                mk = futures[fut]
                try:
                    results[mk] = fut.result()
                except Exception as e:
                    logger.error(f"[{mk}] parallel call failed: {e}")
                    results[mk] = PoolCallResult(
                        model_key=mk,
                        response=f"Error: {e}",
                        cost=PoolModelCost(model_key=mk),
                        success=False,
                        error=str(e),
                    )

        return results

    # ------------------------------------------------------------------
    # Route-R1 compatible access_routing_pool
    # ------------------------------------------------------------------

    def access_routing_pool(
        self,
        queries: List[str],
        *,
        max_workers: int = 4,
    ) -> Dict[str, Any]:
        """Route-R1 compatible access_routing_pool.

        Takes queries in format "model_name:query_text", routes each to the
        appropriate external API model, and returns results.
        """
        task_args = []
        for q_id, single_query in enumerate(queries):
            parts = single_query.split(":", 1)
            if len(parts) != 2:
                logger.warning(f"Invalid query format (missing ':'): {single_query[:50]}")
                target_llm = ""
                query_text = single_query
            else:
                target_llm = parts[0].strip().lower()
                query_text = parts[1]

            _, api_model, tau, _, _ = self._resolve_model(target_llm)
            task_args.append((q_id, query_text, tau, api_model))

        ret = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for q_id, query_text, tau, api_model in task_args:
                futures[
                    executor.submit(
                        self._request_single,
                        q_id, query_text, tau, api_model,
                    )
                ] = q_id

            for fut in as_completed(futures):
                try:
                    ret.append(fut.result())
                except Exception as e:
                    q_id = futures[fut]
                    logger.error(f"[{q_id}] request failed: {e}")
                    ret.append((q_id, "API Request Error", 0.0))

        ret.sort(key=lambda x: x[0])
        resp = []
        completion_tokens_list = []
        for _, response, completion_tokens in ret:
            resp.append(response)
            completion_tokens_list.append(completion_tokens)

        return {"result": resp, "completion_tokens_list": completion_tokens_list}

    def _request_single(
        self,
        q_id: int,
        query_text: str,
        tau: float,
        api_model: str,
    ) -> Tuple[int, str, float]:
        """Single request helper for access_routing_pool."""
        if not api_model:
            return q_id, "LLM Name Error", 0.0

        try:
            result = self.call_pool_model(
                model_key=api_model,
                query=query_text,
                temperature=tau,
                prompt_template=API_PROMPT_TEMPLATE,
            )

            if result.success:
                cost_usd = result.cost.total
                return q_id, result.response, cost_usd
            else:
                return q_id, "API Request Error", 0.0

        except Exception as e:
            logger.error(f"[{q_id}] request failed: {e}")
            return q_id, "API Request Error", 0.0

    # ------------------------------------------------------------------
    # Legacy property (backward compat)
    # ------------------------------------------------------------------

    @property
    def client(self) -> OpenAI:
        """Legacy single-client accessor."""
        if self._legacy_mode:
            if self._legacy_client is None:
                self._legacy_client = OpenAI(
                    base_url=self._legacy_base_url,
                    api_key=self._legacy_api_key,
                    max_retries=self.max_retries,
                    timeout=self.timeout,
                )
            return self._legacy_client
        raise RuntimeError("Use _get_client(provider_name) in config mode; .client is legacy-only")


# ---------------------------------------------------------------------------
# Singleton helper
# ---------------------------------------------------------------------------

_provider_instance: Optional[APIPoolProvider] = None


def get_api_provider(
    base_url: str = "",
    api_key: str = "",
    **kwargs: Any,
) -> APIPoolProvider:
    """Get or create a singleton APIPoolProvider.

    Without arguments, loads from config/api_models.json (multi-provider mode).
    With base_url/api_key, uses legacy single-provider mode.
    """
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = APIPoolProvider(
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
    return _provider_instance


def reset_api_provider() -> None:
    """Reset the singleton provider (for testing)."""
    global _provider_instance
    _provider_instance = None
