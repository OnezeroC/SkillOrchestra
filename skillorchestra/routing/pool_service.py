"""
SGLang pool model service for SkillOrchestra.

Calls pool models via the OpenAI-compatible /v1/chat/completions endpoint,
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from config.pool import (
    API_PRICE_1M_TOKENS,
    DEFAULT_HOST,
    MODEL_CONFIGS,
    POOL_MODEL_KEYS,
    POOL_PROMPT,
    _DISPLAY_NAME_MAP,
    display_name,
    load_distributed_config,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PoolModelCost:
    """Cost breakdown for a single pool model call."""
    model_key: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0

    @property
    def total(self) -> float:
        return self.input_cost + self.output_cost


@dataclass
class PoolCallResult:
    """Result from calling a pool model."""
    model_key: str
    response: str
    cost: PoolModelCost
    success: bool = True
    error: str = ""


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

def resolve_model_key(name: str) -> str:
    """Map a display name / model key to the canonical model_key."""
    key = name.strip().lower()
    if key in _DISPLAY_NAME_MAP:
        return _DISPLAY_NAME_MAP[key]

    cleaned = key.replace("-instruct", "").replace("_instruct", "")
    if "qwen" in cleaned and "3b" not in cleaned:
        return "qwen2.5-7b-instruct"
    if "llama" in cleaned:
        return "llama3.1-70b-instruct" if "70b" in cleaned else "llama3.1-8b-instruct"
    if "mixtral" in cleaned:
        return "mixtral-8x22b-instruct"
    if "mistral" in cleaned:
        return "mistral-7b-instruct"
    if "gemma" in cleaned:
        return "gemma2-27b-it"
    return ""


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def check_server_health(model_key: str) -> bool:
    if model_key not in MODEL_CONFIGS:
        return False
    cfg = MODEL_CONFIGS[model_key]
    host = cfg.get("ip_addr", DEFAULT_HOST)
    try:
        r = requests.get(f"http://{host}:{cfg['port']}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def check_all_servers() -> Dict[str, bool]:
    return {k: check_server_health(k) for k in MODEL_CONFIGS}


# ---------------------------------------------------------------------------
# Model calling via /v1/chat/completions
# ---------------------------------------------------------------------------

def call_pool_model(
    model_key: str,
    query: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    timeout: int = 300,
    max_retries: int = 5,
    prompt_template: str = POOL_PROMPT,
) -> PoolCallResult:
    """Call a single pool model via /v1/chat/completions.

    The chat-completions endpoint applies the model's chat template
    server-side, avoiding the empty-response bug with raw /generate
    for models like Gemma-2.
    """
    if model_key not in MODEL_CONFIGS:
        return PoolCallResult(model_key=model_key, response="", cost=PoolModelCost(),
                              success=False, error=f"Unknown model: {model_key}")

    cfg = MODEL_CONFIGS[model_key]
    host = cfg.get("ip_addr", DEFAULT_HOST)
    url = f"http://{host}:{cfg['port']}/v1/chat/completions"

    user_content = prompt_template.format(query=query)
    payload: Dict[str, Any] = {
        "model": model_key,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9 if temperature > 0 else 1.0,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})
            pt = usage.get("prompt_tokens", len(user_content) // 4)
            ct = usage.get("completion_tokens", len(text) // 4)

            prices = API_PRICE_1M_TOKENS.get(model_key, {"input": 0, "output": 0})
            cost = PoolModelCost(
                model_key=model_key,
                prompt_tokens=pt,
                completion_tokens=ct,
                input_cost=pt * prices["input"] / 1_000_000,
                output_cost=ct * prices["output"] / 1_000_000,
            )
            return PoolCallResult(model_key=model_key, response=text, cost=cost)

        except requests.exceptions.Timeout:
            logger.warning(f"[{model_key}] timeout (attempt {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return PoolCallResult(model_key=model_key, response="Request timed out",
                                      cost=PoolModelCost(model_key=model_key),
                                      success=False, error="timeout")

        except requests.exceptions.ConnectionError:
            logger.error(f"[{model_key}] connection error on port {cfg['port']}")
            return PoolCallResult(model_key=model_key, response="API Request Error",
                                  cost=PoolModelCost(model_key=model_key),
                                  success=False, error="connection_error")

        except Exception as e:
            logger.error(f"[{model_key}] error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return PoolCallResult(model_key=model_key, response=f"Error: {e}",
                                      cost=PoolModelCost(model_key=model_key),
                                      success=False, error=str(e))

    return PoolCallResult(model_key=model_key, response="API Request Error",
                          cost=PoolModelCost(model_key=model_key),
                          success=False, error="max_retries_exceeded")


def call_pool_models_parallel(
    model_keys: List[str],
    query: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    max_workers: int = 15,
    prompt_template: str = POOL_PROMPT,
) -> Dict[str, PoolCallResult]:
    """Call multiple pool models in parallel on the same query."""
    results: Dict[str, PoolCallResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_pool_model, mk, query,
                max_tokens=max_tokens, temperature=temperature,
                seed=seed, prompt_template=prompt_template,
            ): mk
            for mk in model_keys
        }
        for fut in as_completed(futures):
            mk = futures[fut]
            try:
                results[mk] = fut.result()
            except Exception as e:
                results[mk] = PoolCallResult(
                    model_key=mk, response=f"Error: {e}",
                    cost=PoolModelCost(model_key=mk),
                    success=False, error=str(e),
                )
    return results


# ---------------------------------------------------------------------------
# Router model calling (raw /generate for the router, which is fine for Qwen)
# ---------------------------------------------------------------------------

def call_router(
    prompt: str,
    model_key: str = "qwen2.5-3b-instruct",
    *,
    max_tokens: int = 8192,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
    timeout: int = 300,
) -> Tuple[str, int, int]:
    """Call the router model — local SGLang preferred, API fallback.

    Returns (response_text, prompt_tokens, completion_tokens).
    """
    # Try local SGLang first
    local_ok = False
    if model_key in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_key]
        host = cfg.get("ip_addr", DEFAULT_HOST)
        try:
            r = requests.get(f"http://{host}:{cfg['port']}/health", timeout=2)
            local_ok = (r.status_code == 200)
        except Exception:
            pass

    if local_ok:
        cfg = MODEL_CONFIGS[model_key]
        host = cfg.get("ip_addr", DEFAULT_HOST)
        url = f"http://{host}:{cfg['port']}/generate"

        sampling_params: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9 if temperature > 0 else 1.0,
        }
        if seed is not None:
            sampling_params["sampling_seed"] = seed
        if stop:
            sampling_params["stop"] = stop

        try:
            resp = requests.post(url, json={"text": prompt, "sampling_params": sampling_params},
                                 timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("text", "")
            meta = data.get("meta_info", {})
            pt = meta.get("prompt_tokens", len(prompt) // 4)
            ct = meta.get("completion_tokens", len(text) // 4)

            if stop:
                for s in stop:
                    if s.lstrip("<") in text and s not in text:
                        text = text + s

            return text, pt, ct
        except Exception as e:
            logger.error(f"[router:{model_key}] local error: {e}, falling back to API")
            # Fall through to API

    # API fallback: router uses OpenRouter (separate from pool models using NVIDIA)
    logger.info(f"[router:{model_key}] using API fallback")
    provider = _get_router_api_provider()
    if provider is not None:
        result = provider.call_pool_model(
            model_key=model_key, query=prompt,
            max_tokens=max_tokens, temperature=temperature,
            seed=seed, prompt_template="{query}",
        )
        if result.success:
            return result.response, result.cost.prompt_tokens, result.cost.completion_tokens
        logger.error(f"[router:{model_key}] API error: {result.error}")
    return "", 0, 0


def calculate_cost(model_key: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = API_PRICE_1M_TOKENS.get(model_key, {"input": 0, "output": 0})
    return prompt_tokens * prices["input"] / 1_000_000 + completion_tokens * prices["output"] / 1_000_000


# ---------------------------------------------------------------------------
# API-based pool model calling (migrated from Router-R1 route_service)
# ---------------------------------------------------------------------------

# Backend mode: "local" (SGLang) or "api" (OpenRouter/NVIDIA NIM)
# Auto-detected: if OPENAI_BASE_URL or base_url env var is set, use "api"
_POOL_BACKEND: Optional[str] = None
_api_provider: Any = None  # Lazy-loaded APIPoolProvider (pool models)
_router_api_provider: Any = None  # Lazy-loaded APIPoolProvider (router, uses OpenRouter)


def _detect_backend() -> str:
    """Detect which pool backend to use based on environment.

    Checks: legacy env vars (OPENAI_BASE_URL) → unified config (api_models.json) → local.
    """
    global _POOL_BACKEND
    if _POOL_BACKEND is not None:
        return _POOL_BACKEND

    api_base = os.environ.get("OPENAI_BASE_URL", os.environ.get("base_url", ""))
    api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("api_key", ""))

    if api_base and api_key:
        _POOL_BACKEND = "api"
        logger.info("Pool backend: api (legacy env vars)")
    elif os.path.exists(_API_CONFIG_PATH):
        _POOL_BACKEND = "api"
        logger.info("Pool backend: api (unified config api_models.json)")
    else:
        _POOL_BACKEND = "local"
        logger.info("Pool backend: local (SGLang servers)")

    return _POOL_BACKEND


_API_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config", "api_models.json",
)


def _get_api_provider() -> Any:
    """Lazy-load the API provider for pool models.

    Prefers unified config (api_models.json) when available.
    Falls back to legacy env vars (OPENAI_BASE_URL / OPENAI_API_KEY).
    """
    global _api_provider
    if _api_provider is None:
        from skillorchestra.routing.api_provider import APIPoolProvider
        if os.path.exists(_API_CONFIG_PATH):
            logger.info("Pool API provider: unified config (api_models.json)")
            _api_provider = APIPoolProvider(prompt_template=POOL_PROMPT)
        else:
            base_url = os.environ.get("OPENAI_BASE_URL", os.environ.get("base_url", ""))
            api_key = os.environ.get("OPENAI_API_KEY", os.environ.get("api_key", ""))
            model_map = "nvidia" if "nvidia" in base_url.lower() else "openrouter"
            logger.info("Pool API provider: legacy (%s)", model_map)
            _api_provider = APIPoolProvider(
                base_url=base_url, api_key=api_key,
                model_map=model_map, prompt_template=POOL_PROMPT,
            )
    return _api_provider


def _get_router_api_provider() -> Any:
    """Lazy-load the router's API provider.

    Priority: unified config (api_models.json) — supports MiniMax, NVIDIA, OpenRouter.
    Falls back to legacy: NVIDIA NIM → OpenRouter.

    Config mode lets router models specify their own provider (e.g. minimax-m2.7
    uses MiniMax API). Add or change router models by editing api_models.json
    and setting the corresponding API key env var — no code changes needed.
    """
    global _router_api_provider
    if _router_api_provider is None:
        from skillorchestra.routing.api_provider import APIPoolProvider
        if os.path.exists(_API_CONFIG_PATH):
            logger.info("Router API provider: unified config (api_models.json)")
            _router_api_provider = APIPoolProvider(prompt_template="{query}")
        else:
            nvidia_base = os.environ.get("OPENAI_BASE_URL", "")
            nvidia_key = os.environ.get("OPENAI_API_KEY", "")
            if nvidia_base and nvidia_key:
                logger.info("Router API provider: NVIDIA NIM (legacy)")
                _router_api_provider = APIPoolProvider(
                    base_url=nvidia_base, api_key=nvidia_key,
                    model_map="nvidia", prompt_template="{query}",
                )
            else:
                openrouter_base = os.environ.get("OPENROUTER_BASE_URL", "")
                openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
                if openrouter_base and openrouter_key:
                    logger.info("Router API provider: OpenRouter (legacy)")
                    _router_api_provider = APIPoolProvider(
                        base_url=openrouter_base, api_key=openrouter_key,
                        model_map="openrouter", prompt_template="{query}",
                    )
                else:
                    logger.warning("Router API provider: no credentials found")
    return _router_api_provider


def call_pool_models(
    model_keys: List[str],
    query: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    max_workers: int = 15,
    prompt_template: str = POOL_PROMPT,
) -> Dict[str, PoolCallResult]:
    """Unified pool model calling: auto-detects local SGLang vs remote API.

    When OPENAI_BASE_URL/base_url and OPENAI_API_KEY/api_key env vars are set,
    uses external API calls (migrated from Router-R1 route_service).
    Otherwise, uses local SGLang servers (original behavior).

    Args:
        model_keys: List of model keys to call
        query: The question/prompt
        max_tokens: Max tokens per response
        temperature: Sampling temperature
        seed: Random seed
        max_workers: Max parallel workers
        prompt_template: Prompt format string

    Returns:
        Dict mapping model_key → PoolCallResult
    """
    backend = _detect_backend()

    if backend == "api":
        provider = _get_api_provider()
        return provider.call_pool_models_parallel(
            model_keys=model_keys,
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            max_workers=max_workers,
            prompt_template=prompt_template,
        )
    else:
        return call_pool_models_parallel(
            model_keys=model_keys,
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            max_workers=max_workers,
            prompt_template=prompt_template,
        )


def call_pool_model_unified(
    model_key: str,
    query: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    timeout: int = 300,
    max_retries: int = 5,
    prompt_template: str = POOL_PROMPT,
) -> PoolCallResult:
    """Unified single pool model call: auto-detects local SGLang vs remote API.

    Args:
        model_key: Model key to call
        query: The question/prompt
        max_tokens: Max tokens per response
        temperature: Sampling temperature
        seed: Random seed
        timeout: Request timeout (local mode only)
        max_retries: Max retries (local mode only)
        prompt_template: Prompt format string

    Returns:
        PoolCallResult
    """
    backend = _detect_backend()

    if backend == "api":
        provider = _get_api_provider()
        return provider.call_pool_model(
            model_key=model_key,
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            prompt_template=prompt_template,
        )
    else:
        return call_pool_model(
            model_key=model_key,
            query=query,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            timeout=timeout,
            max_retries=max_retries,
            prompt_template=prompt_template,
        )
