"""Model registry and builder for pydantic-ai agents.

Reads model settings from `models.json` and environment variables via `.env`.
Supports switching models/providers at runtime (used by the `/model` command).
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
import configparser
from pathlib import Path
from typing import Any, Dict

from pydantic_ai.models import Model  # type: ignore
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings  # type: ignore
from pydantic_ai.models.cerebras import CerebrasModel  # type: ignore
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings  # type: ignore
from pydantic_ai.models.mistral import MistralModel  # type: ignore
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings  # type: ignore
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore
from pydantic_ai.settings import ModelSettings  # type: ignore

from pydantic_ai.providers.anthropic import AnthropicProvider  # type: ignore
from pydantic_ai.providers.cerebras import CerebrasProvider  # type: ignore
from pydantic_ai.providers.google import GoogleProvider  # type: ignore
from pydantic_ai.providers.mistral import MistralProvider  # type: ignore
from pydantic_ai.providers.ollama import OllamaProvider  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
from pydantic_ai.providers.openrouter import OpenRouterProvider  # type: ignore


logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"

FUNCTION_MODEL_ID = "function:function"
HIDDEN_MODELS = {FUNCTION_MODEL_ID}
DEFAULT_MODEL_PROVIDER = "function"
DEFAULT_MODEL_NAME = "function"
DEFAULT_MODEL_ID = f"{DEFAULT_MODEL_PROVIDER}:{DEFAULT_MODEL_NAME}"
DEFAULT_CONFIG = {
    "models": {
        FUNCTION_MODEL_ID: {
            "provider": "function",
            "model": "function",
            "description": "In-process function model for deterministic testing",
        },
        "openai:gpt-4o-mini": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "description": "OpenAI GPT-4o mini",
        },
        "anthropic:claude-3-5-sonnet-20240620": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20240620",
            "description": "Anthropic Claude 3.5 Sonnet",
        },
        "google:gemini-2.5-flash": {
            "provider": "google",
            "model": "gemini-2.5-flash",
            "description": "Google Gemini 2.5 Flash",
        },
        "openrouter:openai/gpt-oss-120b": {
            "provider": "openrouter",
            "model": "openai/gpt-oss-120b",
            "description": "OpenRouter proxy for openai/gpt-oss-120b",
        },
        "cerebras:openai/gpt-oss-120b": {
            "provider": "cerebras",
            "model": "openai/gpt-oss-120b",
            "description": "Cerebras proxy for openai/gpt-oss-120b",
        },
        "mistral:devstral-medium-latest": {
            "provider": "mistral",
            "model": "devstral-medium-latest",
            "description": "Mistral DevStral Medium (latest)",
        },
        "ollama:ministral-3:3b": {
            "provider": "ollama",
            "model": "ministral-3:3b",
            "description": "Ollama ministral-3:3b (local)",
        },
    },
}

CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "isaac"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
ENV_FILE = CONFIG_DIR / ".env"

MODELS_FILE = CONFIG_DIR / "models.json"
MODELS_DEV_URL = "https://models.dev/api.json"


def load_models_config() -> Dict[str, Any]:
    if not MODELS_FILE.exists():
        config = json.loads(json.dumps(DEFAULT_CONFIG))
        _apply_context_limits(config)
        MODELS_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        try:
            config = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            config = json.loads(json.dumps(DEFAULT_CONFIG))

    # Backfill missing defaults
    config.setdefault("models", {})
    for key, value in DEFAULT_CONFIG["models"].items():
        config["models"].setdefault(key, value)

    # Ensure function model stays safe for testing (no auto tool calls)
    fn_model = config["models"].get(FUNCTION_MODEL_ID, {})
    fn_model["provider"] = "function"
    fn_model["model"] = "function"
    fn_model.setdefault("description", DEFAULT_CONFIG["models"][FUNCTION_MODEL_ID]["description"])
    config["models"][FUNCTION_MODEL_ID] = fn_model

    return config


def current_model_id() -> str:
    """Return the persisted current model id."""

    return _load_current_model()


def list_models() -> Dict[str, Any]:
    return load_models_config().get("models", {})


def list_user_models() -> Dict[str, Any]:
    """Return models suitable for end users (hides internal/testing models)."""
    return {mid: meta for mid, meta in list_models().items() if mid not in HIDDEN_MODELS}


def set_current_model(model_id: str) -> str:
    models_cfg = list_models()
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    _save_current_model(model_id)
    return model_id


def _load_current_model() -> str:
    """Read the persisted current model selection from ini (or default)."""

    settings_file = MODELS_FILE.parent / "isaac.ini"
    parser = configparser.ConfigParser()
    if settings_file.exists():
        try:
            parser.read(settings_file)
            current = parser.get("models", "current_model", fallback=None)
            if current:
                return current
        except Exception:  # pragma: no cover - best effort load
            pass
    return DEFAULT_MODEL_ID


def _save_current_model(model_id: str) -> None:
    """Persist the current model selection separately from models.json."""

    settings_file = MODELS_FILE.parent / "isaac.ini"
    parser = configparser.ConfigParser()
    parser["models"] = {"current_model": model_id}
    try:
        with settings_file.open("w", encoding="utf-8") as f:
            parser.write(f)
    except Exception:  # pragma: no cover - best effort persistence
        pass


def get_context_limit(model_id: str) -> int | None:
    """Optional per-model context window (tokens) if configured."""
    config = load_models_config()
    model_entry = config.get("models", {}).get(model_id, {})
    limit = model_entry.get("context_limit")
    return int(limit) if isinstance(limit, int) else None


def _apply_context_limits(config: Dict[str, Any]) -> None:
    """Populate context_limit fields from models.dev when available."""
    try:
        limits_index = _fetch_models_dev_limits()
    except Exception:  # pragma: no cover - network failure path
        limits_index = {}

    for model_id, meta in config.get("models", {}).items():
        if meta.get("context_limit") is not None:
            continue
        provider = (meta.get("provider") or "").lower()
        model_name = (meta.get("model") or "").lower()
        if not provider or not model_name:
            continue
        limit = limits_index.get((provider, model_name))
        if limit:
            meta["context_limit"] = limit


def _fetch_models_dev_limits() -> Dict[tuple[str, str], int]:
    """Fetch models.dev index and build (provider, model) -> context_limit map."""
    with urllib.request.urlopen(MODELS_DEV_URL, timeout=5) as resp:  # type: ignore[call-arg]
        data = json.loads(resp.read().decode("utf-8"))
    index: Dict[tuple[str, str], int] = {}
    for provider_key, provider_data in data.items():
        models = provider_data.get("models", {}) or {}
        for model_key, model_data in models.items():
            limit = (model_data.get("limit") or {}).get("context")
            if isinstance(limit, (int, float)) and limit > 0:
                index[(provider_key.lower(), model_key.lower())] = int(limit)
    return index


def _build_provider_model(model_id: str, model_entry: Dict[str, Any]) -> tuple[Model, ModelSettings | None]:
    provider = (model_entry.get("provider") or "").lower()
    model_spec = model_entry.get("model") or "test"
    api_key = model_entry.get("api_key")

    def _openai_model_supports_reasoning_effort(name: str) -> bool:
        # OpenAI "reasoning models" are currently named like `o1-*`, `o3-*`, `o4-*`, etc.
        return bool(re.match(r"^o\d", (name or "").strip().lower()))

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for openai models")
        provider_obj = OpenAIProvider(api_key=key)
        settings: OpenAIChatModelSettings | None = None
        if _openai_model_supports_reasoning_effort(str(model_spec)):
            settings = OpenAIChatModelSettings(openai_reasoning_effort="medium")
        return OpenAIChatModel(model_spec, provider=provider_obj), settings

    if provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic models")
        provider_obj = AnthropicProvider(api_key=key)
        settings = AnthropicModelSettings(anthropic_thinking={"type": "enabled", "budget_tokens": 512})
        return AnthropicModel(model_spec, provider=provider_obj), settings

    if provider == "mistral":
        key = api_key or os.getenv("MISTRAL_API_KEY")
        if not key:
            raise RuntimeError("MISTRAL_API_KEY is required for mistral models")
        provider_obj = MistralProvider(api_key=key)
        return MistralModel(model_spec, provider=provider_obj), None

    if provider == "cerebras":
        key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not key:
            raise RuntimeError("CEREBRAS_API_KEY is required for cerebras models")
        provider_obj = CerebrasProvider(api_key=key)
        return CerebrasModel(model_spec, provider=provider_obj), None

    if provider == "google":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for google models")
        provider_obj = GoogleProvider(api_key=key)
        settings = GoogleModelSettings(google_thinking_config={"include_thoughts": True})
        return GoogleModel(model_spec, provider=provider_obj), settings

    if provider == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for openrouter models")
        provider_obj = OpenRouterProvider(api_key=key)
        settings = OpenRouterModelSettings(openrouter_reasoning={"effort": "medium"})
        return OpenRouterModel(model_spec, provider=provider_obj), settings

    if provider == "ollama":
        # Force JSON mode so local models emit parseable tool payloads.
        provider_obj = OllamaProvider(base_url=OLLAMA_BASE_URL)
        return OpenAIChatModel(model_spec, provider=provider_obj), None

    if provider == "function":
        return TestModel(call_tools=[]), None

    # default to test or direct spec
    return model_spec, None
