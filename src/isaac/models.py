"""Model registry and builder for pydantic-ai agents.

Reads model settings from `models.json` and environment variables via `.env`.
Supports switching models/providers at runtime (used by the `/model` command).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.anthropic import AnthropicModel  # type: ignore
from pydantic_ai.models.google import GoogleModel  # type: ignore
from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
from pydantic_ai.models.openrouter import OpenRouterModel  # type: ignore
from pydantic_ai.providers.anthropic import AnthropicProvider  # type: ignore
from pydantic_ai.providers.google import GoogleProvider  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
from pydantic_ai.providers.openrouter import OpenRouterProvider  # type: ignore

DEFAULT_CONFIG = {
    "current": "function-model",
    "models": {
        "test": {
            "model": "test",
            "description": "Deterministic local model for offline/testing",
        },
        "function-model": {
            "model": "test",
            "description": "Deterministic local model for offline/testing",
        },
        "openai-gpt4o-mini": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
            "description": "OpenAI GPT-4o mini",
        },
        "anthropic-claude-3-5-sonnet": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20240620",
            "base_url": "https://api.anthropic.com",
            "description": "Anthropic Claude 3.5 Sonnet",
        },
        "google-gemini-2.5-pro": {
            "provider": "google",
            "model": "gemini-2.5-pro",
            "base_url": "https://generativelanguage.googleapis.com",
            "description": "Google Gemini 2.5 Pro",
        },
        "openrouter-gpt4o-mini": {
            "provider": "openrouter",
            "model": "openai/gpt-4o-mini",
            "base_url": "https://openrouter.ai/api/v1",
            "description": "OpenRouter proxy for GPT-4o mini",
        },
    },
}

MODELS_FILE = Path(__file__).resolve().parents[2] / "models.json"


def load_models_config() -> Dict[str, Any]:
    if not MODELS_FILE.exists():
        MODELS_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        return DEFAULT_CONFIG.copy()
    try:
        config = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    except Exception:
        config = DEFAULT_CONFIG.copy()
    # Backfill missing defaults
    for key, value in DEFAULT_CONFIG["models"].items():
        config.setdefault("models", {})
        config["models"].setdefault(key, value)
    config.setdefault("current", DEFAULT_CONFIG["current"])
    return config


def save_models_config(config: Dict[str, Any]) -> None:
    MODELS_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def list_models() -> Dict[str, Any]:
    return load_models_config().get("models", {})


def set_current_model(model_id: str) -> str:
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    config["current"] = model_id
    save_models_config(config)
    return model_id


def _build_provider_model(model_entry: Dict[str, Any]) -> Any:
    provider = (model_entry.get("provider") or "").lower()
    model_spec = model_entry.get("model") or "test"
    base_url = model_entry.get("base_url")
    api_key = model_entry.get("api_key")

    if provider == "openai" or str(model_spec).startswith("openai:"):
        key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CEREBRAS_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY (or CEREBRAS_API_KEY) is required for openai models")
        url = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        model_name = str(model_spec).split(":", 1)[-1]
        provider_obj = OpenAIProvider(base_url=url, api_key=key)
        return OpenAIChatModel(model_name, provider=provider_obj)

    if provider == "anthropic" or str(model_spec).startswith("anthropic:"):
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic models")
        url = base_url or os.getenv("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        model_name = str(model_spec).split(":", 1)[-1]
        provider_obj = AnthropicProvider(api_key=key, base_url=url) if url else AnthropicProvider(api_key=key)
        return AnthropicModel(model_name, provider=provider_obj)

    if provider == "google" or str(model_spec).startswith("google:"):
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for google models")
        url = base_url or os.getenv("GOOGLE_API_BASE_URL") or "https://generativelanguage.googleapis.com"
        model_name = str(model_spec).split(":", 1)[-1]
        provider_obj = GoogleProvider(api_key=key, base_url=url) if url else GoogleProvider(api_key=key)
        return GoogleModel(model_name, provider=provider_obj)

    if provider == "openrouter" or str(model_spec).startswith("openrouter:"):
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for openrouter models")
        url = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        model_name = str(model_spec).split(":", 1)[-1]
        provider_obj = OpenRouterProvider(api_key=key, base_url=url)
        return OpenRouterModel(model_name, provider=provider_obj)

    if provider == "function" or str(model_spec).startswith("function:"):
        return "test"

    # default to test or direct spec
    return model_spec


def build_agent(model_id: str, register_tools: Callable[[Any], None]) -> Any:
    """Build a pydantic-ai Agent for the given model id."""
    load_dotenv()
    config = load_models_config()
    model_entry = config.get("models", {}).get(model_id) or DEFAULT_CONFIG["models"]["test"]

    model_obj = _build_provider_model(model_entry)
    agent = PydanticAgent(model_obj)
    register_tools(agent)
    return agent
