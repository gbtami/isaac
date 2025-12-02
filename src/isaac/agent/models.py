"""Model registry and builder for pydantic-ai agents.

Reads model settings from `models.json` and environment variables via `.env`.
Supports switching models/providers at runtime (used by the `/model` command).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict

from dotenv import load_dotenv
from pydantic_ai import Agent as PydanticAgent  # type: ignore
from pydantic_ai.models.anthropic import AnthropicModel  # type: ignore
from pydantic_ai.models.google import GoogleModel  # type: ignore
from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
from pydantic_ai.models.openrouter import OpenRouterModel  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore
from pydantic_ai.providers.anthropic import AnthropicProvider  # type: ignore
from pydantic_ai.providers.google import GoogleProvider  # type: ignore
from pydantic_ai.providers.ollama import OllamaProvider  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
from pydantic_ai.providers.openrouter import OpenRouterProvider  # type: ignore

from isaac.agent.brain.prompt import SYSTEM_PROMPT

logger = logging.getLogger("acp_server")

OLLAMA_BASE_URL = "http://localhost:11434/v1"

HIDDEN_MODELS = {"test", "function-model"}
DEFAULT_CONFIG = {
    "current": "function-model",
    "models": {
        "test": {
            "model": "test",
            "description": "Deterministic local model for offline/testing",
        },
        "function-model": {
            "provider": "function",
            "model": "function",
            "description": "In-process function model for deterministic testing",
        },
        "openai-gpt4o-mini": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "description": "OpenAI GPT-4o mini",
        },
        "anthropic-claude-3-5-sonnet": {
            "provider": "anthropic",
            "model": "claude-3-5-sonnet-20240620",
            "description": "Anthropic Claude 3.5 Sonnet",
        },
        "google-gemini-2.5-pro": {
            "provider": "google",
            "model": "gemini-2.5-pro",
            "description": "Google Gemini 2.5 Pro",
        },
        "openrouter-x-ai/grok-4.1-fast:free": {
            "provider": "openrouter",
            "model": "x-ai/grok-4.1-fast:free",
            "description": "OpenRouter proxy for x-ai/grok-4.1-fast:free",
        },
        "openrouter-z-ai/glm-4.5-air:free": {
            "provider": "openrouter",
            "model": "z-ai/glm-4.5-air:free",
            "description": "OpenRouter proxy for z-ai/glm-4.5-air:free",
        },
        "openrouter-kwaipilot/kat-coder-pro:free": {
            "provider": "openrouter",
            "model": "kwaipilot/kat-coder-pro:free",
            "description": "OpenRouter proxy for kwaipilot/kat-coder-pro:free",
        },
        "ollama-qwen2.5-coder-3b": {
            "provider": "ollama",
            "model": "hhao/qwen2.5-coder-tools:3b",
            "description": "Ollama qwen2.5-coder-tools:3b (local)",
        },
        "ollama-NazareAI-Python-Programmer-3B": {
            "provider": "ollama",
            "model": "0xroyce/NazareAI-Python-Programmer-3B",
            "description": "NazareAI-Python-Programmer-3B (local)",
        },
    },
}

CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME") or (Path.home() / ".config")) / "isaac"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS_FILE = CONFIG_DIR / "models.json"


def load_models_config() -> Dict[str, Any]:
    if not MODELS_FILE.exists():
        MODELS_FILE.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        config = DEFAULT_CONFIG.copy()
    else:
        try:
            config = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
        except Exception:
            config = DEFAULT_CONFIG.copy()
    dirty = False
    # Backfill missing defaults
    for key, value in DEFAULT_CONFIG["models"].items():
        config.setdefault("models", {})
        if key not in config["models"]:
            config["models"][key] = value
            dirty = True
    # Ensure function-model stays safe for testing (no auto tool calls)
    fn_model = config["models"].get("function-model", {})
    if fn_model.get("provider") != "function":
        fn_model["provider"] = "function"
        dirty = True
    if fn_model.get("model") != "function":
        fn_model["model"] = "function"
        dirty = True
    fn_model.setdefault("description", DEFAULT_CONFIG["models"]["function-model"]["description"])
    config["models"]["function-model"] = fn_model

    config.setdefault("current", DEFAULT_CONFIG["current"])
    if dirty:
        save_models_config(config)
    return config


def save_models_config(config: Dict[str, Any]) -> None:
    MODELS_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def list_models() -> Dict[str, Any]:
    return load_models_config().get("models", {})


def list_user_models() -> Dict[str, Any]:
    """Return models suitable for end users (hides internal/testing models)."""
    return {mid: meta for mid, meta in list_models().items() if mid not in HIDDEN_MODELS}


def set_current_model(model_id: str) -> str:
    config = load_models_config()
    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Unknown model id: {model_id}")
    config["current"] = model_id
    save_models_config(config)
    return model_id


def _build_provider_model(model_id: str, model_entry: Dict[str, Any]) -> Any:
    provider = (model_entry.get("provider") or "").lower()
    model_spec = model_entry.get("model") or "test"
    api_key = model_entry.get("api_key")

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for openai models")
        provider_obj = OpenAIProvider(api_key=key)
        return OpenAIChatModel(model_spec, provider=provider_obj)

    if provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for anthropic models")
        provider_obj = AnthropicProvider(api_key=key)
        return AnthropicModel(model_spec, provider=provider_obj)

    if provider == "google":
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is required for google models")
        provider_obj = GoogleProvider(api_key=key)
        return GoogleModel(model_spec, provider=provider_obj)

    if provider == "openrouter":
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for openrouter models")
        provider_obj = OpenRouterProvider(api_key=key)
        return OpenRouterModel(model_spec, provider=provider_obj)

    if provider == "ollama":
        provider_obj = OllamaProvider(base_url=OLLAMA_BASE_URL)
        return OpenAIChatModel(model_spec, provider=provider_obj)

    if provider == "function":
        return TestModel(call_tools=[])

    # default to test or direct spec
    return model_spec


def build_agent(
    model_id: str,
    register_tools: Callable[[Any], None],
    *,
    toolsets: list[Any] | None = None,
) -> Any:
    """Build a pydantic-ai Agent for the given model id."""
    load_dotenv()
    config = load_models_config()
    model_entry = config.get("models", {}).get(model_id) or DEFAULT_CONFIG["models"]["test"]

    model_obj = _build_provider_model(model_id, model_entry)
    logger.info("MODEL: %s", model_obj.model_name)

    agent = PydanticAgent(model_obj, toolsets=toolsets or (), system_prompt=SYSTEM_PROMPT)
    register_tools(agent)
    return agent
