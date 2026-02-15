"""Model registry and builder for pydantic-ai agents.

Loads model defaults from the offline models.dev catalog, then applies local/user
overrides from `models.json`. Supports switching models/providers at runtime
(used by the `/model` command).
"""

from __future__ import annotations

import inspect
import json
import logging
import configparser
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict

import httpx
from dotenv import load_dotenv
from pydantic_ai.models import Model  # type: ignore
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings  # type: ignore
from pydantic_ai.models.bedrock import BedrockConverseModel  # type: ignore
from pydantic_ai.models.cerebras import CerebrasModel  # type: ignore
from pydantic_ai.models.cohere import CohereModel  # type: ignore
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings  # type: ignore
from pydantic_ai.models.groq import GroqModel  # type: ignore
from pydantic_ai.models.huggingface import HuggingFaceModel  # type: ignore
from pydantic_ai.models.mistral import MistralModel  # type: ignore
from pydantic_ai.models.openai import (  # type: ignore
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings  # type: ignore
from pydantic_ai.models.test import TestModel  # type: ignore
from pydantic_ai.models.xai import XaiModel  # type: ignore
from pydantic_ai.settings import ModelSettings  # type: ignore

from pydantic_ai.providers.alibaba import AlibabaProvider  # type: ignore
from pydantic_ai.providers.anthropic import AnthropicProvider  # type: ignore
from pydantic_ai.providers.azure import AzureProvider  # type: ignore
from pydantic_ai.providers.bedrock import BedrockProvider  # type: ignore
from pydantic_ai.providers.cerebras import CerebrasProvider  # type: ignore
from pydantic_ai.providers.cohere import CohereProvider  # type: ignore
from pydantic_ai.providers.deepseek import DeepSeekProvider  # type: ignore
from pydantic_ai.providers.fireworks import FireworksProvider  # type: ignore
from pydantic_ai.providers.github import GitHubProvider  # type: ignore
from pydantic_ai.providers.google import GoogleProvider  # type: ignore
from pydantic_ai.providers.google_vertex import GoogleVertexProvider  # type: ignore
from pydantic_ai.providers.groq import GroqProvider  # type: ignore
from pydantic_ai.providers.huggingface import HuggingFaceProvider  # type: ignore
from pydantic_ai.providers.mistral import MistralProvider  # type: ignore
from pydantic_ai.providers.moonshotai import MoonshotAIProvider  # type: ignore
from pydantic_ai.providers.nebius import NebiusProvider  # type: ignore
from pydantic_ai.providers.ollama import OllamaProvider  # type: ignore
from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore
from pydantic_ai.providers.openrouter import OpenRouterProvider  # type: ignore
from pydantic_ai.providers.ovhcloud import OVHcloudProvider  # type: ignore
from pydantic_ai.providers.together import TogetherProvider  # type: ignore
from pydantic_ai.providers.vercel import VercelProvider  # type: ignore
from pydantic_ai.providers.xai import XaiProvider  # type: ignore

from isaac.agent.oauth.code_assist import CodeAssistModel
from isaac.agent.oauth.openai_codex import (
    OPENAI_CODEX_BASE_URL,
    has_openai_tokens,
    openai_auth_request_hook,
)
from isaac.agent.oauth.openai_codex.client import OpenAICodexAsyncClient
from isaac.paths import config_dir

logger = logging.getLogger(__name__)

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LLM_TIMEOUT_S = float(os.getenv("ISAAC_LLM_TIMEOUT_S", "60"))
DEFAULT_LLM_CONNECT_TIMEOUT_S = float(os.getenv("ISAAC_LLM_CONNECT_TIMEOUT_S", "5"))
DEFAULT_LLM_RETRIES = int(os.getenv("ISAAC_LLM_RETRIES", "2"))
_HTTP_CLIENTS: dict[str, httpx.AsyncClient] = {}

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
    },
}

CONFIG_DIR = config_dir()
ENV_FILE = CONFIG_DIR / ".env"

# Local models.json in the repository
LOCAL_MODELS_FILE = Path(__file__).parent.parent / "models.json"
# User-specific models.json in the config directory
USER_MODELS_FILE = CONFIG_DIR / "models.json"
MODELS_DEV_SNAPSHOT_FILE = Path(__file__).parent / "data" / "models_dev_api.json"
MODELS_DEV_CATALOG_FILE = Path(__file__).parent / "data" / "models_dev_catalog.json"


def load_runtime_env() -> None:
    """Load shared environment from the platform config dir."""

    load_dotenv(ENV_FILE, override=False)


def load_models_config() -> Dict[str, Any]:
    """Load models from defaults, offline catalog, then local/user overrides."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config.setdefault("models", {})
    models = config["models"]

    # Merge offline catalog generated from models.dev snapshot.
    models.update(_load_catalog_models())

    # Merge local models.json in the repository.
    if LOCAL_MODELS_FILE.exists():
        try:
            local_config = json.loads(LOCAL_MODELS_FILE.read_text(encoding="utf-8"))
            if isinstance(local_config, dict):
                for key, value in (local_config.get("models", {}) or {}).items():
                    models[key] = value
        except Exception:
            logger.warning("Failed to load local models.json, keeping defaults/catalog")

    # Try to load from user-specific models.json in the config directory
    if USER_MODELS_FILE.exists():
        try:
            user_config = json.loads(USER_MODELS_FILE.read_text(encoding="utf-8"))
            for key, value in user_config.get("models", {}).items():
                models[key] = value
        except Exception:
            logger.warning("Failed to load user models.json, ignoring")

    # Ensure function model stays safe for testing (no auto tool calls)
    fn_model = models.get(FUNCTION_MODEL_ID, {})
    fn_model["provider"] = "function"
    fn_model["model"] = "function"
    fn_model.setdefault("description", DEFAULT_CONFIG["models"][FUNCTION_MODEL_ID]["description"])
    models[FUNCTION_MODEL_ID] = fn_model

    _apply_context_limits(config)

    return config


def update_models_from_models_dev() -> None:
    """Backfill context limits for models declared in local models.json."""
    if not LOCAL_MODELS_FILE.exists():
        logger.info("No local models.json found; skipping context limit backfill")
        return
    try:
        limits_index = _fetch_models_dev_limits()
        config = json.loads(LOCAL_MODELS_FILE.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            logger.warning("Local models.json is invalid; skipping context limit backfill")
            return
        config.setdefault("models", {})
        updated = False

        for model_id, meta in config.get("models", {}).items():
            if not isinstance(meta, dict):
                continue
            if meta.get("context_limit") is not None:
                continue
            provider = (meta.get("provider") or "").lower()
            model_name = (meta.get("model") or "").lower()
            if not provider or not model_name:
                continue
            limit = limits_index.get((provider, model_name))
            if limit:
                meta["context_limit"] = limit
                updated = True

        if updated:
            LOCAL_MODELS_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")
            logger.info("Updated local models.json with context limits from snapshot")
        else:
            logger.info("No updates needed for models.json")
    except Exception as e:
        logger.error(f"Failed to update models.json from snapshot: {e}")
        raise


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
    if not _save_current_model(model_id):
        raise RuntimeError("Failed to persist current model selection")
    return model_id


def _load_current_model() -> str:
    """Read the persisted current model selection from ini (or default)."""

    settings_file = USER_MODELS_FILE.parent / "isaac.ini"
    parser = configparser.ConfigParser()
    models_cfg = list_models()
    fallback_model_id = _fallback_current_model_id(models_cfg)
    if settings_file.exists():
        try:
            parser.read(settings_file)
            current = parser.get("models", "current_model", fallback=None)
            if current:
                if current in models_cfg:
                    return current
                logger.warning(
                    "Persisted current model '%s' is unavailable; falling back to '%s'", current, fallback_model_id
                )
                _save_current_model(fallback_model_id)
                return fallback_model_id
        except Exception:  # pragma: no cover - best effort load
            pass
    return fallback_model_id


def _save_current_model(model_id: str) -> bool:
    """Persist the current model selection separately from models.json."""

    settings_file = USER_MODELS_FILE.parent / "isaac.ini"
    parser = configparser.ConfigParser()
    parser["models"] = {"current_model": model_id}
    try:
        with settings_file.open("w", encoding="utf-8") as f:
            parser.write(f)
        return True
    except Exception:  # pragma: no cover - best effort persistence
        logger.warning("Failed to persist current model '%s' in %s", model_id, settings_file)
        return False


def _fallback_current_model_id(models_cfg: Dict[str, Any] | None = None) -> str:
    """Return a safe fallback model id that always exists in current config."""

    cfg = models_cfg or list_models()
    if DEFAULT_MODEL_ID in cfg:
        return DEFAULT_MODEL_ID
    user_models = sorted(mid for mid in cfg if mid not in HIDDEN_MODELS)
    if user_models:
        return user_models[0]
    model_ids = sorted(cfg.keys())
    if model_ids:
        return model_ids[0]
    return DEFAULT_MODEL_ID


def get_context_limit(model_id: str) -> int | None:
    """Optional per-model context window (tokens) if configured."""
    config = load_models_config()
    model_entry = config.get("models", {}).get(model_id, {})
    limit = model_entry.get("context_limit")
    return int(limit) if isinstance(limit, int) else None


def _apply_context_limits(config: Dict[str, Any]) -> None:
    """Populate context_limit fields from offline models.dev snapshot when available."""
    try:
        limits_index = _fetch_models_dev_limits()
    except Exception:
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
    """Read models.dev snapshot and build (provider, model) -> context_limit map."""
    if not MODELS_DEV_SNAPSHOT_FILE.exists():
        logger.debug("models.dev snapshot missing: %s", MODELS_DEV_SNAPSHOT_FILE)
        return {}
    data = json.loads(MODELS_DEV_SNAPSHOT_FILE.read_text(encoding="utf-8"))
    index: Dict[tuple[str, str], int] = {}
    for provider_key, provider_data in data.items() if isinstance(data, dict) else []:
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models", {}) or {}
        if not isinstance(models, dict):
            continue
        for model_key, model_data in models.items():
            if not isinstance(model_data, dict):
                continue
            limit = (model_data.get("limit") or {}).get("context")
            if isinstance(limit, (int, float)) and limit > 0:
                index[(provider_key.lower(), model_key.lower())] = int(limit)
    return index


def _load_catalog_models() -> Dict[str, Any]:
    """Load model entries from offline catalog generated from models.dev."""
    if not MODELS_DEV_CATALOG_FILE.exists():
        logger.debug("models.dev catalog missing: %s", MODELS_DEV_CATALOG_FILE)
        return {}
    try:
        catalog = json.loads(MODELS_DEV_CATALOG_FILE.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read models.dev catalog; ignoring")
        return {}

    providers = catalog.get("providers", {})
    if not isinstance(providers, dict):
        return {}

    models: dict[str, Any] = {}
    for provider, provider_entry in providers.items():
        if not isinstance(provider_entry, dict):
            continue
        label = str(provider_entry.get("label") or provider)
        raw_models = provider_entry.get("models", [])
        if not isinstance(raw_models, list):
            continue
        for raw_model in raw_models:
            if not isinstance(raw_model, dict):
                continue
            model_name = str(raw_model.get("id") or "").strip()
            if not model_name:
                continue
            model_id = f"{provider}:{model_name}"
            model_entry: dict[str, Any] = {
                "provider": provider,
                "model": model_name,
                "description": f"{label} {str(raw_model.get('name') or model_name)}",
            }
            context_limit = raw_model.get("context_limit")
            if isinstance(context_limit, int) and context_limit > 0:
                model_entry["context_limit"] = context_limit
            models[model_id] = model_entry
    return models


def _get_http_client(
    provider: str,
    *,
    event_hooks: dict[str, list[Callable[..., Any]]] | None = None,
    client_cls: type[httpx.AsyncClient] = httpx.AsyncClient,
) -> httpx.AsyncClient:
    """Return a cached HTTP client with explicit timeouts and retries."""

    cache_key = provider if event_hooks is None else f"{provider}:hooks"
    if client_cls is not httpx.AsyncClient:
        cache_key = f"{cache_key}:{client_cls.__name__}"
    client = _HTTP_CLIENTS.get(cache_key)
    if client and not client.is_closed:
        return client
    timeout = httpx.Timeout(
        DEFAULT_LLM_TIMEOUT_S,
        connect=DEFAULT_LLM_CONNECT_TIMEOUT_S,
        read=DEFAULT_LLM_TIMEOUT_S,
        write=DEFAULT_LLM_TIMEOUT_S,
    )
    transport = httpx.AsyncHTTPTransport(retries=DEFAULT_LLM_RETRIES)
    client = client_cls(timeout=timeout, transport=transport, event_hooks=event_hooks)
    _HTTP_CLIENTS[cache_key] = client
    return client


def _provider_with_http_client(provider_cls: type, provider: str, **kwargs: object) -> Any:
    """Build a provider using a shared HTTP client when supported."""

    if "http_client" in inspect.signature(provider_cls).parameters and "http_client" not in kwargs:
        kwargs["http_client"] = _get_http_client(provider)
    return provider_cls(**kwargs)


def _build_provider_model(model_id: str, model_entry: Dict[str, Any]) -> tuple[Model, ModelSettings | None]:
    provider = (model_entry.get("provider") or "").lower()
    model_spec = model_entry.get("model") or "test"
    api_key = model_entry.get("api_key")
    _ = model_id

    def _openai_model_supports_reasoning_effort(name: str) -> bool:
        # OpenAI "reasoning models" are currently named like `o1-*`, `o3-*`, `o4-*`, etc.
        return bool(re.match(r"^o\d", (name or "").strip().lower()))

    def _resolve_api_key(provider_name: str, *env_names: str) -> str:
        if isinstance(api_key, str) and api_key:
            return api_key
        for env_name in env_names:
            value = os.getenv(env_name)
            if value:
                return value
        required_names = " or ".join(env_names)
        raise RuntimeError(f"{required_names} is required for {provider_name} models")

    if provider == "openai":
        key = _resolve_api_key("openai", "OPENAI_API_KEY")
        provider_obj = _provider_with_http_client(OpenAIProvider, "openai", api_key=key)
        settings: OpenAIChatModelSettings | None = None
        if _openai_model_supports_reasoning_effort(str(model_spec)):
            settings = OpenAIChatModelSettings(openai_reasoning_effort="medium")
        return OpenAIChatModel(model_spec, provider=provider_obj), settings

    if provider == "azure":
        endpoint = model_entry.get("azure_endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = model_entry.get("api_version") or os.getenv("OPENAI_API_VERSION")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for azure models")
        azure_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        provider_obj = _provider_with_http_client(
            AzureProvider,
            "azure",
            azure_endpoint=endpoint,
            api_version=api_version,
            api_key=azure_key,
        )
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "alibaba":
        key = _resolve_api_key("alibaba", "ALIBABA_API_KEY", "DASHSCOPE_API_KEY")
        provider_obj = _provider_with_http_client(AlibabaProvider, "alibaba", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "openai-codex":
        if not has_openai_tokens():
            raise RuntimeError("OpenAI Codex OAuth tokens not found. Run /login openai first.")
        http_client = _get_http_client(
            "openai-codex",
            event_hooks={"request": [openai_auth_request_hook]},
            client_cls=OpenAICodexAsyncClient,
        )
        provider_obj = _provider_with_http_client(
            OpenAIProvider,
            "openai-codex",
            api_key="oauth",
            base_url=OPENAI_CODEX_BASE_URL,
            http_client=http_client,
        )
        oauth_settings: OpenAIResponsesModelSettings | None = None
        if _openai_model_supports_reasoning_effort(str(model_spec)):
            oauth_settings = OpenAIResponsesModelSettings(openai_reasoning_effort="medium")
        return OpenAIResponsesModel(model_spec, provider=provider_obj), oauth_settings

    if provider == "anthropic":
        key = _resolve_api_key("anthropic", "ANTHROPIC_API_KEY")
        provider_obj = _provider_with_http_client(AnthropicProvider, "anthropic", api_key=key)
        settings = AnthropicModelSettings(anthropic_thinking={"type": "enabled", "budget_tokens": 512})
        return AnthropicModel(model_spec, provider=provider_obj), settings

    if provider == "bedrock":
        provider_obj = _provider_with_http_client(BedrockProvider, "bedrock", api_key=api_key)
        return BedrockConverseModel(str(model_spec), provider=provider_obj), None

    if provider == "cohere":
        key = _resolve_api_key("cohere", "CO_API_KEY")
        provider_obj = _provider_with_http_client(CohereProvider, "cohere", api_key=key)
        return CohereModel(str(model_spec), provider=provider_obj), None

    if provider == "deepseek":
        key = _resolve_api_key("deepseek", "DEEPSEEK_API_KEY")
        provider_obj = _provider_with_http_client(DeepSeekProvider, "deepseek", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "fireworks":
        key = _resolve_api_key("fireworks", "FIREWORKS_API_KEY")
        provider_obj = _provider_with_http_client(FireworksProvider, "fireworks", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "github":
        key = _resolve_api_key("github", "GITHUB_API_KEY")
        provider_obj = _provider_with_http_client(GitHubProvider, "github", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "mistral":
        key = _resolve_api_key("mistral", "MISTRAL_API_KEY")
        provider_obj = _provider_with_http_client(MistralProvider, "mistral", api_key=key)
        return MistralModel(model_spec, provider=provider_obj), None

    if provider == "cerebras":
        key = _resolve_api_key("cerebras", "CEREBRAS_API_KEY")
        provider_obj = _provider_with_http_client(CerebrasProvider, "cerebras", api_key=key)
        return CerebrasModel(model_spec, provider=provider_obj), None

    if provider == "google":
        key = _resolve_api_key("google", "GOOGLE_API_KEY", "GEMINI_API_KEY")
        provider_obj = _provider_with_http_client(GoogleProvider, "google", api_key=key)
        settings = GoogleModelSettings(google_thinking_config={"include_thoughts": True})
        return GoogleModel(model_spec, provider=provider_obj), settings

    if provider == "google-vertex":
        provider_obj = _provider_with_http_client(GoogleVertexProvider, "google-vertex")
        settings = GoogleModelSettings(google_thinking_config={"include_thoughts": True})
        return GoogleModel(str(model_spec), provider=provider_obj), settings

    if provider == "groq":
        key = _resolve_api_key("groq", "GROQ_API_KEY")
        provider_obj = _provider_with_http_client(GroqProvider, "groq", api_key=key)
        return GroqModel(str(model_spec), provider=provider_obj), None

    if provider == "huggingface":
        key = _resolve_api_key("huggingface", "HF_TOKEN")
        provider_obj = _provider_with_http_client(HuggingFaceProvider, "huggingface", api_key=key)
        return HuggingFaceModel(str(model_spec), provider=provider_obj), None

    if provider == "moonshotai":
        key = _resolve_api_key("moonshotai", "MOONSHOTAI_API_KEY")
        provider_obj = _provider_with_http_client(MoonshotAIProvider, "moonshotai", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "nebius":
        key = _resolve_api_key("nebius", "NEBIUS_API_KEY")
        provider_obj = _provider_with_http_client(NebiusProvider, "nebius", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "code-assist":
        return CodeAssistModel(str(model_spec)), None

    if provider == "openrouter":
        key = _resolve_api_key("openrouter", "OPENROUTER_API_KEY")
        provider_obj = _provider_with_http_client(OpenRouterProvider, "openrouter", api_key=key)
        settings = OpenRouterModelSettings(openrouter_reasoning={"effort": "medium"})
        return OpenRouterModel(model_spec, provider=provider_obj), settings

    if provider == "ovhcloud":
        key = _resolve_api_key("ovhcloud", "OVHCLOUD_API_KEY")
        provider_obj = _provider_with_http_client(OVHcloudProvider, "ovhcloud", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "together":
        key = _resolve_api_key("together", "TOGETHER_API_KEY")
        provider_obj = _provider_with_http_client(TogetherProvider, "together", api_key=key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "vercel":
        provider_obj = _provider_with_http_client(VercelProvider, "vercel", api_key=api_key)
        return OpenAIChatModel(str(model_spec), provider=provider_obj), None

    if provider == "xai":
        key = _resolve_api_key("xai", "XAI_API_KEY")
        provider_obj = _provider_with_http_client(XaiProvider, "xai", api_key=key)
        return XaiModel(str(model_spec), provider=provider_obj), None

    if provider == "ollama":
        # Force JSON mode so local models emit parseable tool payloads.
        provider_obj = _provider_with_http_client(OllamaProvider, "ollama", base_url=OLLAMA_BASE_URL)
        return OpenAIChatModel(model_spec, provider=provider_obj), None

    if provider == "function":
        return TestModel(call_tools=[]), None

    # default to test or direct spec
    return model_spec, None
