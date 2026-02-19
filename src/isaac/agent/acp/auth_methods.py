"""Auth method parsing and defaults for ACP initialization."""

from __future__ import annotations

from typing import Any, Iterable, TypeAlias

from acp.schema import AuthMethod

AuthMethodInput: TypeAlias = AuthMethod | dict[str, Any] | str

_ENV_VAR_AUTH_FIELDS: list[tuple[str, str]] = [
    ("OpenRouter", "OPENROUTER_API_KEY"),
    ("OpenAI", "OPENAI_API_KEY"),
    ("Anthropic", "ANTHROPIC_API_KEY"),
    ("Google", "GOOGLE_API_KEY"),
    ("Gemini", "GEMINI_API_KEY"),
    ("Groq", "GROQ_API_KEY"),
    ("Mistral", "MISTRAL_API_KEY"),
    ("Cohere", "CO_API_KEY"),
    ("xAI", "XAI_API_KEY"),
    ("DeepSeek", "DEEPSEEK_API_KEY"),
    ("Together", "TOGETHER_API_KEY"),
    ("GitHub", "GITHUB_API_KEY"),
    ("Hugging Face", "HF_TOKEN"),
    ("MoonshotAI", "MOONSHOTAI_API_KEY"),
    ("Nebius", "NEBIUS_API_KEY"),
    ("Cerebras", "CEREBRAS_API_KEY"),
    ("Alibaba", "ALIBABA_API_KEY"),
    ("DashScope", "DASHSCOPE_API_KEY"),
    ("Fireworks", "FIREWORKS_API_KEY"),
    ("OVHcloud", "OVHCLOUD_API_KEY"),
    ("Azure OpenAI", "AZURE_OPENAI_API_KEY"),
]


def normalize_method_id(method_id: str) -> str:
    return method_id.strip().lower()


def coerce_auth_method(entry: AuthMethodInput) -> AuthMethod:
    if isinstance(entry, AuthMethod):
        return entry
    if isinstance(entry, str):
        token = entry.strip()
        if not token:
            raise ValueError("Authentication method id cannot be empty.")
        if ":" in token:
            method_id, method_name = token.split(":", 1)
            method_id = method_id.strip()
            method_name = method_name.strip()
        else:
            method_id = token
            method_name = token
        if not method_id:
            raise ValueError("Authentication method id cannot be empty.")
        return AuthMethod(id=method_id, name=method_name or method_id)
    method_id = str(entry.get("id", "")).strip()
    method_name = str(entry.get("name", "")).strip() or method_id
    description = entry.get("description")
    field_meta = entry.get("_meta")
    if not isinstance(field_meta, dict):
        field_meta = None
    if not method_id:
        raise ValueError("Authentication method id cannot be empty.")
    return AuthMethod(id=method_id, name=method_name, description=description, _meta=field_meta)


def default_auth_methods() -> list[AuthMethod]:
    agent_methods = [
        AuthMethod(
            id="openai",
            name="OpenAI Codex OAuth",
            description="Authenticate with OpenAI Codex in a browser.",
            _meta={"type": "agent"},
        ),
        AuthMethod(
            id="code-assist",
            name="Code Assist OAuth",
            description="Authenticate with Google Code Assist in a browser.",
            _meta={"type": "agent"},
        ),
    ]
    env_var_methods = [
        AuthMethod(
            id=f"env_var:{env_name.lower()}",
            name=f"{provider_name} API key",
            description=f"Set {env_name} in the agent environment.",
            _meta={
                "type": "env_var",
                "varName": env_name,
                "var_name": env_name,
            },
        )
        for provider_name, env_name in _ENV_VAR_AUTH_FIELDS
    ]
    return [*agent_methods, *env_var_methods]


def find_auth_method(methods: Iterable[AuthMethod], method_id: str) -> AuthMethod | None:
    normalized = normalize_method_id(method_id)
    for method in methods:
        if normalize_method_id(method.id) == normalized:
            return method
    return None


def auth_method_payload(method: AuthMethod) -> dict[str, Any]:
    return method.model_dump(by_alias=True, exclude_none=True)


def auth_method_type(method: AuthMethod) -> str:
    payload = auth_method_payload(method)
    field_meta = payload.get("_meta")
    method_meta = field_meta if isinstance(field_meta, dict) else {}
    return str(method_meta.get("type") or "agent").strip().lower()


def auth_method_env_var_name(method: AuthMethod) -> str | None:
    payload = auth_method_payload(method)
    field_meta = payload.get("_meta")
    method_meta = field_meta if isinstance(field_meta, dict) else {}
    env_name = str(method_meta.get("varName") or method_meta.get("var_name") or "").strip()
    return env_name or None
