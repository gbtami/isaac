#!/usr/bin/env python3
"""Refresh offline models.dev snapshot and regenerate isaac model catalog.

This script downloads `https://models.dev/api.json`, stores the full payload in
the repository, and derives a compact top-model catalog per provider that isaac
can use offline for model selection.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODELS_DEV_URL = "https://models.dev/api.json"
USER_AGENT = "isaac-acp/0.1 (+https://github.com/gbtami/isaac)"

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_FILE = REPO_ROOT / "src" / "isaac" / "agent" / "data" / "models_dev_api.json"
CATALOG_FILE = REPO_ROOT / "src" / "isaac" / "agent" / "data" / "models_dev_catalog.json"

EXCLUDED_MODEL_TERMS = (
    "embedding",
    "rerank",
    "moderation",
    "transcribe",
    "whisper",
    "tts",
    "speech",
)


@dataclass(frozen=True)
class ProviderMapping:
    target_provider: str
    source_provider: str
    label: str
    preferred_substring: str | None = None
    excluded_substring: str | None = None


PROVIDER_MAPPINGS: tuple[ProviderMapping, ...] = (
    ProviderMapping("openai", "openai", "OpenAI", excluded_substring="codex"),
    ProviderMapping("openai-codex", "openai", "OpenAI Codex (OAuth)", preferred_substring="codex"),
    ProviderMapping("azure", "azure", "Azure OpenAI"),
    ProviderMapping("alibaba", "alibaba", "Alibaba"),
    ProviderMapping("bedrock", "amazon-bedrock", "Amazon Bedrock"),
    ProviderMapping("anthropic", "anthropic", "Anthropic"),
    ProviderMapping("cohere", "cohere", "Cohere"),
    ProviderMapping("deepseek", "deepseek", "DeepSeek"),
    ProviderMapping("fireworks", "fireworks-ai", "Fireworks AI"),
    ProviderMapping("github", "github-models", "GitHub Models"),
    ProviderMapping("google", "google", "Google", preferred_substring="gemini"),
    ProviderMapping("google-vertex", "google-vertex", "Google Vertex"),
    ProviderMapping("code-assist", "google", "Google Code Assist (OAuth)", preferred_substring="gemini"),
    ProviderMapping("groq", "groq", "Groq"),
    ProviderMapping("huggingface", "huggingface", "Hugging Face"),
    ProviderMapping("mistral", "mistral", "Mistral"),
    ProviderMapping("moonshotai", "moonshotai", "Moonshot AI"),
    ProviderMapping("nebius", "nebius", "Nebius"),
    ProviderMapping("openrouter", "openrouter", "OpenRouter"),
    ProviderMapping("ovhcloud", "ovhcloud", "OVHcloud"),
    ProviderMapping("together", "togetherai", "Together"),
    ProviderMapping("cerebras", "cerebras", "Cerebras"),
    ProviderMapping("vercel", "vercel", "Vercel AI Gateway"),
    ProviderMapping("xai", "xai", "xAI"),
)


def _download_models_dev() -> dict[str, Any]:
    req = urllib.request.Request(MODELS_DEV_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # type: ignore[call-arg]
        payload = resp.read().decode("utf-8")
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise ValueError("Unexpected models.dev payload shape (expected provider dictionary).")
    return raw


def _parse_date(model_meta: dict[str, Any]) -> int:
    for key in ("last_updated", "release_date"):
        value = model_meta.get(key)
        if not isinstance(value, str) or not value:
            continue
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    return 0


def _context_limit(model_meta: dict[str, Any]) -> int:
    limit = model_meta.get("limit")
    if not isinstance(limit, dict):
        return 0
    context = limit.get("context")
    return int(context) if isinstance(context, (int, float)) and context > 0 else 0


def _is_text_capable(model_meta: dict[str, Any]) -> bool:
    modalities = model_meta.get("modalities")
    if not isinstance(modalities, dict):
        return True
    inputs = modalities.get("input")
    outputs = modalities.get("output")
    if isinstance(inputs, list) and inputs and "text" not in inputs:
        return False
    if isinstance(outputs, list) and outputs and "text" not in outputs:
        return False
    return True


def _is_model_candidate(model_id: str, model_meta: dict[str, Any]) -> bool:
    model_id_lc = model_id.lower()
    if any(term in model_id_lc for term in EXCLUDED_MODEL_TERMS):
        return False
    return _is_text_capable(model_meta)


def _rank_model(model_id: str, model_meta: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        1 if bool(model_meta.get("tool_call")) else 0,
        1 if bool(model_meta.get("structured_output")) else 0,
        1 if bool(model_meta.get("reasoning")) else 0,
        _parse_date(model_meta),
        _context_limit(model_meta),
    )


def _sorted_candidates(models: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for model_id, raw_meta in models.items():
        if not isinstance(raw_meta, dict):
            continue
        if not _is_model_candidate(model_id, raw_meta):
            continue
        candidates.append((model_id, raw_meta))
    candidates.sort(key=lambda item: (_rank_model(item[0], item[1]), item[0]), reverse=True)
    return candidates


def _select_top_models(
    models: dict[str, Any],
    max_models: int,
    preferred_substring: str | None,
    excluded_substring: str | None,
) -> list[tuple[str, dict[str, Any]]]:
    ranked = _sorted_candidates(models)
    preferred = preferred_substring.lower() if preferred_substring else ""
    excluded = excluded_substring.lower() if excluded_substring else ""
    if excluded:
        ranked = [(model_id, meta) for model_id, meta in ranked if excluded not in model_id.lower()]
    selected: list[tuple[str, dict[str, Any]]] = []
    if preferred:
        for model_id, meta in ranked:
            if preferred in model_id.lower():
                selected.append((model_id, meta))
            if len(selected) >= max_models:
                break
    if len(selected) >= max_models:
        return selected[:max_models]

    seen = {model_id for model_id, _meta in selected}
    for model_id, meta in ranked:
        if model_id in seen:
            continue
        selected.append((model_id, meta))
        if len(selected) >= max_models:
            break
    return selected[:max_models]


def _build_catalog(
    raw: dict[str, Any],
    top_per_provider: int,
) -> dict[str, Any]:
    providers: dict[str, Any] = {}
    for mapping in PROVIDER_MAPPINGS:
        source_entry = raw.get(mapping.source_provider, {})
        source_models = source_entry.get("models", {}) if isinstance(source_entry, dict) else {}
        if not isinstance(source_models, dict):
            source_models = {}
        selected = _select_top_models(
            source_models,
            top_per_provider,
            mapping.preferred_substring,
            mapping.excluded_substring,
        )
        providers[mapping.target_provider] = {
            "label": mapping.label,
            "source_provider": mapping.source_provider,
            "models": [
                {
                    "id": model_id,
                    "name": str(meta.get("name") or model_id),
                    "context_limit": _context_limit(meta) or None,
                    "release_date": meta.get("release_date"),
                    "last_updated": meta.get("last_updated"),
                }
                for model_id, meta in selected
            ],
        }
    providers["ollama"] = {
        "label": "Ollama",
        "source_provider": None,
        "models": [
            {
                "id": "ministral-3:3b",
                "name": "ministral-3:3b",
                "context_limit": None,
                "release_date": None,
                "last_updated": None,
            }
        ],
    }
    return {
        "source_url": MODELS_DEV_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_per_provider": top_per_provider,
        "providers": providers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top-per-provider",
        type=int,
        default=8,
        help="Maximum number of models to keep per supported provider (default: 8).",
    )
    args = parser.parse_args()

    if args.top_per_provider < 1 or args.top_per_provider > 20:
        raise SystemExit("--top-per-provider must be between 1 and 20.")

    raw = _download_models_dev()
    catalog = _build_catalog(raw, args.top_per_provider)

    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(json.dumps(raw, indent=2, sort_keys=True), encoding="utf-8")
    CATALOG_FILE.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    print(f"wrote {SNAPSHOT_FILE}")
    print(f"wrote {CATALOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
