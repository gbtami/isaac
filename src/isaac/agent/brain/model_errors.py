"""Shared model errors for prompt handling."""

from __future__ import annotations


class ModelBuildError(RuntimeError):
    """Raised when a model fails to build for a session."""
