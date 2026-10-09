"""Compute-device auto-detection for local ML models.

Resolves the configured device (``auto`` by default) to the best backend
available on the host, so the same code runs unchanged on CPU-only machines,
NVIDIA CUDA hosts, and Apple Silicon (Metal/MPS, when run natively).
"""

from __future__ import annotations

from functools import lru_cache

from knowledge_assistant.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def resolve_device(device: str | None) -> str:
    """Resolve ``auto`` to the best backend: cuda > mps > cpu.

    An explicit device (``cpu``/``cuda``/``mps``) is returned unchanged.
    """
    if device and device != "auto":
        return device
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is always present in practice
        return "cpu"

    if torch.cuda.is_available():
        resolved = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        resolved = "mps"
    else:
        resolved = "cpu"
    logger.info("device_resolved", requested=device, resolved=resolved)
    return resolved
