from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from service.config import settings


class AlphaGenomeNotConfigured(RuntimeError):
    pass


BackendName = Literal["alphagenome", "mock"]


def _get_hf_token() -> str | None:
    return os.getenv(settings.hf_token_env) or os.getenv("HUGGINGFACE_HUB_TOKEN")


@dataclass
class LoadedModel:
    backend: BackendName
    handle: str | None
    model: Any


class ModelManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: LoadedModel | None = None

    def backend(self) -> BackendName:
        b = settings.backend.strip().lower()
        return "mock" if b == "mock" else "alphagenome"

    def loaded(self) -> bool:
        return self._loaded is not None

    def get(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        with self._lock:
            if self._loaded is not None:
                return self._loaded
            self._loaded = self._load()
            return self._loaded

    def _load(self) -> LoadedModel:
        backend = self.backend()
        if backend == "mock":
            return LoadedModel(backend="mock", handle=None, model=_MockAlphaGenome())

        token = _get_hf_token()
        if not token:
            raise AlphaGenomeNotConfigured(
                f"Missing Hugging Face token. Set ${settings.hf_token_env} (or $HUGGINGFACE_HUB_TOKEN) after accepting the model terms."
            )

        # Import lazily so server can boot without heavy deps.
        from alphagenome_research.model import dna_model  # type: ignore

        # Hugging Face token is picked up from env by huggingface_hub underneath.
        model = dna_model.create_from_huggingface(settings.hf_model_handle)
        return LoadedModel(backend="alphagenome", handle=settings.hf_model_handle, model=model)


class _MockAlphaGenome:
    """
    A tiny deterministic mock to let the API run without large dependencies.
    """

    def predict_variant(self, **kwargs: Any) -> Any:
        # Mimic the shape: return an object with .reference.<output> and .alternate.<output>
        requested_outputs = kwargs.get("requested_outputs") or []
        rng = np.random.default_rng(0)

        class _RefAlt:
            def __init__(self) -> None:
                self.interval = kwargs.get("interval")

        class _Out:
            def __init__(self) -> None:
                self.interval = kwargs.get("interval")
                self.values = rng.normal(size=(4096, 1)).astype(np.float32)

        ref = _RefAlt()
        alt = _RefAlt()
        for name in requested_outputs:
            setattr(ref, name.lower(), _Out())
            setattr(alt, name.lower(), _Out())

        class _Resp:
            def __init__(self) -> None:
                self.reference = ref
                self.alternate = alt

        return _Resp()


model_manager = ModelManager()

