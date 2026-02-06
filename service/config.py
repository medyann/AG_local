from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    # AlphaGenome / HF
    hf_model_handle: str = os.getenv("ALPHAGENOME_HF_HANDLE", "all_folds")
    hf_token_env: str = os.getenv("ALPHAGENOME_HF_TOKEN_ENV", "HF_TOKEN")
    hf_home: str | None = os.getenv("HF_HOME")

    # Server behavior
    backend: str = os.getenv("ALPHAGENOME_BACKEND", "alphagenome")  # alphagenome|mock
    eager_load: bool = _env_bool("ALPHAGENOME_EAGER_LOAD", False)

    # Response shaping defaults
    default_window_bp: int = int(os.getenv("ALPHAGENOME_DEFAULT_WINDOW_BP", str(2**15)))
    default_downsample_to: int = int(os.getenv("ALPHAGENOME_DEFAULT_DOWNSAMPLE_TO", "2048"))


settings = Settings()

