"""Configuration for AlphaGenome web deployment."""

import os
from pydantic import BaseModel


class Settings(BaseModel):
    """Application settings loaded from environment variables."""

    # Hugging Face authentication token (required to download model weights)
    hf_token: str = os.environ.get("HF_TOKEN", "")

    # Model version to load
    model_version: str = os.environ.get("ALPHAGENOME_MODEL_VERSION", "all_folds")

    # Server settings
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))

    # Maximum sequence length (AlphaGenome supports up to 1Mb = 2^20)
    max_sequence_length: int = 1_048_576  # 2**20

    # Default organism
    default_organism: str = os.environ.get("DEFAULT_ORGANISM", "human")

    # Whether to preload the model on startup
    preload_model: bool = os.environ.get("PRELOAD_MODEL", "true").lower() == "true"


settings = Settings()
