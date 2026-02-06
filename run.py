"""Entry point for AlphaGenome Web Deployment."""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level="info",
        workers=1,  # Single worker needed for shared model state
    )
