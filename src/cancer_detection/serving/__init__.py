# Lazy imports — do not eagerly load api.py here, as it registers a FastAPI lifespan
# which triggers model loading. Import directly: from cancer_detection.serving.api import app

__all__ = ["app"]
