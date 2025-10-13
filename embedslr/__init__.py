from importlib import metadata as _m
from .embeddings import get_embeddings, list_models
from .similarity import rank_by_cosine
from .bibliometrics import full_report, indicators
from .colab_app import run as colab_run
from .advanced_scoring import rank_with_advanced_scoring
from .config import ScoringConfig, ColumnMap

try:
    __version__ = _m.version(__name__)
except _m.PackageNotFoundError:
    __version__ = "0.6.0"

__all__ = [
    "get_embeddings", "list_models", "rank_by_cosine",
    "full_report", "indicators", "colab_run",
    "rank_with_advanced_scoring", "ScoringConfig", "ColumnMap"
]
