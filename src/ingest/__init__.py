import logging

logger = logging.getLogger(__name__)

try:
    from .ingest_games_new import NBADataIngestion
except Exception as exc:  # pragma: no cover - optional dependency fallback
    logger.warning("Could not import NBADataIngestion: %s", exc)
    NBADataIngestion = None

__all__ = ["NBADataIngestion"]
