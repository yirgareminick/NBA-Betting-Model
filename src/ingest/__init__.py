try:
    from .ingest_games_new import NBADataIngestion
except Exception:  # pragma: no cover - optional dependency fallback
    NBADataIngestion = None

__all__ = ["NBADataIngestion"]
