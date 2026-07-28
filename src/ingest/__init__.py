try:
    from .ingest_games_new import NBADataIngestion
except Exception as exc:  # pragma: no cover - optional dependency fallback
    print(f"⚠️  Could not import NBADataIngestion: {exc}")
    NBADataIngestion = None

__all__ = ["NBADataIngestion"]
