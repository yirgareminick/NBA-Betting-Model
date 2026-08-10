import logging

DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"


def setup_basic_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure basic logging for script entry points."""
    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
    return logging.getLogger()
