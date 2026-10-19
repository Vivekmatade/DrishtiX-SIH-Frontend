import logging
import sys

def get_logger(name: str = "ai_feature_extraction", level: int = logging.INFO) -> logging.Logger:
    """Returns a configured logger with clean standardized formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger
