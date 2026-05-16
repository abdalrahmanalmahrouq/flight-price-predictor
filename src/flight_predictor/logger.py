import sys
from loguru import logger


def setup_logger(level: str = "INFO") -> None:
    # Remove the default loguru handler
    logger.remove()

    # Console handler — clean format for terminal
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True,
    )

    # File handler — full debug logs with rotation
    logger.add(
        "logs/app.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",       # new file when current hits 10MB
        retention="7 days",     # delete logs older than 7 days
        compression="zip",      # compress rotated files
        colorize=False,         # no color codes in file
    )


# Export the configured logger
__all__ = ["logger", "setup_logger"]
