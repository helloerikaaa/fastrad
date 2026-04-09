import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add an awesome colored stdout handler.
# The user approved keeping INFO logs out of the box because it's beautiful.
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Export our customized logger
__all__ = ["logger"]
