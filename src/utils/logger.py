"""
Utility Functions Module
"""
from loguru import logger
import sys
import os


def setup_logger(log_level: str = "INFO", log_file: str = "./logs/rag_system.log"):
    """
    Configures the logging system.

    Args:
        log_level: The logging level.
        log_file: The path to the log file.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Remove the default logger
    logger.remove()

    # Add a console logger
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )

    # Add a file logger
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",  # Rotate when the file size reaches 10 MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        encoding="utf-8"
    )

    logger.info(f"Logger initialized: level={log_level}, file={log_file}")


def print_banner():
    """Prints the system banner."""
    banner = """
    ╔═════════════════════════════════╗
    ║        tiny RAG System          ║
    ╚═════════════════════════════════╝
    """
    print(banner)
