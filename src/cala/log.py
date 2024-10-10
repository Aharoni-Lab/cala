import logging
from pathlib import Path
from rich.logging import RichHandler


def setup_logger(
    log_file: Path = None, level=logging.INFO, name: str = "cala.log"
) -> logging.Logger:
    """
    Sets up the logging configuration for the application.

    Args:
        log_file (Path): Optional path to a log file where logs will be saved.
        level (int): Logging level (INFO, DEBUG, etc.).

    Returns:
        logging.Logger: Configured logger instance.
    """
    if log_file is not None:
        log_file.parent.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(level)

    # Create formatter and add it to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    rich_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(rich_handler)

    # Optional: Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
