import logging
from pathlib import Path


def setup_logger(log_file: Path = None, level=logging.INFO) -> logging.Logger:
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

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create formatter and add it to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)

    # Optional: Add file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
