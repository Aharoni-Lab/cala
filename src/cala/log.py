import logging
from datetime import datetime
from pathlib import Path

from rich.logging import RichHandler


def setup_logger(
    log_path: Path | str | None = None, level: int = logging.INFO, name: str = "cala"
) -> logging.Logger:
    """
    Sets up the logging configuration for the application.

    Args:
        log_path (Path): Optional path to a log file where logs will be saved.
        level (int): Logging level (INFO, DEBUG, etc.).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(level)

    # Create formatter and add it to handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    rich_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(rich_handler)

    # Add file handler if log_file is provided
    if log_path:
        if isinstance(log_path, str):
            log_path = Path(log_path)
        log_path.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_path / f"{datetime.now().strftime('%Y%m%d_%H-%M-%S')}.log"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
