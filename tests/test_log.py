import logging
from pathlib import Path

import pytest

from cala.log import setup_logger


@pytest.fixture
def log_file(tmp_path: Path):
    """
    Pytest fixture to create a temporary log file.
    """
    return tmp_path / "test.log"


def test_logger_initialization(caplog):
    """
    Test that the logger initializes correctly and logs to the console.
    """
    with caplog.at_level(logging.INFO):
        logger = setup_logger()
        logger.info("This is a test log message.")

    assert "This is a test log message." in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"


@pytest.mark.parametrize(
    "level, level_name",
    [
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRITICAL"),
    ],
)
def test_logger_with_all_levels(caplog, level, level_name):
    """
    Test logger setup with different logging levels.
    """
    logger = setup_logger(level=level)

    with caplog.at_level(level):
        logger.log(level, f"Testing {level_name} level")

    assert f"Testing {level_name} level" in caplog.text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.levelname == level_name


def test_logger_with_file_handler(log_file: Path) -> None:
    """
    Test logger setup with file handler.
    """
    logger = setup_logger(log_path=log_file)

    logger.info("Log message to file")

    assert log_file.is_dir()

    # Check if the log message is written to the file
    for handler in logger.handlers:
        if file_path := getattr(handler, "baseFilename", None):
            with open(file_path) as f:
                logs = f.read()

    assert "Log message to file" in logs
    assert "INFO" in logs


def test_logger_with_different_name(caplog):
    """
    Test logger setup with a different logger name.
    """
    logger_name = "custom_logger"
    logger = setup_logger(name=logger_name)

    with caplog.at_level(logging.INFO):
        logger.info("Custom logger test")

    assert "Custom logger test" in caplog.text
    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == logger_name
