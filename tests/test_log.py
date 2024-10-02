import logging
import pytest
from src.log import setup_logger


def test_logger_initialization(caplog):
    """
    Test that the logger initializes correctly and logs to the console.
    """
    with caplog.at_level(logging.INFO):
        logger = setup_logger()
        logger.info("This is a test log message.")

    # Check that the message was logged
    assert "This is a test log message." in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"


def test_logger_with_file(tmp_path):
    """
    Test that the logger writes to a file as expected.
    """
    # Create a temporary log file using tmp_path
    log_file = tmp_path / "test_log_file.log"
    logger = setup_logger(log_file=log_file, level=logging.INFO)

    # Log a message
    logger.info("Logging to a file!")

    # Ensure the log file exists
    assert log_file.exists()

    # Read the log file and check contents
    with log_file.open() as f:
        logs = f.read()

    assert "Logging to a file!" in logs


@pytest.mark.parametrize("log_level", [logging.DEBUG, logging.INFO, logging.WARNING])
def test_logger_levels(caplog, log_level):
    """
    Test that different log levels work as expected.
    """
    with caplog.at_level(log_level):
        logger = setup_logger(level=log_level)
        logger.log(log_level, f"This is a {logging.getLevelName(log_level)} log")

    # Check that the message was logged at the correct level
    assert f"This is a {logging.getLevelName(log_level)} log" in caplog.text
    assert caplog.records[0].levelno == log_level
