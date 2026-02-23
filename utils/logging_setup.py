"""Logging configuration for multiview-CRL training."""

import logging
import os
from datetime import datetime


def setup_logging(save_dir: str, log_level: int = logging.INFO) -> logging.Logger:
    """
    Configure a logger that writes to both stdout and a timestamped file.

    Args:
        save_dir: Directory where the log file will be written.
                  If the directory does not exist yet, file logging is skipped.
        log_level: Python logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger named ``'multiview_crl'``.
    """
    logger = logging.getLogger("multiview_crl")
    logger.setLevel(log_level)
    logger.handlers = []  # Clear any handlers added by a previous call

    detailed_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_formatter = logging.Formatter("%(levelname)-8s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if save_dir and os.path.exists(save_dir):
        log_file = os.path.join(
            save_dir,
            f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    return logger
