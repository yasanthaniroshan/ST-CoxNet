import logging
import os
from typing import Dict, Optional

import wandb


_logger: Optional[logging.Logger] = None


def _setup_logging() -> logging.Logger:
    """
    Lazily configure logging the first time a logger is requested.
    Avoids side effects at import time.
    """
    global _logger
    if _logger is not None:
        return _logger

    os.makedirs("Logs", exist_ok=True)

    logger = logging.getLogger("st_coxnet")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler("Logs/debug.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    _logger = logger
    return logger


class Logger:
    def __init__(self, run: Optional[wandb.Run] = None):
        self.run = run
        self.logger = _setup_logging()

    def log(self, data: Dict, step: Optional[int] = None):
        if self.run is not None:
            self.run.log(data, step=step)
        # self.logger.info(str(data))

    def info(self, message: str):
        self.logger.info(message)


def get_logger(run: Optional[wandb.Run] = None) -> Logger:
    """
    Return a logger instance, optionally bound to a WandB run.
    """
    return Logger(run)
