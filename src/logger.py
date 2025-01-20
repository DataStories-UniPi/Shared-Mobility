import sys
from pathlib import Path

from loguru import logger


def configure_logger(
    log_dir: str = "logs",
    log_file: str = "app.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove the default handler
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}:{line}</cyan> | <level>{message}</level>",
    )

    # Info-level file sink
    logger.add(
        sink=Path(log_dir) / f"{log_file}-info.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Error-level file sink
    logger.add(
        sink=Path(log_dir) / f"{log_file}-error.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message} | {exception}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # Custom method for logging parameters and metadata
    def log_experiment_params(params):
        """
        Logs experiment parameters and metadata.

        Parameters:
        - params (dict): Dictionary of parameters to log.
        """
        logger.info("Experiment Parameters:")
        for key, value in params.items():
            logger.info(f"{key}: {value}")

    # Add the custom method to the logger
    logger.log_experiment_params = log_experiment_params

    logger.info("Logger has been configured.")
    return logger
