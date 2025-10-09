import sys

from loguru import logger

from config.paths import LOG_DIR


def configure_logger(
    log_file: str = "app",
    rotation: str = "100 MB",
    retention: str = "7 days",
):
    """
    Configures the application logger with various sinks and settings.

    Sets up logging to console and files with different levels of severity
    and formats. Creates log directory if it doesn't exist. Removes default
    logger handlers and adds customized ones for stdout and log files with
    specified rotation and retention policies.

    Parameters
    ----------
    log_dir : str, optional
        Directory where log files will be stored. Default is "logs".
    log_file : str, optional
        Base name of the log files. Default is "app.log".
    rotation : str, optional
        Rotation policy for log files. Default is "10 MB".
    retention : str, optional
        Retention policy for log files. Default is "7 days".

    Returns
    -------
    logger
        Configured loguru logger instance.
    """

    logger.remove()  # Remove the default handler
    logger.add(
        sys.stdout,
        level="DEBUG",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{module}:{line}</cyan> | <level>{message}</level>",
    )

    # File sink
    logger.add(
        sink=LOG_DIR / f"{log_file}.log",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
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
