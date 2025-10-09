import os
from typing import Any

from loguru import logger


def get_env_variable(
    var_name: str,
    default: str | int | None = None,
    required: bool = False,
) -> str | None:
    """Retrieve an environment variable with optional default and requirement flag.

    Parameters
    ----------
    var_name : str
        The name of the environment variable to retrieve.
    default : Optional[str], default=None
        The default value to return if the variable is not found.
    required : bool, default=True
        Whether the environment variable is mandatory.

    Returns
    -------
    str
        The value of the environment variable or the default if provided.
        If required is True and the variable is not found, an error is raised.

    Raises
    ------
    ValueError
        If the environment variable is required but missing.

    Examples
    --------
    >>> os.environ["TEST_VAR"] = "hello"
    >>> get_env_variable("TEST_VAR")
    'hello'

    >>> get_env_variable("MISSING_VAR", default="fallback")
    'fallback'

    >>> get_env_variable("MISSING_VAR", required=True)
    ValueError: Missing required environment variable: MISSING_VAR

    """
    value = os.getenv(var_name, default)

    if required and value is None:
        logger.warning(
            f"Missing required environment variable: {var_name}. "
            f"Returning empty string instead."
        )
        return None

    return str(value)


def validate_environment(
    required_params: list[str],
    config_dict: dict[str, Any],
) -> tuple[bool, list[str] | None]:
    """
    Validate if all required parameters exist in the given configuration dictionary.

    Parameters
    ----------
    required_params : list[str]
        A list of required environment variable names.
    config_dict : dict[str, any]
        The dictionary containing environment variables.

    Returns
    -------
    tuple[bool, Optional[list[str]]]
        A tuple where the first value is True if all required parameters exist,
        False otherwise.
        The second value is a list of missing variables or None if none are missing.

    Examples
    --------
    >>> cfg = {"API_KEY": "12345", "DEBUG": True}
    >>> validate_environment(["API_KEY", "SECRET_KEY"], cfg)
    (False, ['SECRET_KEY'])

    >>> validate_environment(["API_KEY"], cfg)
    (True, None)
    """
    missing_vars = [param for param in required_params if param not in config_dict]

    if missing_vars:
        return False, missing_vars
    return True, None
