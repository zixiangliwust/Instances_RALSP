# -*- coding: utf-8 -*-
"""Logging configuration module for the evolu framework.
 
This module provides logging configuration and logger retrieval functions.
It configures Python's logging system to output messages to the console with
a standardized format that includes timestamp, module name, log level, and message.
"""

import logging.config
from typing import Union


def configure_logging() -> None:
    """Configure logging settings for the evolu framework.
    
    This function sets up the Python logging system with:
    - Console output to stderr
    - Standardized format: [timestamp] [logger_name] [level] message
    - DEBUG level for evolu logger (can be changed as needed)
    - Existing loggers are not disabled
    
    The logging configuration is applied globally and should be called once
    at the start of the application. It's automatically called when importing
    the evolu package.
    
    Note:
        Users can override this configuration by calling logging.config.dictConfig()
        with their own configuration after importing evolu.
    
    Example:
        >>> configure_logging()
        >>> logger = get_logger(__name__)
        >>> logger.info("Logging is now configured")
        [2024-01-16 10:30:45,123] [__main__] [INFO] Logging is now configured
    """
    DEFAULT_LOGGING_CONFIG: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "basic": {
                "format": "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            }
        },
        "handlers": {
            "console": {
                "formatter": "basic",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            }
        },
        "loggers": {
            "evolu": {"handlers": ["console"], "level": "DEBUG"},
        },
    }
    logging.config.dictConfig(DEFAULT_LOGGING_CONFIG)


def get_logger(module: Union[str, object]) -> logging.Logger:
    """Get a logger instance for the specified module.
    
    This function creates or retrieves a logger with an appropriate name.
    If a string is provided, it's used directly as the logger name. If an object
    (typically a module) is provided, its __name__ attribute is used.
    
    Args:
        module (Union[str, object]): Module name (string) or module object.
            If an object is provided, its __name__ attribute is used.
    
    Returns:
        logging.Logger: Logger instance configured for the specified module.
    
    Example:
        >>> # Using module name as string
        >>> logger = get_logger("evolu.algorithm")
        >>> logger.info("Algorithm started")
        
        >>> # Using module object
        >>> import evolu.core.algorithm as algorithm_module
        >>> logger = get_logger(algorithm_module)
        >>> logger.info("Algorithm started")
        
        >>> # Using __name__ in a module file
        >>> logger = get_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(module if isinstance(module, str) else module.__name__)
