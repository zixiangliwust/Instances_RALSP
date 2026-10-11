"""Core module for evolu framework.

This module provides the fundamental building blocks of the evolu framework:

- Algorithm base classes: Base classes for all optimization algorithms
- Problem base classes: Base classes for defining optimization problems
- Solution classes: Representations of candidate solutions
- Operator base classes: Base classes for genetic operators
- Observer pattern: Observable and Observer interfaces
- Quality indicators: Metrics for evaluating solution sets
- Exceptions: Custom exception hierarchy for framework errors
- Error Handling: Unified error handling mechanism with recovery strategies

All core components follow a modular, extensible design that allows for easy
customization and extension to new problem types and algorithms.
"""

from evolu.core.exceptions import (
    EvoluException,
    InvalidParentsException,
    InvalidParameterException,
    ReferenceFrontNotSetException,
    InvalidRankException,
    InvalidVariantException,
    InvalidSolutionException,
    EmptyFrontException,
    InvalidFigureException,
    InvalidDimensionException,
)
from evolu.core.error_handler import (
    ErrorCode,
    ErrorSeverity,
    ErrorRecoveryStrategy,
    ErrorHandler,
    get_error_handler,
    set_error_handler,
)

__all__ = [
    "EvoluException",
    "InvalidParentsException",
    "InvalidParameterException",
    "ReferenceFrontNotSetException",
    "InvalidRankException",
    "InvalidVariantException",
    "InvalidSolutionException",
    "EmptyFrontException",
    "InvalidFigureException",
    "InvalidDimensionException",
    "ErrorCode",
    "ErrorSeverity",
    "ErrorRecoveryStrategy",
    "ErrorHandler",
    "get_error_handler",
    "set_error_handler",
]

