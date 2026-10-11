"""Unified error handling mechanism for the evolu framework.

This module provides a comprehensive error handling system including:
- Error code enumeration for consistent error categorization
- Error recovery strategies for common error scenarios
- Unified error handler for centralized exception management

The error handling system is designed to:
- Provide consistent error reporting across the framework
- Enable automatic error recovery where possible
- Support error logging and tracking
- Facilitate debugging and troubleshooting
"""
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type
import traceback

from evolu.core.exceptions import EvoluException
from evolu.logger import get_logger

logger = get_logger(__name__)

# Import friendly error provider for enhanced error messages
try:
    from evolu.util.friendly_errors import get_friendly_error_provider, format_error
    _FRIENDLY_ERRORS_AVAILABLE = True
except ImportError:
    _FRIENDLY_ERRORS_AVAILABLE = False


class ErrorCode(Enum):
    """Error codes for categorizing different types of errors.
    
    Error codes are organized by category to enable hierarchical error handling
    and consistent error reporting across the framework.
    
    Categories:
        VALIDATION: Parameter and input validation errors
        EXECUTION: Runtime execution errors
        RESOURCE: Resource-related errors (file I/O, memory, etc.)
        ALGORITHM: Algorithm-specific errors
        PROBLEM: Problem-specific errors
    """
    
    # Validation Errors (1000-1999)
    INVALID_PARAMETER = 1001
    INVALID_PROBABILITY = 1002
    INVALID_RANGE = 1003
    NONE_PARAMETER = 1004
    INVALID_TYPE = 1005
    EMPTY_COLLECTION = 1006
    INVALID_CONDITION = 1007
    
    # Execution Errors (2000-2999)
    INVALID_PARENTS = 2001
    INVALID_SOLUTION = 2002
    INVALID_RANK = 2003
    INVALID_VARIANT = 2004
    EMPTY_FRONT = 2005
    INVALID_DIMENSION = 2006
    INVALID_FIGURE = 2007
    
    # Resource Errors (3000-3999)
    FILE_NOT_FOUND = 3001
    FILE_READ_ERROR = 3002
    FILE_WRITE_ERROR = 3003
    MEMORY_ERROR = 3004
    IO_ERROR = 3005
    
    # Algorithm Errors (4000-4999)
    ALGORITHM_INIT_ERROR = 4001
    ALGORITHM_EXECUTION_ERROR = 4002
    CONVERGENCE_ERROR = 4003
    POPULATION_ERROR = 4004
    
    # Problem Errors (5000-5999)
    PROBLEM_INIT_ERROR = 5001
    EVALUATION_ERROR = 5002
    CONSTRAINT_VIOLATION = 5003
    REFERENCE_FRONT_NOT_SET = 5004
    
    # Unknown/General Errors (9000-9999)
    UNKNOWN_ERROR = 9001
    SYSTEM_ERROR = 9002


class ErrorSeverity(Enum):
    """Error severity levels for prioritizing error handling.
    
    Attributes:
        LOW: Non-critical errors that may be ignored or logged
        MEDIUM: Errors that should be handled but don't stop execution
        HIGH: Critical errors that require immediate attention
        CRITICAL: Fatal errors that prevent continued execution
    """
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies.
    
    Recovery strategies define how to handle specific errors and attempt
    to recover from them. Subclasses should implement specific recovery logic.
    """
    
    def __init__(self, error_code: ErrorCode, recovery_func: Optional[Callable] = None) -> None:
        """Initialize recovery strategy.
        
        Args:
            error_code (ErrorCode): The error code this strategy handles.
            recovery_func (Optional[Callable]): Function to execute for recovery.
                Should accept the exception and return a recovery result.
        """
        self.error_code: ErrorCode = error_code
        self.recovery_func: Optional[Callable] = recovery_func
    
    def can_recover(self, exception: Exception) -> bool:
        """Check if this strategy can recover from the given exception.
        
        Args:
            exception (Exception): The exception to check.
        
        Returns:
            bool: True if this strategy can handle the exception.
        """
        return isinstance(exception, EvoluException)
    
    def recover(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Any:
        """Attempt to recover from the exception.
        
        Args:
            exception (Exception): The exception to recover from.
            context (Optional[Dict[str, Any]]): Additional context for recovery.
        
        Returns:
            Any: Recovery result, or None if recovery fails.
        
        Raises:
            Exception: If recovery is not possible or fails.
        """
        if self.recovery_func:
            try:
                return self.recovery_func(exception, context or {})
            except Exception as e:
                logger.error(f"Recovery function failed: {e}")
                raise
        return None


class DefaultRecoveryStrategies:
    """Default recovery strategies for common error scenarios.
    
    Provides pre-defined recovery strategies for frequently encountered errors.
    """
    
    @staticmethod
    def recover_invalid_probability(exception: Exception, context: Dict[str, Any]) -> float:
        """Recover from invalid probability by clamping to valid range.
        
        Args:
            exception (Exception): The invalid probability exception.
            context (Dict[str, Any]): Context containing the invalid value.
        
        Returns:
            float: Clamped probability value in [0.0, 1.0].
        """
        value = context.get('value', 0.5)
        return max(0.0, min(1.0, float(value)))
    
    @staticmethod
    def recover_empty_collection(exception: Exception, context: Dict[str, Any]) -> list:
        """Recover from empty collection by returning default empty list.
        
        Args:
            exception (Exception): The empty collection exception.
            context (Dict[str, Any]): Context for recovery.
        
        Returns:
            list: Empty list as fallback.
        """
        return []
    
    @staticmethod
    def recover_none_parameter(exception: Exception, context: Dict[str, Any]) -> Any:
        """Recover from None parameter using default value.
        
        Args:
            exception (Exception): The None parameter exception.
            context (Dict[str, Any]): Context containing default value.
        
        Returns:
            Any: Default value from context, or None if not provided.
        """
        return context.get('default', None)


class ErrorHandler:
    """Unified error handler for centralized exception management.
    
    This class provides a centralized error handling mechanism that:
    - Maps exceptions to error codes
    - Attempts automatic recovery using registered strategies
    - Logs errors with appropriate severity levels
    - Provides error context for debugging
    
    Example:
        >>> handler = ErrorHandler()
        >>> handler.register_strategy(
        ...     ErrorCode.INVALID_PROBABILITY,
        ...     DefaultRecoveryStrategies.recover_invalid_probability
        ... )
        >>> try:
        ...     # Some operation that might fail
        ...     pass
        ... except InvalidProbabilityValueException as e:
        ...     result = handler.handle(e, context={'value': 1.5})
    """
    
    def __init__(
        self,
        enable_recovery: bool = True,
        enable_logging: bool = True,
        enable_friendly_messages: bool = True,
        debug_mode: bool = False
    ) -> None:
        """Initialize error handler.
        
        Args:
            enable_recovery (bool): Whether to attempt automatic recovery.
                Default is True.
            enable_logging (bool): Whether to log errors. Default is True.
            enable_friendly_messages (bool): Whether to use friendly error messages.
                Default is True.
            debug_mode (bool): Whether to include debug information in messages.
                Default is False.
        """
        self.enable_recovery: bool = enable_recovery
        self.enable_logging: bool = enable_logging
        self.enable_friendly_messages: bool = enable_friendly_messages
        self.debug_mode: bool = debug_mode
        self.recovery_strategies: Dict[ErrorCode, ErrorRecoveryStrategy] = {}
        self.error_mappings: Dict[Type[Exception], ErrorCode] = {}
        self._initialize_default_mappings()
        self._initialize_default_strategies()
    
    def _initialize_default_mappings(self) -> None:
        """Initialize default exception to error code mappings."""
        from evolu.core.exceptions import (
            InvalidParameterException, InvalidParentsException, InvalidSolutionException,
            InvalidRankException, InvalidVariantException, EmptyFrontException,
            InvalidDimensionException, InvalidFigureException, ReferenceFrontNotSetException
        )
        from evolu.util.checking import (
            InvalidProbabilityValueException, NoneParameterException, EmptyCollectionException,
            ValueOutOfRangeException, InvalidConditionException
        )
        
        self.register_mapping(InvalidParameterException, ErrorCode.INVALID_PARAMETER)
        self.register_mapping(InvalidParentsException, ErrorCode.INVALID_PARENTS)
        self.register_mapping(InvalidSolutionException, ErrorCode.INVALID_SOLUTION)
        self.register_mapping(InvalidRankException, ErrorCode.INVALID_RANK)
        self.register_mapping(InvalidVariantException, ErrorCode.INVALID_VARIANT)
        self.register_mapping(EmptyFrontException, ErrorCode.EMPTY_FRONT)
        self.register_mapping(InvalidDimensionException, ErrorCode.INVALID_DIMENSION)
        self.register_mapping(InvalidFigureException, ErrorCode.INVALID_FIGURE)
        self.register_mapping(ReferenceFrontNotSetException, ErrorCode.REFERENCE_FRONT_NOT_SET)
        self.register_mapping(InvalidProbabilityValueException, ErrorCode.INVALID_PROBABILITY)
        self.register_mapping(NoneParameterException, ErrorCode.NONE_PARAMETER)
        self.register_mapping(EmptyCollectionException, ErrorCode.EMPTY_COLLECTION)
        self.register_mapping(ValueOutOfRangeException, ErrorCode.INVALID_RANGE)
        self.register_mapping(InvalidConditionException, ErrorCode.INVALID_CONDITION)
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default recovery strategies."""
        self.register_strategy(
            ErrorCode.INVALID_PROBABILITY,
            DefaultRecoveryStrategies.recover_invalid_probability
        )
        self.register_strategy(
            ErrorCode.EMPTY_COLLECTION,
            DefaultRecoveryStrategies.recover_empty_collection
        )
        self.register_strategy(
            ErrorCode.NONE_PARAMETER,
            DefaultRecoveryStrategies.recover_none_parameter
        )
    
    def register_mapping(self, exception_type: Type[Exception], error_code: ErrorCode) -> None:
        """Register a mapping from exception type to error code.
        
        Args:
            exception_type (Type[Exception]): The exception type to map.
            error_code (ErrorCode): The error code to assign.
        """
        self.error_mappings[exception_type] = error_code
    
    def register_strategy(
        self,
        error_code: ErrorCode,
        recovery_func: Optional[Callable] = None,
        strategy: Optional[ErrorRecoveryStrategy] = None
    ) -> None:
        """Register a recovery strategy for an error code.
        
        Args:
            error_code (ErrorCode): The error code to register strategy for.
            recovery_func (Optional[Callable]): Recovery function to use.
            strategy (Optional[ErrorRecoveryStrategy]): Pre-configured strategy.
                If provided, recovery_func is ignored.
        """
        if strategy:
            self.recovery_strategies[error_code] = strategy
        else:
            self.recovery_strategies[error_code] = ErrorRecoveryStrategy(
                error_code, recovery_func
            )
    
    def get_error_code(self, exception: Exception) -> ErrorCode:
        """Get the error code for an exception.
        
        Args:
            exception (Exception): The exception to categorize.
        
        Returns:
            ErrorCode: The corresponding error code, or UNKNOWN_ERROR if not mapped.
        """
        exception_type = type(exception)
        
        # Check exact type match
        if exception_type in self.error_mappings:
            return self.error_mappings[exception_type]
        
        # Check parent class matches
        for mapped_type, error_code in self.error_mappings.items():
            if issubclass(exception_type, mapped_type):
                return error_code
        
        # Check for common Python exceptions
        if isinstance(exception, FileNotFoundError):
            return ErrorCode.FILE_NOT_FOUND
        if isinstance(exception, IOError):
            return ErrorCode.IO_ERROR
        if isinstance(exception, MemoryError):
            return ErrorCode.MEMORY_ERROR
        
        return ErrorCode.UNKNOWN_ERROR
    
    def get_severity(self, error_code: ErrorCode) -> ErrorSeverity:
        """Get the severity level for an error code.
        
        Args:
            error_code (ErrorCode): The error code to check.
        
        Returns:
            ErrorSeverity: The severity level.
        """
        # Validation errors are typically medium severity
        if 1000 <= error_code.value < 2000:
            return ErrorSeverity.MEDIUM
        # Execution errors are typically high severity
        elif 2000 <= error_code.value < 4000:
            return ErrorSeverity.HIGH
        # Resource errors are critical
        elif 3000 <= error_code.value < 5000:
            return ErrorSeverity.CRITICAL
        # Algorithm and problem errors are high
        elif 4000 <= error_code.value < 6000:
            return ErrorSeverity.HIGH
        # Unknown errors are critical
        else:
            return ErrorSeverity.CRITICAL
    
    def handle(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        raise_after_recovery: bool = False
    ) -> Optional[Any]:
        """Handle an exception with recovery and logging.
        
        Args:
            exception (Exception): The exception to handle.
            context (Optional[Dict[str, Any]]): Additional context for recovery.
            raise_after_recovery (bool): Whether to re-raise exception after
                attempting recovery. Default is False.
        
        Returns:
            Optional[Any]: Recovery result if recovery succeeded, None otherwise.
        
        Raises:
            Exception: If recovery fails or raise_after_recovery is True.
        """
        error_code = self.get_error_code(exception)
        severity = self.get_severity(error_code)
        context = context or {}
        
        # Log the error with friendly messages
        if self.enable_logging:
            if self.enable_friendly_messages and _FRIENDLY_ERRORS_AVAILABLE:
                try:
                    friendly_msg = format_error(
                        exception,
                        error_code,
                        include_debug=self.debug_mode,
                        debug_info=context
                    )
                    error_msg = f"\n{friendly_msg}"
                except Exception:
                    # Fallback to standard message if friendly message fails
                    error_msg = f"Error [{error_code.name}]: {str(exception)}"
            else:
                error_msg = f"Error [{error_code.name}]: {str(exception)}"
            
            if severity == ErrorSeverity.CRITICAL:
                logger.critical(error_msg, exc_info=self.debug_mode)
            elif severity == ErrorSeverity.HIGH:
                logger.error(error_msg, exc_info=self.debug_mode)
            elif severity == ErrorSeverity.MEDIUM:
                logger.warning(error_msg)
            else:
                logger.info(error_msg)
        
        # Attempt recovery
        if self.enable_recovery and error_code in self.recovery_strategies:
            strategy = self.recovery_strategies[error_code]
            if strategy.can_recover(exception):
                try:
                    result = strategy.recover(exception, context)
                    if result is not None:
                        logger.info(f"Successfully recovered from {error_code.name}")
                        if raise_after_recovery:
                            raise exception
                        return result
                except Exception as recovery_error:
                    logger.error(f"Recovery failed for {error_code.name}: {recovery_error}")
        
        # If recovery failed or not attempted, re-raise if critical
        if severity == ErrorSeverity.CRITICAL:
            raise exception
        
        # For non-critical errors, return None
        return None
    
    def handle_with_fallback(
        self,
        operation: Callable,
        fallback: Callable,
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute an operation with a fallback if it fails.
        
        Args:
            operation (Callable): The operation to execute.
            fallback (Callable): Fallback function to call if operation fails.
            context (Optional[Dict[str, Any]]): Context to pass to error handler.
        
        Returns:
            Any: Result from operation or fallback.
        """
        try:
            return operation()
        except Exception as e:
            self.handle(e, context, raise_after_recovery=False)
            return fallback()


# Global error handler instance
_default_handler: Optional[ErrorHandler] = None


def get_error_handler() -> ErrorHandler:
    """Get the default global error handler instance.
    
    Returns:
        ErrorHandler: The default error handler instance.
    
    Note:
        Creates a new instance on first call. Subsequent calls return the same instance.
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = ErrorHandler()
    return _default_handler


def set_error_handler(handler: ErrorHandler) -> None:
    """Set the default global error handler instance.
    
    Args:
        handler (ErrorHandler): The error handler instance to use as default.
    """
    global _default_handler
    _default_handler = handler

