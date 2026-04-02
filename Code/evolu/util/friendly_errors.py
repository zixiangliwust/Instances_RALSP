"""Friendly error messages with usage suggestions for the evolu framework.

This module provides enhanced error messages that include:
- Clear error descriptions
- Usage suggestions
- Common solutions
- Debug information
"""
from typing import Dict, List, Optional, Tuple
from enum import Enum

from evolu.core.exceptions import EvoluException
from evolu.core.error_handler import ErrorCode, ErrorSeverity


class ErrorCategory(Enum):
    """Error categories for organizing error messages."""
    VALIDATION = "Parameter Validation Error"
    EXECUTION = "Execution Error"
    RESOURCE = "Resource Error"
    ALGORITHM = "Algorithm Error"
    PROBLEM = "Problem Definition Error"
    OPERATOR = "Operator Error"


class FriendlyErrorMessage:
    """Container for friendly error messages with suggestions."""
    
    def __init__(
        self,
        title: str,
        description: str,
        category: ErrorCategory,
        suggestions: List[str],
        examples: Optional[List[str]] = None,
        debug_info: Optional[Dict] = None
    ):
        """Initialize friendly error message.
        
        Args:
            title: Short error title.
            description: Detailed error description.
            category: Error category.
            suggestions: List of suggested solutions.
            examples: Optional code examples.
            debug_info: Optional debug information dictionary.
        """
        self.title = title
        self.description = description
        self.category = category
        self.suggestions = suggestions
        self.examples = examples or []
        self.debug_info = debug_info or {}
    
    def format_message(self, include_debug: bool = False) -> str:
        """Format the error message for display.
        
        Args:
            include_debug: Whether to include debug information.
        
        Returns:
            Formatted error message string.
        """
        lines = [
            f"❌ {self.title}",
            "",
            f"📋 Category: {self.category.value}",
            "",
            f"📝 Description:",
            f"  {self.description}",
            "",
            f"💡 Suggested Solutions:",
        ]
        
        for i, suggestion in enumerate(self.suggestions, 1):
            lines.append(f"  {i}. {suggestion}")
        
        if self.examples:
            lines.extend([
                "",
                "📚 Example Code:",
            ])
            for example in self.examples:
                lines.append(f"  {example}")
        
        if include_debug and self.debug_info:
            lines.extend([
                "",
                "🔍 Debug Info:",
            ])
            for key, value in self.debug_info.items():
                lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


class FriendlyErrorProvider:
    """Provides friendly error messages with suggestions."""
    
    def __init__(self):
        """Initialize the error message provider."""
        self._messages: Dict[ErrorCode, FriendlyErrorMessage] = {}
        self._exception_messages: Dict[type, FriendlyErrorMessage] = {}
        self._initialize_messages()
    
    def _initialize_messages(self) -> None:
        """Initialize default error messages."""
        from evolu.core.exceptions import (
            InvalidParameterException, InvalidParentsException,
            InvalidSolutionException, InvalidRankException,
            InvalidVariantException, EmptyFrontException,
            InvalidDimensionException, ReferenceFrontNotSetException
        )
        from evolu.util.checking import (
            InvalidProbabilityValueException, NoneParameterException,
            EmptyCollectionException, ValueOutOfRangeException
        )
        
        # Invalid probability
        self.register_message(
            ErrorCode.INVALID_PROBABILITY,
            FriendlyErrorMessage(
                title="Invalid Probability Value",
                description="Probability value must be in the range [0.0, 1.0].",
                category=ErrorCategory.VALIDATION,
                suggestions=[
                    "Check if the probability parameter is in the valid range [0.0, 1.0]",
                    "If probability > 1.0, limit it to 1.0",
                    "If probability < 0.0, limit it to 0.0",
                    "Use Check.probability_is_valid() to validate probability values"
                ],
                examples=[
                    "# Error example",
                    "mutation = FloatPolynomialMutation(probability=1.5)  # ❌ Error",
                    "",
                    "# Correct example",
                    "mutation = FloatPolynomialMutation(probability=0.8)  # ✅ Correct"
                ]
            )
        )
        
        # Invalid parents
        self.register_message(
            ErrorCode.INVALID_PARENTS,
            FriendlyErrorMessage(
                title="Invalid Number of Parents",
                description="Crossover operator requires a specific number of parents (usually 2).",
                category=ErrorCategory.OPERATOR,
                suggestions=[
                    "Ensure the correct number of parents are provided to the crossover operator",
                    "Binary crossover operator requires exactly 2 parents",
                    "Check the parent list passed to the execute() method",
                    "Verify the parent list is not empty"
                ],
                examples=[
                    "# Error example",
                    "children = crossover.execute([parent1])  # ❌ Only 1 parent",
                    "",
                    "# Correct example",
                    "children = crossover.execute([parent1, parent2])  # ✅ 2 parents"
                ]
            )
        )
        
        # Empty front
        self.register_message(
            ErrorCode.EMPTY_FRONT,
            FriendlyErrorMessage(
                title="Solution Set is Empty",
                description="Operation requires a non-empty solution set, but received an empty list or None.",
                category=ErrorCategory.EXECUTION,
                suggestions=[
                    "Ensure the population has been initialized before the operation",
                    "Check if solutions have been generated and evaluated",
                    "Verify the population size is greater than 0",
                    "Check if all solutions have been filtered out"
                ],
                examples=[
                    "# Error example",
                    "selected = selector.execute([])  # ❌ Empty list",
                    "",
                    "# Correct example",
                    "solutions = algorithm.create_initial_solutions(100)",
                    "selected = selector.execute(solutions)  # ✅ Non-empty list"
                ]
            )
        )
        
        # None parameter
        self.register_message(
            ErrorCode.NONE_PARAMETER,
            FriendlyErrorMessage(
                title="Parameter is None",
                description="Required parameter cannot be None.",
                category=ErrorCategory.VALIDATION,
                suggestions=[
                    "Check if all required parameters are properly initialized",
                    "Ensure parameters are assigned before calling methods",
                    "Verify parameter passing is correct",
                    "Use Check.is_not_none() to validate parameters"
                ],
                examples=[
                    "# Error example",
                    "problem = None",
                    "algorithm = NSGAII(problem=problem)  # ❌ problem is None",
                    "",
                    "# Correct example",
                    "problem = ZDT1()",
                    "algorithm = NSGAII(problem=problem)  # ✅ problem is initialized"
                ]
            )
        )
        
        # Reference front not set
        self.register_message(
            ErrorCode.REFERENCE_FRONT_NOT_SET,
            FriendlyErrorMessage(
                title="Reference Front Not Set",
                description="Quality indicator requires reference Pareto front for calculation, but not set.",
                category=ErrorCategory.PROBLEM,
                suggestions=[
                    "Set the reference front before calculating quality indicators",
                    "Use set_reference_front() method to set the reference front",
                    "Ensure the reference front format is correct (list of solutions)",
                    "Check if the reference front is empty"
                ],
                examples=[
                    "# Error example",
                    "indicator = GenerationalDistance(problem)",
                    "value = indicator.compute(front)  # ❌ Reference front not set",
                    "",
                    "# Correct example",
                    "indicator = GenerationalDistance(problem)",
                    "indicator.set_reference_front(reference_front)",
                    "value = indicator.compute(front)  # ✅ Reference front is set"
                ]
            )
        )
        
        # Invalid solution
        self.register_message(
            ErrorCode.INVALID_SOLUTION,
            FriendlyErrorMessage(
                title="Invalid Solution Object",
                description="Operation requires a valid solution object, but received None or invalid object.",
                category=ErrorCategory.EXECUTION,
                suggestions=[
                    "Ensure solution object is created correctly",
                    "Verify solution object is not None",
                    "Check if solution object type is correct",
                    "Ensure solution has been evaluated (has objective values)"
                ],
                examples=[
                    "# Error example",
                    "solution = None",
                    "problem.evaluate_solution(solution)  # ❌ solution is None",
                    "",
                    "# Correct example",
                    "solution = problem.create_solution()",
                    "problem.evaluate_solution(solution)  # ✅ solution is valid"
                ]
            )
        )
        
        # File not found
        self.register_message(
            ErrorCode.FILE_NOT_FOUND,
            FriendlyErrorMessage(
                title="File Not Found",
                description="Specified file does not exist or path is incorrect.",
                category=ErrorCategory.RESOURCE,
                suggestions=[
                    "Check if file path is correct (including case sensitivity)",
                    "Verify file exists at the specified location",
                    "Use absolute path or relative path from project root directory",
                    "Check file permissions allow reading"
                ],
                examples=[
                    "# Error example",
                    "problem = PermutationDLBP1('wrong/path/', 'instance.txt')  # ❌",
                    "",
                    "# Correct example",
                    "problem = PermutationDLBP1('resources/DLBP1_instances/', 'instance.txt')  # ✅"
                ]
            )
        )
        
        # Register exception type mappings
        self._exception_messages[InvalidProbabilityValueException] = self._messages[ErrorCode.INVALID_PROBABILITY]
        self._exception_messages[InvalidParentsException] = self._messages[ErrorCode.INVALID_PARENTS]
        self._exception_messages[EmptyFrontException] = self._messages[ErrorCode.EMPTY_FRONT]
        self._exception_messages[NoneParameterException] = self._messages[ErrorCode.NONE_PARAMETER]
        self._exception_messages[ReferenceFrontNotSetException] = self._messages[ErrorCode.REFERENCE_FRONT_NOT_SET]
        self._exception_messages[InvalidSolutionException] = self._messages[ErrorCode.INVALID_SOLUTION]
    
    def register_message(
        self,
        error_code: ErrorCode,
        message: FriendlyErrorMessage
    ) -> None:
        """Register a friendly error message for an error code.
        
        Args:
            error_code: The error code to register.
            message: The friendly error message.
        """
        self._messages[error_code] = message
    
    def get_message(
        self,
        error_code: ErrorCode,
        exception: Optional[Exception] = None
    ) -> Optional[FriendlyErrorMessage]:
        """Get friendly error message for an error code.
        
        Args:
            error_code: The error code.
            exception: Optional exception instance for context.
        
        Returns:
            Friendly error message, or None if not found.
        """
        return self._messages.get(error_code)
    
    def get_message_for_exception(
        self,
        exception: Exception
    ) -> Optional[FriendlyErrorMessage]:
        """Get friendly error message for an exception.
        
        Args:
            exception: The exception instance.
        
        Returns:
            Friendly error message, or None if not found.
        """
        # Check exact type match
        exception_type = type(exception)
        if exception_type in self._exception_messages:
            return self._exception_messages[exception_type]
        
        # Check parent class matches
        for mapped_type, message in self._exception_messages.items():
            if issubclass(exception_type, mapped_type):
                return message
        
        return None
    
    def format_exception(
        self,
        exception: Exception,
        error_code: Optional[ErrorCode] = None,
        include_debug: bool = False,
        debug_info: Optional[Dict] = None
    ) -> str:
        """Format an exception with friendly message.
        
        Args:
            exception: The exception to format.
            error_code: Optional error code.
            include_debug: Whether to include debug information.
            debug_info: Optional additional debug information.
        
        Returns:
            Formatted error message string.
        """
        # Try to get friendly message
        friendly_msg = None
        
        if error_code:
            friendly_msg = self.get_message(error_code, exception)
        
        if not friendly_msg:
            friendly_msg = self.get_message_for_exception(exception)
        
        if friendly_msg:
            # Add debug info if provided
            if debug_info:
                friendly_msg.debug_info.update(debug_info)
            
            # Add exception details to debug info
            if include_debug:
                friendly_msg.debug_info.update({
                    "Exception Type": type(exception).__name__,
                    "Exception Message": str(exception),
                    "Error Code": error_code.name if error_code else "Unknown"
                })
            
            return friendly_msg.format_message(include_debug=include_debug)
        
        # Fallback to standard exception message
        return f"Error: {type(exception).__name__}: {str(exception)}"


# Global instance
_default_provider: Optional[FriendlyErrorProvider] = None


def get_friendly_error_provider() -> FriendlyErrorProvider:
    """Get the default friendly error provider instance.
    
    Returns:
        The default friendly error provider.
    """
    global _default_provider
    if _default_provider is None:
        _default_provider = FriendlyErrorProvider()
    return _default_provider


def format_error(
    exception: Exception,
    error_code: Optional[ErrorCode] = None,
    include_debug: bool = False,
    debug_info: Optional[Dict] = None
) -> str:
    """Format an exception with friendly error message.
    
    Args:
        exception: The exception to format.
        error_code: Optional error code.
        include_debug: Whether to include debug information.
        debug_info: Optional additional debug information.
    
    Returns:
        Formatted friendly error message.
    
    Example:
        >>> try:
        ...     mutation = FloatPolynomialMutation(probability=1.5)
        ... except InvalidProbabilityValueException as e:
        ...     print(format_error(e, include_debug=True))
    """
    provider = get_friendly_error_provider()
    return provider.format_exception(exception, error_code, include_debug, debug_info)

