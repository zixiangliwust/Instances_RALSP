# -*- coding: utf-8 -*-
"""Utility classes and functions for parameter validation and checking.
 
This module provides validation utilities and exceptions for checking various
conditions (None values, probability ranges, collection emptiness, etc.) used
throughout the evolu framework.
"""

from typing import Any


class NoneParameterException(Exception):
    """Exception raised when a required parameter is None.
    
    This exception is raised by validation functions when a parameter that
    cannot be None is found to be None.
    
    Attributes:
        error_message (str): Optional error message describing the issue.
    """
    
    def __init__(self, message: str = "") -> None:
        """Initialize exception with an optional message.
        
        Args:
            message (str, optional): Error message describing the issue.
                Defaults to empty string.
        """
        self.error_message = message


class InvalidConditionException(Exception):
    """Exception raised when a validation condition is not met.
    
    This exception is raised when a boolean condition check fails, indicating
    that some expected condition is not satisfied.
    
    Attributes:
        error_message (str): Error message describing the failed condition.
    """
    
    def __init__(self, message: str) -> None:
        """Initialize exception with a message.
        
        Args:
            message (str): Error message describing the failed condition.
        """
        self.error_message = message


class EmptyCollectionException(RuntimeError):
    """Exception raised when a required collection is empty.
    
    This exception is raised when an operation expects a non-empty collection
    (list, set, dict, etc.) but receives an empty one.
    """
    
    def __init__(self) -> None:
        """Initialize exception for empty collection."""
        super(EmptyCollectionException, self).__init__("The collection is empty")


# class InvalidConditionException(RuntimeError):
#    def __init__(self, message):
#        super(InvalidConditionException, self).__init__(message)

class InvalidProbabilityValueException(RuntimeError):
    """Exception raised when a probability value is outside [0.0, 1.0].
    
    This exception is raised when a probability parameter is expected to be
    in the range [0.0, 1.0] but is outside this range.
    """
    
    def __init__(self, value: float) -> None:
        """Initialize exception for invalid probability value.
        
        Args:
            value (float): The invalid probability value that was provided.
        """
        super(InvalidProbabilityValueException, self).__init__(
            "The parameter " + str(value) + " is not a valid probability value"
        )


class ValueOutOfRangeException(RuntimeError):
    """Exception raised when a value is outside the expected range.
    
    This exception is raised when a numeric value is expected to be within
    a specific range [lowest_value, highest_value] but is outside this range.
    """
    
    def __init__(self, value: float, lowest_value: float, highest_value: float) -> None:
        """Initialize exception for value out of range.
        
        Args:
            value (float): The value that is out of range.
            lowest_value (float): The minimum allowed value.
            highest_value (float): The maximum allowed value.
        """
        super(ValueOutOfRangeException, self).__init__(
            "The parameter "
            + str(value)
            + " is not in the range ("
            + str(lowest_value)
            + ", "
            + str(highest_value)
            + ")"
        )


class Check:
    """Utility class for parameter validation and condition checking.
    
    This class provides static methods for validating parameters and checking
    conditions throughout the evolu framework. All methods raise exceptions
    if the validation fails, allowing for early error detection.
    
    Example:
        >>> Check.is_not_none(my_object)
        >>> Check.probability_is_valid(0.5)
        >>> Check.value_is_in_range(x, 0.0, 100.0)
        >>> Check.that(len(solutions) > 0, "Solution list is empty")
    """
    
    @staticmethod
    def is_not_none(obj: Any) -> None:
        """Check if object is not None.
        
        Args:
            obj (Any): Object to check.
        
        Raises:
            NoneParameterException: If obj is None.
        
        Example:
            >>> Check.is_not_none(my_solution)  # Raises if my_solution is None
        """
        if obj is None:
            raise NoneParameterException()

    @staticmethod
    def probability_is_valid(value: float) -> None:
        """Check if probability value is valid (0.0 to 1.0).
        
        Args:
            value (float): Probability value to validate.
        
        Raises:
            InvalidProbabilityValueException: If value is outside [0.0, 1.0].
        
        Example:
            >>> Check.probability_is_valid(0.5)  # Valid
            >>> Check.probability_is_valid(1.5)  # Raises exception
        """
        if value < 0.0 or value > 1.0:
            raise InvalidProbabilityValueException(value)

    @staticmethod
    def value_is_in_range(value: float, lowest_value: float, highest_value: float) -> None:
        """Check if value is within the specified range.
        
        Args:
            value (float): Value to check.
            lowest_value (float): Minimum allowed value (inclusive).
            highest_value (float): Maximum allowed value (inclusive).
        
        Raises:
            ValueOutOfRangeException: If value is outside [lowest_value, highest_value].
        
        Example:
            >>> Check.value_is_in_range(x, 0.0, 100.0)  # Valid if x in [0, 100]
        """
        if value < lowest_value or value > highest_value:
            raise ValueOutOfRangeException(value, lowest_value, highest_value)

    @staticmethod
    def collection_is_not_empty(collection: Any) -> None:
        """Check if collection is not empty.
        
        Args:
            collection (Any): Collection to check (must support len()).
        
        Raises:
            EmptyCollectionException: If collection is empty.
        
        Example:
            >>> Check.collection_is_not_empty(solutions)  # Raises if empty list
        """
        if len(collection) == 0:
            raise EmptyCollectionException

    @staticmethod
    def that(expression: bool, message: str) -> None:
        """Check if expression is true, otherwise raise exception.
        
        This is a general-purpose assertion method for validating any boolean
        condition with a custom error message.
        
        Args:
            expression (bool): Condition to check. Should be True for validation to pass.
            message (str): Error message to raise if expression is False.
        
        Raises:
            InvalidConditionException: If expression is False.
        
        Example:
            >>> Check.that(len(solutions) > 0, "Solution list cannot be empty")
            >>> Check.that(x > y, "x must be greater than y")
        """
        if not expression:
            raise InvalidConditionException(message)


"""
class Check:
    @staticmethod
    def is_not_null(o: object, message: str = ""):
        if o is None:
            raise NoneParameterException(message)
    @staticmethod
    def that(expression: bool, message: str = ""):
        if not expression:
            raise InvalidConditionException(message)
"""
