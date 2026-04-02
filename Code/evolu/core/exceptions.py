# -*- coding: utf-8 -*-
"""Custom exceptions for the evolu framework.
 
This module defines the exception hierarchy used throughout the evolu framework.
All exceptions inherit from EvoluException, allowing for easy exception handling
and providing a consistent error reporting mechanism.
"""
from typing import Optional


class EvoluException(Exception):
    """Base exception class for all evolu framework exceptions.
    
    This is the root exception class from which all other evolu-specific
    exceptions inherit. It allows catching all evolu exceptions with a single
    except clause while maintaining specific exception types for detailed handling.
    
    Attributes:
        error_code (Optional[str]): Optional error code for categorization.
        context (Optional[dict]): Optional context dictionary with additional error information.
    
    Example:
        >>> try:
        ...     # evolu framework code
        ...     pass
        ... except EvoluException as e:
        ...     # Handle any evolu exception
        ...     print(f"evolu error: {e}")
    """
    
    def __init__(self, message: str = "", error_code: Optional[str] = None, context: Optional[dict] = None) -> None:
        """Initialize exception with optional error code and context.
        
        Args:
            message (str): Error message describing the issue.
            error_code (Optional[str]): Error code for categorization.
            context (Optional[dict]): Additional context information.
        """
        super().__init__(message)
        self.error_code: Optional[str] = error_code
        self.context: Optional[dict] = context or {}


class InvalidParentsException(EvoluException):
    """Exception raised when an invalid number of parents is provided to an operator.
    
    This exception is typically raised by crossover operators when they receive
    an unexpected number of parent solutions. For example, a binary crossover
    operator expects exactly 2 parents.
    
    Example:
        >>> crossover = FloatOnePointCrossover()
        >>> try:
        ...     children = crossover.execute([parent1])  # Only 1 parent
        ... except InvalidParentsException as e:
        ...     print(f"Error: {e}")  # "The number of parents is not two: 1"
    """
    pass


class InvalidParameterException(EvoluException):
    """Exception raised when an invalid parameter value is provided.
    
    This exception is raised when a parameter value is outside the expected
    range or type. Common cases include:
    - Probability values outside [0.0, 1.0]
    - Negative values where positive are required
    - Invalid parameter types
    
    Example:
        >>> try:
        ...     mutation = FloatPolynomialMutation(probability=1.5)  # > 1.0
        ... except InvalidParameterException as e:
        ...     print(f"Error: {e}")  # "The probability is greater than one: 1.5"
    """
    pass


class ReferenceFrontNotSetException(EvoluException):
    """Exception raised when a reference front is not set but required.
    
    This exception is raised by quality indicators (e.g., GD, IGD) that require
    a reference Pareto front for computation. The reference front must be set
    before calling compute().
    
    Example:
        >>> indicator = GenerationalDistance()
        >>> try:
        ...     value = indicator.compute(front)  # Reference front not set
        ... except ReferenceFrontNotSetException as e:
        ...     print(f"Error: {e}")  # "Reference front is not set"
    """
    pass


class InvalidRankException(EvoluException):
    """Exception raised when an invalid rank is accessed.
    
    This exception is raised when trying to access a rank (front index) that
    is out of bounds in a Ranking object. Ranks are zero-indexed, with rank 0
    being the best (non-dominated) front.
    
    Example:
        >>> ranking = FastNonDominatedRanking()
        >>> ranking.compute_ranking(solutions)
        >>> try:
        ...     front = ranking.get_sub_front(10)  # Invalid rank
        ... except InvalidRankException as e:
        ...     print(f"Error: {e}")  # "Invalid rank: 10. Max rank: 2"
    """
    pass


class InvalidVariantException(EvoluException):
    """Exception raised when an invalid algorithm variant is specified.
    
    This exception is raised when an algorithm or operator is configured with
    a variant string that is not recognized or supported.
    
    Example:
        >>> crossover = FloatDifferentialEvolutionCrossover(DE_Variant="invalid/variant")
        >>> try:
        ...     child = crossover.execute(parents)
        ... except InvalidVariantException as e:
        ...     print(f"Error: {e}")  # "Invalid DE_Variant 'invalid/variant'"
    """
    pass


class InvalidSolutionException(EvoluException):
    """Exception raised when a solution is None or invalid.
    
    This exception is raised when an operation expects a valid Solution object
    but receives None or an object that doesn't meet the required criteria.
    
    Example:
        >>> try:
        ...     problem.evaluate_solution(None)  # None solution
        ... except InvalidSolutionException as e:
        ...     print(f"Error: {e}")
    """
    pass


class EmptyFrontException(EvoluException):
    """Exception raised when a front is empty or null.
    
    This exception is raised by selection operators and other components that
    expect a non-empty list of solutions but receive an empty list or None.
    
    Example:
        >>> selector = BinaryTournamentSelection()
        >>> try:
        ...     selected = selector.execute([])  # Empty front
        ... except EmptyFrontException as e:
        ...     print(f"Error: {e}")  # "The front is empty"
    """
    pass


class InvalidFigureException(EvoluException):
    """Exception raised when a figure object is None or invalid.
    
    This exception is raised by visualization modules when a matplotlib figure
    object is expected but is None or invalid.
    
    Example:
        >>> try:
        ...     draw_line_chart(None, data)  # None figure
        ... except InvalidFigureException as e:
        ...     print(f"Error: {e}")
    """
    pass


class InvalidDimensionException(EvoluException):
    """Exception raised when an invalid dimension is specified.
    
    This exception is raised when an operation expects a valid dimension (e.g.,
    objective index, variable index) but receives an out-of-bounds or invalid value.
    
    Example:
        >>> try:
        ...     value = solution.objectives[10]  # Invalid objective index
        ... except InvalidDimensionException as e:
        ...     print(f"Error: {e}")
    """
    pass

