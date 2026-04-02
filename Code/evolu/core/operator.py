# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from evolu.core.solution import Solution
from evolu.core.exceptions import InvalidParameterException

S = TypeVar("S", bound=Solution)
R = TypeVar("R", bound=Solution)
"""
module:: Operator
synopsis: Templates for operators.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Operator(Generic[S, R], ABC):
    """Base class for all genetic operators.
    
    Operators are fundamental components of evolutionary algorithms that manipulate
    solutions to create new solutions. This class defines the common interface
    that all operators must implement.
    
    Attributes:
        S: Type of source solution (input).
        R: Type of result solution (output).
    
    Note:
        Subclasses must implement execute() and get_name() methods.
    """

    @abstractmethod
    def execute(self, source: S) -> R:
        """Execute the operator on the given source.
        
        Args:
            source (S): The source solution(s) to operate on.
        
        Returns:
            R: The result solution(s) after applying the operator.
        
        Note:
            Subclasses must implement this method to define the specific
            operator behavior (e.g., mutation, crossover, selection).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `execute`.")

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: The name of the operator (e.g., "FloatPolynomialMutation").
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_name`.")


def check_valid_probability_value(func):
    """Decorator to check if probability value is valid (0.0 to 1.0).
    
    This decorator validates that probability values are in the valid range
    [0.0, 1.0] before the function is executed.
    
    Args:
        func: The function to be decorated.
    
    Returns:
        The decorated function that validates probability values.
    
    Raises:
        InvalidParameterException: If probability is outside [0.0, 1.0].
    
    Example:
        >>> @check_valid_probability_value
        >>> def __init__(self, probability: float):
        >>>     self.probability = probability
    """
    def func_wrapper(self, probability: float):
        if probability > 1.0:
            raise InvalidParameterException(
                f"The probability is greater than one: {probability}"
            )
        elif probability < 0.0:
            raise InvalidParameterException(
                f"The probability is lower than zero: {probability}"
            )
        res = func(self, probability)
        return res

    return func_wrapper


class Mutation(Operator[S, S], ABC):
    """Base class for mutation operators.
    
    Mutation operators modify a single solution to create a new solution.
    They are used to introduce diversity into the population.
    
    Attributes:
        probability (float): Probability of applying mutation (in [0.0, 1.0]).
    
    Note:
        Subclasses must implement execute() and get_name() methods.
    """

    @check_valid_probability_value
    def __init__(self, probability: float) -> None:
        """Initialize mutation operator with probability.
        
        Args:
            probability (float): Probability of applying mutation (must be in [0.0, 1.0]).
        
        Raises:
            InvalidParameterException: If probability is outside [0.0, 1.0].
        """
        self.probability = probability


class Crossover(Operator[List[S], List[R]], ABC):
    """Base class for crossover operators.
    
    Crossover operators combine multiple parent solutions to create offspring solutions.
    They are used to exchange genetic material between solutions.
    
    Attributes:
        probability (float): Probability of applying crossover (in [0.0, 1.0]).
    
    Note:
        Subclasses must implement execute(), get_name(), get_number_of_parents(),
        and get_number_of_children() methods.
    """

    @check_valid_probability_value
    def __init__(self, probability: float) -> None:
        """Initialize crossover operator with probability.
        
        Args:
            probability (float): Probability of applying crossover (must be in [0.0, 1.0]).
        
        Raises:
            InvalidParameterException: If probability is outside [0.0, 1.0].
        """
        self.probability = probability

    @abstractmethod
    def get_number_of_parents(self) -> int:
        """Get the number of parent solutions required.
        
        Returns:
            int: Number of parent solutions needed for this crossover operator
                (typically 2 for binary crossover).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_number_of_parents`.")

    @abstractmethod
    def get_number_of_children(self) -> int:
        """Get the number of offspring solutions produced.
        
        Returns:
            int: Number of child solutions produced by this crossover operator
                (typically 2 for binary crossover).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_number_of_children`.")


class Selection(Operator[S, R], ABC):
    """Base class for selection operators.
    
    Selection operators choose solutions from a population or front based on
    some criteria (e.g., fitness, dominance, crowding distance).
    
    Note:
        Subclasses must implement execute() and get_name() methods.
    """

    def __init__(self) -> None:
        """Initialize selection operator."""
        pass
