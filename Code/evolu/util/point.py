# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List

"""
module:: point
synopsis: implementation of points of n-dimensions (e.g, ideal point, nadir point, etc.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Point(ABC):
    """Base class for reference points in multi-objective optimization.
    
    Reference points are important landmarks in the objective space, such as
    ideal points (best possible values) and nadir points (worst possible values
    on the Pareto front). They are used in various algorithms for normalization,
    aggregation functions, and quality assessment.
    
    Note:
        Subclasses must implement the update() method to define how the point
        is updated with new objective vectors.
    """
    
    @abstractmethod
    def update(self, objective_vector: List[float]) -> None:
        """Update point with new objective vector.
        
        Args:
            objective_vector (List[float]): New objective values to use for updating
                the reference point. Length should match the point's dimension.
        
        Note:
            The update strategy depends on the point type:
            - Ideal point: Takes minimum value for each objective
            - Nadir point: Takes maximum value for each objective
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `update`.")


class IdealPoint(Point):
    """Ideal point (utopia point) in multi-objective optimization.
    
    The ideal point represents the best possible value for each objective
    independently. For minimization problems, it contains the minimum value
    achieved (or theoretically possible) for each objective. The ideal point
    is used in decomposition-based algorithms (e.g., MOEA/D) and quality
    indicators (e.g., IGD).
    
    The ideal point is initialized with infinite values and updated by taking
    the minimum value seen so far for each objective.
    
    Attributes:
        point (List[float]): Current ideal point values, one per objective.
            Initially set to infinity for each dimension.
    
    Example:
        >>> ideal = IdealPoint(dimension=3)
        >>> ideal.update([1.5, 2.0, 1.0])
        >>> ideal.update([1.0, 2.5, 0.8])
        >>> ideal.point
        [1.0, 2.0, 0.8]  # Minimum value for each objective
    """
    
    def __init__(self, dimension: int) -> None:
        """Initialize ideal point with infinite values.
        
        Args:
            dimension (int): Number of objectives (dimensionality of the point).
        """
        self.point: List[float] = dimension * [float("inf")]

    def update(self, objective_vector: List[float]) -> None:
        """Update ideal point with new objective vector.
        
        For each objective, takes the minimum value between the current ideal
        point value and the new objective value. This ensures the ideal point
        always contains the best (minimum) value seen so far for each objective.
        
        Args:
            objective_vector (List[float]): New objective values to consider.
                Length should match the point's dimension.
        
        Note:
            This method is typically called during optimization to track the
            best values achieved for each objective independently.
        """
        self.point = [y if x > y else x for x, y in zip(self.point, objective_vector)]
