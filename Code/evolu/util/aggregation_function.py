# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from math import sqrt
from typing import List
from evolu.util.point import IdealPoint

"""
module:: aggregation_function
synopsis: Implementation of aggregative (scalarizing) functions.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class AggregationFunction(ABC):
    """Base class for aggregation (scalarizing) functions.
    
    Aggregation functions convert multi-objective optimization problems into
    single-objective problems by combining multiple objectives into a single
    scalar value. They are commonly used in decomposition-based multi-objective
    algorithms like MOEA/D.
    
    Aggregation functions take an objective vector and a weight vector, producing
    a single scalar value that represents the quality of the solution from the
    perspective defined by the weights.
    
    Note:
        Subclasses must implement compute() and update() methods.
    """
    
    @abstractmethod
    def compute(self, objective_vector: List[float], weight_vector: List[float]) -> float:
        """Compute aggregation function value.
        
        Args:
            objective_vector (List[float]): Objective values of a solution.
                Length should match the number of objectives.
            weight_vector (List[float]): Weight vector defining the direction
                of search. Should sum to 1.0 and have same length as objective_vector.
        
        Returns:
            float: Aggregated scalar value. Lower values typically indicate
                better solutions (for minimization problems).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `compute`.")

    @abstractmethod
    def update(self, objective_vector: List[float]) -> None:
        """Update internal state of the aggregation function.
        
        Some aggregation functions maintain internal state (e.g., ideal point,
        nadir point) that needs to be updated as new solutions are evaluated.
        
        Args:
            objective_vector (List[float]): Objective values to use for updating
                internal state (e.g., updating ideal point).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `update`.")


class WeightedSum(AggregationFunction):
    """Weighted sum (linear) aggregation function.
    
    The weighted sum aggregates objectives as a linear combination:
        f_agg = Σ(w_i * f_i)
    where w_i are the weights and f_i are the objective values.
    
    This is the simplest aggregation function and works well for convex
    Pareto fronts but may struggle with non-convex regions.
    
    Note:
        This function has no internal state, so update() is a no-op.
    """
    
    def compute(self, objective_vector: List[float], weight_vector: List[float]) -> float:
        """Compute weighted sum of objectives.
        
        Args:
            objective_vector (List[float]): Objective values.
            weight_vector (List[float]): Weight vector.
        
        Returns:
            float: Weighted sum = Σ(w_i * f_i).
        """
        return sum(map(lambda x, y: x * y, objective_vector, weight_vector))

    def update(self, objective_vector: List[float]) -> None:
        """Update internal state (no-op for weighted sum).
        
        Weighted sum has no internal state, so this method does nothing.
        
        Args:
            objective_vector (List[float]): Not used.
        """
        pass


class PenaltyBoundaryIntersection(AggregationFunction):
    """Penalty-based Boundary Intersection (PBI) aggregation function.
    
    PBI combines the distance along the weight vector direction (d1) with a
    penalty term (d2) that measures the distance perpendicular to the weight
    vector. The aggregated value is:
        f_agg = d1 + theta * d2
    where theta is a penalty parameter controlling the balance between
    convergence and diversity.
    
    Attributes:
        ideal_point (IdealPoint): Ideal point (best value for each objective)
            used in distance calculations. Updated during optimization.
        theta (float): Penalty parameter controlling the balance between
            convergence (d1) and diversity (d2). Larger values emphasize diversity.
    
    Reference:
        Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary
        algorithm based on decomposition. IEEE Transactions on evolutionary
        computation, 11(6), 712-731.
    """
    
    def __init__(self, dimension: int, theta: float = 5.0) -> None:
        """Initialize penalty boundary intersection aggregation function.
        
        Args:
            dimension (int): Number of objectives.
            theta (float, optional): Penalty parameter. Defaults to 5.0.
                Typical values range from 0.1 to 10.0.
        """
        self.ideal_point = IdealPoint(dimension)
        self.theta = theta

    def compute(self, objective_vector: List[float], weight_vector: List[float]) -> float:
        d1 = d2 = nl = 0.0
        for i in range(len(objective_vector)):
            d1 += (objective_vector[i] - self.ideal_point.point[i]) * weight_vector[i]
            nl += pow(weight_vector[i], 2.0)
        nl = sqrt(nl)
        d1 = abs(d1) / nl
        for i in range(len(objective_vector)):
            d2 += pow((objective_vector[i] - self.ideal_point.point[i]) - d1 * (weight_vector[i] / nl), 2.0)
        d2 = sqrt(d2)
        return d1 + self.theta * d2

    def update(self, objective_vector: List[float]) -> None:
        """Update ideal point."""
        self.ideal_point.update(objective_vector)


class Tschebycheff(AggregationFunction):
    """Tchebycheff (Chebyshev) aggregation function.
    
    The Tchebycheff function aggregates objectives using the L∞ norm:
        f_agg = max_i(w_i * |f_i - z_i*|)
    where z* is the ideal point. This function can handle both convex and
    non-convex Pareto fronts effectively.
    
    Attributes:
        ideal_point (IdealPoint): Ideal point (best value for each objective)
            used as reference. Updated during optimization.
    
    Reference:
        Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary
        algorithm based on decomposition. IEEE Transactions on evolutionary
        computation, 11(6), 712-731.
    """
    
    def __init__(self, dimension: int) -> None:
        """Initialize Tchebycheff aggregation function.
        
        Args:
            dimension (int): Number of objectives.
        """
        self.ideal_point = IdealPoint(dimension)

    def compute(self, objective_vector: List[float], weight_vector: List[float]) -> float:
        max_fun = -1.0e30
        for i in range(len(objective_vector)):
            diff = abs(objective_vector[i] - self.ideal_point.point[i])
            if weight_vector[i] == 0:
                feval = 0.0001 * diff
            else:
                feval = diff * weight_vector[i]
            if feval > max_fun:
                max_fun = feval
        return max_fun

    def update(self, objective_vector: List[float]) -> None:
        """Update ideal point."""
        self.ideal_point.update(objective_vector)
