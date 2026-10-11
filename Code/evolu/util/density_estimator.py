# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from functools import cmp_to_key
from typing import List, TypeVar
import numpy
from scipy.spatial.distance import euclidean
from evolu.core.solution import Solution
from evolu.logger import get_logger
from evolu.util.comparator import Comparator, SolutionAttributeComparator

logger = get_logger(__name__)
S = TypeVar("S", bound=Solution)
"""
module:: density_estimator
synopsis: Module including the implementation of density estimators.
moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>
"""


class DensityEstimator(ABC):
    """Base class for density estimation algorithms.
    
    Density estimators measure the crowding or density of solutions in the
    objective space. They are commonly used in multi-objective optimization
    to maintain diversity and select solutions for survival or reproduction.
    Examples include crowding distance (NSGA-II) and k-nearest neighbor density.
    
    Note:
        Subclasses must implement compute_density_estimator(), sort(), and
        get_comparator() methods.
    """

    @abstractmethod
    def compute_density_estimator(self, solutions: List[S]) -> None:
        """Compute density estimator values for solutions.
        
        This method computes the density estimate for each solution and typically
        stores it in the solution's attributes dictionary for later use in
        selection or sorting operations.
        
        Args:
            solutions (List[S]): List of solutions for which to compute density
                estimates. Modified in-place with density values stored in
                solution.attributes.
        
        Note:
            The density value is typically stored in solution.attributes with
            a specific key (e.g., "crowding_distance"). Check the implementation
            for the exact attribute name used.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `compute_density_estimator`.")

    @abstractmethod
    def sort(self, solutions: List[S]) -> List[S]:
        """Sort solutions by density estimator value.
        
        Args:
            solutions (List[S]): List of solutions to sort.
        
        Returns:
            List[S]: Sorted list of solutions (typically sorted in descending
                order of density, so solutions with higher density come first).
        
        Note:
            This method assumes compute_density_estimator() has already been
            called on the solutions.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `sort`.")

    @classmethod
    @abstractmethod
    def get_comparator(cls) -> Comparator:
        """Get the comparator used for sorting solutions by density.
        
        Returns:
            Comparator: Comparator that compares solutions based on their
                density estimator values.
        """
        raise NotImplementedError(f"{cls.__name__} must implement `get_comparator`.")


class CrowdingDistance(DensityEstimator):
    """Crowding distance density estimator from NSGA-II.
    
    Crowding distance measures the density of solutions around a particular
    solution in the objective space. Solutions with larger crowding distances
    are in less crowded regions and are preferred for maintaining diversity.
    
    The crowding distance for a solution is computed as the sum of normalized
    distances to neighboring solutions along each objective dimension.
    Boundary solutions (those with minimum or maximum values in any objective)
    are assigned infinite crowding distance to ensure they are always selected.
    
    The computed distance is stored in solution.attributes["crowding_distance"].
    
    Reference:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
        elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on
        evolutionary computation, 6(2), 182-197.
    """

    def compute_density_estimator(self, front: List[S]) -> None:
        """Compute crowding distance for all solutions in the front.
        
        This method computes the crowding distance for each solution and stores
        it in solution.attributes["crowding_distance"]. Boundary solutions
        (extreme points) are assigned infinite distance to ensure preservation.
        
        Args:
            front (List[S]): List of solutions (typically a non-dominated front)
                for which to compute crowding distances. Modified in-place.
        
        Note:
            - Solutions are assumed to have the same number of objectives.
            - For empty front: no operation.
            - For 1-2 solutions: all assigned infinite distance.
            - For 3+ solutions: interior solutions get computed distance,
              boundary solutions get infinite distance.
        """
        size = len(front)
        if size == 0:
            return
        elif size == 1:
            front[0].attributes["crowding_distance"] = float("inf")
            return
        elif size == 2:
            front[0].attributes["crowding_distance"] = float("inf")
            front[1].attributes["crowding_distance"] = float("inf")
            return
        for i in range(len(front)):
            front[i].attributes["crowding_distance"] = 0.0
        number_of_objectives = front[0].number_of_objectives
        for i in range(number_of_objectives):
            # Sort the population by Obj n
            front = sorted(front, key=lambda x: x.objectives[i])
            objective_min = front[0].objectives[i]
            objective_max = front[len(front) - 1].objectives[i]
            # Set de crowding distance
            front[0].attributes["crowding_distance"] = float("inf")
            front[size - 1].attributes["crowding_distance"] = float("inf")
            for j in range(1, size - 1):
                distance = front[j + 1].objectives[i] - front[j - 1].objectives[i]
                if objective_max - objective_min == 0:
                    pass
                else:
                    distance = distance / (objective_max - objective_min)
                distance += front[j].attributes["crowding_distance"]
                front[j].attributes["crowding_distance"] = distance

    def sort(self, solutions: List[S]) -> List[S]:
        solutions.sort(key=cmp_to_key(self.get_comparator().compare))

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("crowding_distance", lowest_is_best=False)


class ModifiedCrowdingDistance(DensityEstimator):
    """This class implements a DensityEstimator based on the crowding distance of algorithm NSGA-II."""

    def compute_density_estimator(self, front: List[S]) -> None:
        """This function performs the computation of the crowding density estimation over the solution list.
        Note:
        This method assign the distance in the inner elements of the solution list.
        :param front: The list of solutions.
        """
        size = len(front)
        if size == 0:
            return
        elif size == 1:
            front[0].attributes["modified_crowding_distance"] = 1.0
            return
        elif size == 2:
            front[0].attributes["modified_crowding_distance"] = 1.0
            front[1].attributes["modified_crowding_distance"] = 1.0
            return
        for i in range(len(front)):
            front[i].attributes["modified_crowding_distance"] = 0.0
        number_of_objectives = front[0].number_of_objectives
        for i in range(number_of_objectives):
            # Sort the population by Obj n
            front = sorted(front, key=lambda x: x.objectives[i])
            objective_min = front[0].objectives[i]
            objective_max = front[len(front) - 1].objectives[i]
            # Set de crowding distance
            front[0].attributes["modified_crowding_distance"] = 1.0
            front[size - 1].attributes["modified_crowding_distance"] = 1.0
            for j in range(1, size - 1):
                distance = front[j + 1].objectives[i] - front[j - 1].objectives[i]
                if objective_max - objective_min == 0:
                    pass
                else:
                    distance = distance / (objective_max - objective_min)
                distance += front[j].attributes["modified_crowding_distance"]
                front[j].attributes["modified_crowding_distance"] = distance
        for i in range(len(front)):
            # front[i].attributes["modified_crowding_distance"] = \
            #     front[i].attributes["modified_crowding_distance"] / (front[i].survive_time + 1.0)
            front[i].attributes["modified_crowding_distance"] = \
                front[i].attributes["modified_crowding_distance"] * pow(0.5, front[i].survive_time)

    def sort(self, solutions: List[S]) -> List[S]:
        solutions.sort(key=cmp_to_key(self.get_comparator().compare))

    @classmethod
    def get_comparator(cls) -> Comparator:
        return SolutionAttributeComparator("modified_crowding_distance", lowest_is_best=False)


class KNearestNeighborDensityEstimator(DensityEstimator):
    """Density estimator based on distance to k-th nearest neighbor.
    
    This density estimator computes the distance from each solution to its
    k-th nearest neighbor in objective space. Solutions in dense regions
    will have smaller k-th nearest neighbor distances, while solutions in
    sparse regions will have larger distances.
    
    The density value for a solution is the distance to its k-th nearest
    neighbor. This is useful for maintaining diversity in multi-objective
    optimization, as solutions in less crowded regions are preferred.
    
    Attributes:
        k (int): Number of nearest neighbors to consider (default is 1, meaning
            distance to nearest neighbor).
        distance_matrix (List[List[float]]): Precomputed pairwise distance matrix
            between all solutions in objective space.
    
    Reference:
        Based on k-nearest neighbor density estimation method commonly used in
        multi-objective optimization for diversity maintenance.
    """

    def __init__(self, k: int = 1) -> None:
        """Initialize k-nearest neighbor density estimator.
        
        Args:
            k (int, optional): Number of nearest neighbors to consider for
                density calculation. Defaults to 1 (distance to nearest neighbor).
        """
        super().__init__()
        self.k = k
        self.distance_matrix: List[List[float]] = []

    def compute_density_estimator(self, solutions: List[S]) -> None:
        """Compute k-nearest neighbor density for each solution.
        
        Calculates pairwise Euclidean distances between all solutions in
        objective space, then for each solution finds the distance to its
        k-th nearest neighbor. Stores this distance as the density value.
        
        Args:
            solutions (List[S]): List of solutions for which to compute density.
        
        Note:
            - Computes full pairwise distance matrix (O(n²) complexity)
            - Density is stored in solution.attributes["knn_density"]
            - If population size <= k, density is not computed
            - Uses Euclidean distance in objective space
        """
        solutions_size = len(solutions)
        if solutions_size <= self.k:
            return
        points = []
        for i in range(solutions_size):
            points.append(solutions[i].objectives)
        # Compute distance matrix
        self.distance_matrix = numpy.zeros(shape=(solutions_size, solutions_size))
        for i in range(solutions_size):
            for j in range(solutions_size):
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = euclidean(
                    solutions[i].objectives, solutions[j].objectives
                )
        # Gets the k-nearest distance of all the solutions
        for i in range(solutions_size):
            distances = []
            for j in range(solutions_size):
                distances.append(self.distance_matrix[i, j])
            distances.sort()
            solutions[i].attributes["knn_density"] = distances[self.k]

    def sort(self, solutions: List[S]) -> List[S]:
        """Sort solutions by k-nearest neighbor density.
        
        Sorts solutions in descending order of their k-th nearest neighbor
        distance (higher distance = less crowded = better). For solutions
        with equal k-th distance, uses (k+1)-th, (k+2)-th, etc. distances
        as tie-breakers.
        
        Args:
            solutions (List[S]): List of solutions to sort.
        
        Returns:
            List[S]: Sorted list of solutions (least crowded first).
        
        Note:
            - Computes sorted distance lists for all solutions
            - Stores distances in solution.attributes["distances_"]
            - Solutions with larger k-th distance come first (less crowded)
        """
        def compare(solution1, solution2):
            distances1 = solution1.attributes["distances_"]
            distances2 = solution2.attributes["distances_"]
            tmp_k = self.k
            if distances1[tmp_k] > distances2[tmp_k]:
                return -1
            elif distances1[tmp_k] < distances2[tmp_k]:
                return 1
            else:
                while tmp_k < (len(distances1) - 1):
                    tmp_k += 1
                    if distances1[tmp_k] > distances2[tmp_k]:
                        return -1
                    elif distances1[tmp_k] < distances2[tmp_k]:
                        return 1
            return 0

        for i in range(len(solutions)):
            distances = []
            for j in range(len(solutions)):
                distances.append(self.distance_matrix[i, j])
            distances.sort()
            solutions[i].attributes["distances_"] = distances
        solutions.sort(key=cmp_to_key(compare))

    @classmethod
    def get_comparator(cls) -> Comparator:
        """Get comparator for k-nearest neighbor density.
        
        Returns:
            Comparator: SolutionAttributeComparator that compares solutions
                based on "knn_density" attribute (lower is better = False,
                meaning higher density values are preferred).
        """
        return SolutionAttributeComparator("knn_density", lowest_is_best=False)
