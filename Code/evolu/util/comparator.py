# -*- coding: utf-8 -*-
import math
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from evolu.core.solution import Solution
from evolu.core.exceptions import InvalidSolutionException
from evolu.util.distance import EuclideanDistance
from evolu.util.constraint_handling import overall_constraint_violation_degree

S = TypeVar("S", bound=Solution)


class Comparator(Generic[S], ABC):
    """Base class for comparing solutions.
    
    Comparators are used to determine the relative quality of solutions.
    The compare method returns:
    - -1 if solution1 is better than solution2
    - 1 if solution2 is better than solution1
    - 0 if they are equivalent
    
    Note:
        Subclasses must implement the compare() method.
    """
    
    @abstractmethod
    def compare(self, solution1: S, solution2: S) -> int:
        """Compare two solutions.
        
        Args:
            solution1 (S): First solution to compare.
            solution2 (S): Second solution to compare.
        
        Returns:
            int: Comparison result:
                - -1: solution1 is better than solution2
                - 1: solution2 is better than solution1
                - 0: solutions are equivalent
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `compare`.")


class MultiComparator(Comparator):
    """Comparator that uses multiple comparators in sequence.
    
    This comparator takes a list of comparators and checks all of them iteratively
    until a non-zero value is obtained or the list becomes empty. This allows for
    hierarchical comparison criteria.
    
    Attributes:
        comparator_list (List[Comparator[S]]): List of comparators to apply in order.
    """

    def __init__(self, comparator_list: List[Comparator[S]]):
        """Initialize with a list of comparators.
        
        Args:
            comparator_list (List[Comparator[S]]): List of comparators to apply sequentially.
        """
        self.comparator_list: List[Comparator[S]] = comparator_list

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        for comparator in self.comparator_list:
            flag = comparator.compare(solution1, solution2)
            if flag != 0:
                return flag
        return 0


class OverallConstraintViolationComparator(Comparator):
    """Comparator that prioritizes feasible solutions over infeasible ones.
    
    This comparator compares solutions based on their constraint violation degrees.
    Feasible solutions (violation degree = 0) are always preferred over infeasible
    solutions. Among infeasible solutions, those with smaller violation degrees
    are considered better.
    
    Returns:
        - -1: solution1 is better (either feasible when solution2 is infeasible,
              or both infeasible with solution1 having smaller violation)
        - 1: solution2 is better
        - 0: both solutions have equal constraint violation status
    """
    
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        violation_degree_solution_1 = overall_constraint_violation_degree(solution1)
        violation_degree_solution_2 = overall_constraint_violation_degree(solution2)
        if violation_degree_solution_1 < 0 and violation_degree_solution_2 < 0:
            if violation_degree_solution_1 > violation_degree_solution_2:
                result = -1
            elif violation_degree_solution_1 < violation_degree_solution_2:
                result = 1
            else:
                result = 0
        elif violation_degree_solution_1 == 0 and violation_degree_solution_2 < 0:
            result = -1
        elif violation_degree_solution_1 < 0 and violation_degree_solution_2 == 0:
            result = 1
        else:
            result = 0
        return result


class ObjectiveComparator(Comparator):
    """Comparator that compares solutions based on a single objective value.
    
    Compares two solutions by comparing their objective values at a specified
    objective index. Supports both ascending (minimization) and descending
    (maximization) order comparison.
    
    Attributes:
        objective_index (int): Index of the objective to compare.
        ascending_order (bool): If True, lower values are better (minimization).
            If False, higher values are better (maximization).
    
    Example:
        >>> comparator = ObjectiveComparator(objective_index=0, descending_order=False)
        >>> # Compares solutions by first objective in ascending order (minimize)
    """
    
    def __init__(self, objective_index: int = 0, descending_order: bool = False):
        """Initialize objective comparator.
        
        Args:
            objective_index (int, optional): Index of objective to compare.
                Defaults to 0 (first objective).
            descending_order (bool, optional): If True, higher values are better
                (maximization). If False, lower values are better (minimization).
                Defaults to False.
        """
        self.objective_index = objective_index
        self.ascending_order = not descending_order

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        value1 = solution1.objectives[self.objective_index]
        value2 = solution2.objectives[self.objective_index]
        if self.ascending_order:
            if value1 < value2:
                return -1
            elif value1 > value2:
                return 1
            else:
                return 0
        else:
            if value1 < value2:
                return 1
            elif value1 > value2:
                return -1
            else:
                return 0


class EpsilonObjectiveComparator(Comparator):
    """Comparator with epsilon tolerance for objective comparison.
    
    Compares solutions by a single objective value, using epsilon tolerance
    to handle numerical precision issues. Uses value1 / (1 + epsilon) for
    comparison to account for floating-point errors.
    
    Attributes:
        objective_index (int): Index of the objective to compare.
        __EPS (float): Epsilon tolerance value for numerical comparison.
    
    Note:
        This comparator is useful when comparing floating-point objective
        values where small differences should be treated as equal.
    """
    
    def __init__(self, objective_index: int = 0, epsilon: float = 1e-10):
        """Initialize epsilon objective comparator.
        
        Args:
            objective_index (int, optional): Index of objective to compare.
                Defaults to 0.
            epsilon (float, optional): Tolerance for numerical comparison.
                Defaults to 1e-10.
        """
        self.objective_index = objective_index
        self.__EPS = epsilon

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        value1 = solution1.objectives[self.objective_index]
        value2 = solution2.objectives[self.objective_index]
        if value1 / (1 + self.__EPS) < value2:
            return -1
        elif value1 / (1 + self.__EPS) > value2:
            return 1
        else:
            return 0


class EqualSolutionsComparator(Comparator):
    """Comparator that checks if solutions have equal objective values.
    
    Determines if two solutions have equal objective values across all objectives.
    Returns 0 if solutions have equal objectives, otherwise returns dominance
    relationship based on which solution has better objectives.
    
    Note:
        Solutions are considered equal if neither dominates the other
        (no objective is strictly better in either solution).
    """
    
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        dominate1 = 0
        dominate2 = 0
        for i in range(len(solution1.objectives)):
            value1 = solution1.objectives[i]
            value2 = solution2.objectives[i]
            if value1 < value2:
                flag = -1
            elif value1 > value2:
                flag = 1
            else:
                flag = 0
            if flag == -1:
                dominate1 = 1
            if flag == 1:
                dominate2 = 1
        if dominate1 == 0 and dominate2 == 0:
            return 0
        elif dominate1 == 1:
            return -1
        elif dominate2 == 1:
            return 1


class EpsilonEqualSolutionComparator(Comparator):
    """Comparator that checks if solutions are equal within epsilon tolerance.
    
    Compares solutions by their decision variables using Euclidean distance.
    Solutions are considered equal if their variables are within epsilon
    distance of each other. Used for detecting duplicate solutions in
    decision variable space.
    
    Attributes:
        __EPS (float): Epsilon tolerance for distance comparison (default 1e-10).
    
    Note:
        Uses Euclidean distance in decision variable space, not objective space.
    """
    
    __EPS = 1e-10

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        if solution1.number_of_variables != solution2.number_of_variables:
            return -1
        euclidean_distance = EuclideanDistance()
        if euclidean_distance.get_distance(solution1.variables, solution2.variables) < self.__EPS:
            return 0
        return -1


class IdenticalSolutionsComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        result = 0
        if not hasattr(solution1, "sub_solutions"):
            for i in range(0, solution1.number_of_variables):
                value1 = solution1.variables[i]
                value2 = solution2.variables[i]
                if value1 < value2:
                    result = -1
                    break
                elif value1 > value2:
                    result = 1
                    break
                else:
                    result = 0
        else:
            for j in range(0, len(solution1.sub_solutions)):
                for i in range(0, solution1.sub_solutions[j].number_of_variables):
                    value1 = solution1.sub_solutions[j].variables[i]
                    value2 = solution2.sub_solutions[j].variables[i]
                    if value1 < value2:
                        result = -1
                        break
                    elif value1 > value2:
                        result = 1
                        break
                    else:
                        result = 0
                if result != 0:
                    break
        return result


class SolutionAttributeComparator(Comparator):
    """Comparator that compares solutions based on an attribute value.
    
    Compares solutions by comparing a specific attribute stored in
    solution.attributes dictionary. Supports both minimization (lowest_is_best=True)
    and maximization (lowest_is_best=False) modes.
    
    Attributes:
        key (str): Key of the attribute to compare in solution.attributes.
        lowest_is_best (bool): If True, lower attribute values are better.
            If False, higher attribute values are better.
    
    Example:
        >>> comparator = SolutionAttributeComparator("crowding_distance", lowest_is_best=False)
        >>> # Compares solutions by crowding_distance, higher is better
    """
    
    def __init__(self, key: str, lowest_is_best: bool = True):
        """Initialize solution attribute comparator.
        
        Args:
            key (str): Attribute key to compare in solution.attributes.
            lowest_is_best (bool, optional): If True, lower values are better.
                Defaults to True.
        """
        self.key = key
        self.lowest_is_best = lowest_is_best

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        value1 = solution1.attributes.get(self.key)
        value2 = solution2.attributes.get(self.key)
        result = 0
        if value1 is not None and value2 is not None:
            if self.lowest_is_best:
                if value1 < value2:
                    result = -1
                elif value1 > value2:
                    result = 1
                else:
                    result = 0
            else:
                if value1 > value2:
                    result = -1
                elif value1 < value2:
                    result = 1
                else:
                    result = 0
        return result


class RankingAndCrowdingDistanceComparator(Comparator):
    """Comparator using dominance ranking and crowding distance.
    
    A two-level comparator used in NSGA-II and similar algorithms. First compares
    solutions by their dominance ranking (lower rank = better), then by crowding
    distance as a tie-breaker (higher crowding distance = better for diversity).
    
    Comparison priority:
    1. Dominance rank (solution.attributes["dominance_ranking"])
    2. Crowding distance (solution.attributes["crowding_distance"]) if ranks equal
    
    Note:
        This comparator is commonly used in NSGA-II for population sorting and
        selection to maintain both convergence and diversity.
    """
    
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        result = SolutionAttributeComparator("dominance_ranking").compare(solution1, solution2)
        if result == 0:
            result = SolutionAttributeComparator("crowding_distance", lowest_is_best=False).compare(
                solution1, solution2
            )
        return result


class RankingAndKNNDistanceComparator(Comparator):
    """Comparator using dominance ranking and k-nearest neighbor distance.
    
    A two-level comparator similar to RankingAndCrowdingDistanceComparator, but
    uses k-nearest neighbor density instead of crowding distance for tie-breaking.
    First compares by dominance rank, then by k-NN density (higher is better).
    
    Comparison priority:
    1. Dominance rank (solution.attributes["dominance_ranking"])
    2. K-NN density (solution.attributes["knn_density"]) if ranks equal
    
    Note:
        K-NN density is stored by KNearestNeighborDensityEstimator and represents
        the distance to the k-th nearest neighbor (larger = less crowded = better).
    """
    
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1
        result = SolutionAttributeComparator("dominance_ranking").compare(solution1, solution2)
        if result == 0:
            result = SolutionAttributeComparator("knn_density", lowest_is_best=False).compare(solution1, solution2)
        return result


class DominanceComparator(Comparator):
    """Comparator implementing Pareto dominance relation.
    
    Compares two solutions based on Pareto dominance. A solution dominates another
    if it is better in at least one objective and not worse in any objective.
    For minimization problems, "better" means smaller objective value.
    
    Returns:
        - -1: solution1 dominates solution2
        - 1: solution2 dominates solution1
        - 0: solutions are non-dominated (incomparable)
    
    Note:
        This is the fundamental comparison relation in multi-objective optimization.
        Raises InvalidSolutionException if either solution is None.
    """
    
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            raise InvalidSolutionException("Solution1 is None")
        elif solution2 is None:
            raise InvalidSolutionException("Solution2 is None")
        # return self.__dominance_test(solution1, solution2)
        return self.dominance_test(solution1.objectives, solution2.objectives)

    def __dominance_test(self, solution1: Solution, solution2: Solution) -> float:
        best_is_one = 0
        best_is_two = 0
        for i in range(solution1.number_of_objectives):
            value1 = solution1.objectives[i]
            value2 = solution2.objectives[i]
            if value1 != value2:
                if value1 < value2:
                    best_is_one = 1
                if value1 > value2:
                    best_is_two = 1
        if best_is_one > best_is_two:
            result = -1
        elif best_is_two > best_is_one:
            result = 1
        else:
            result = 0
        return result

    @staticmethod
    def dominance_test(vector1: [float], vector2: [float]) -> int:
        """Test dominance relation between two objective vectors.
        
        Static method for comparing two objective value vectors (not Solution objects).
        Useful for comparing objective vectors directly without creating Solution objects.
        
        Args:
            vector1 (List[float]): First objective vector.
            vector2 (List[float]): Second objective vector.
        
        Returns:
            int: Dominance relation:
                - -1: vector1 dominates vector2
                - 1: vector2 dominates vector1
                - 0: vectors are non-dominated
        
        Note:
            Assumes minimization of all objectives. For maximization, negate the vectors.
        """
        result = 0
        for i in range(len(vector1)):
            if vector1[i] > vector2[i]:
                if result == -1:
                    return 0
                result = 1
            elif vector2[i] > vector1[i]:
                if result == 1:
                    return 0
                result = -1
        return result


class DominanceWithConstraintsComparator(Comparator):
    """Comparator that considers constraints before dominance.
    
    A two-level comparator that first compares solutions by constraint violation,
    then by Pareto dominance if constraint violation is equal. This ensures that
    feasible solutions are always preferred over infeasible ones, and among
    feasible solutions, Pareto dominance is used.
    
    Attributes:
        constraint_comparator (Comparator): Comparator for constraint violation
            (default: OverallConstraintViolationComparator).
        dominance_comparator (DominanceComparator): Comparator for Pareto dominance.
    
    Note:
        This comparator is essential for constrained multi-objective optimization,
        where feasibility takes priority over objective values.
    """
    
    def __init__(self, constraint_comparator: Comparator = OverallConstraintViolationComparator()):
        """Initialize dominance with constraints comparator.
        
        Args:
            constraint_comparator (Comparator, optional): Comparator for constraint
                violation. Defaults to OverallConstraintViolationComparator.
        """
        self.constraint_comparator = constraint_comparator
        self.dominance_comparator = DominanceComparator()

    def compare(self, solution1: S, solution2: S) -> int:
        if solution1 is None:
            raise InvalidSolutionException("Solution1 is None")
        elif solution2 is None:
            raise InvalidSolutionException("Solution2 is None")
        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            result = self.dominance_comparator.compare(solution1, solution2)
        return result


class EpsilonDominanceComparator(Comparator):
    """Comparator implementing epsilon-dominance relation.
    
    Epsilon-dominance is a relaxation of Pareto dominance that uses an epsilon
    grid to reduce the number of non-dominated solutions. Solutions are compared
    in an epsilon-quantized objective space, and ties are broken by distance
    to the grid point.
    
    First compares by constraint violation, then by epsilon-dominance relation.
    This helps in maintaining archive diversity while reducing archive size.
    
    Attributes:
        constraint_comparator (Comparator): Comparator for constraint checking.
        __EPS (float): Epsilon value for objective space quantization.
    
    Reference:
        Laumanns, M., Thiele, L., Deb, K., & Zitzler, E. (2002). Combining
        convergence and diversity in evolutionary multiobjective optimization.
        Evolutionary computation, 10(3), 263-282.
    """
    
    def __init__(
            self,
            epsilon: float,
            constraint_comparator: Comparator = OverallConstraintViolationComparator(), ):
        """Initialize epsilon-dominance comparator.
        
        Args:
            epsilon (float): Epsilon value for objective space grid quantization.
            constraint_comparator (Comparator, optional): Comparator for constraint
                violation. Defaults to OverallConstraintViolationComparator.
        """
        self.constraint_comparator = constraint_comparator
        self.__EPS = epsilon

    def compare(self, solution1: Solution, solution2: Solution):
        if solution1 is None:
            raise InvalidSolutionException("Solution1 is None")
        elif solution2 is None:
            raise InvalidSolutionException("Solution2 is None")
        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            result = self.__dominance_test(solution1, solution2)
        return result

    def __dominance_test(self, solution1: Solution, solution2: Solution):
        best_is_one = False
        best_is_two = False
        for i in range(solution1.number_of_objectives):
            value1 = math.floor(solution1.objectives[i] / self.__EPS)
            value2 = math.floor(solution2.objectives[i] / self.__EPS)
            if value1 < value2:
                best_is_one = True
                if best_is_two:
                    return 0
            elif value2 < value1:
                best_is_two = True
                if best_is_one:
                    return 0
        if not best_is_one and not best_is_two:
            dist1 = 0.0
            dist2 = 0.0
            for i in range(solution1.number_of_objectives):
                index1 = math.floor(solution1.objectives[i] / self.__EPS)
                index2 = math.floor(solution2.objectives[i] / self.__EPS)
                dist1 += math.pow(solution1.objectives[i] - index1 * self.__EPS, 2.0)
                dist2 += math.pow(solution2.objectives[i] - index2 * self.__EPS, 2.0)
            if dist1 < dist2:
                return -1
            else:
                return 1
        else:
            if best_is_two:
                return 1
            else:
                return -1


class GDominanceComparator(DominanceComparator):
    """Comparator implementing g-dominance relation.
    
    G-dominance (geometric dominance) is a preference-based dominance relation
    that considers solutions based on a reference point. Solutions are first
    compared by their position relative to the reference point, then by standard
    Pareto dominance if both solutions are on the same side of the reference point.
    
    A solution dominates another if it is closer to or better relative to the
    reference point. This is useful for reference point-based algorithms like
    R-NSGA-II.
    
    Attributes:
        reference_point (tuple): Reference point in objective space.
    
    Reference:
        Deb, K., & Sundar, J. (2006). Reference point based multi-objective
        optimization using evolutionary algorithms. In Proceedings of the 8th
        annual conference on Genetic and evolutionary computation.
    """
    
    def __init__(self, reference_point: (), ):
        """Initialize g-dominance comparator.
        
        Args:
            reference_point (tuple): Reference point coordinates in objective space.
        """
        super(GDominanceComparator, self).__init__()
        self.reference_point = reference_point

    def compare(self, solution1: Solution, solution2: Solution):
        if solution1 is None:
            raise InvalidSolutionException("Solution1 is None")
        elif solution2 is None:
            raise InvalidSolutionException("Solution2 is None")
        if self.__flag(solution1) > self.__flag(solution2):
            result = -1
        elif self.__flag(solution1) < self.__flag(solution2):
            result = 1
        else:
            result = super(GDominanceComparator, self).compare(solution1, solution2)
        return result

    def __flag(self, solution: Solution):
        result = 1
        for i in range(solution.number_of_objectives):
            if solution.objectives[i] > self.reference_point[i]:
                result = 0
        if result == 0:
            result = 1
            for i in range(solution.number_of_objectives):
                if solution.objectives[i] < self.reference_point[i]:
                    result = 0
        return result
