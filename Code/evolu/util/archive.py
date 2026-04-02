# -*- coding: utf-8 -*-
import copy
import random
from abc import ABC, abstractmethod
from threading import Lock
from typing import Generic, List, TypeVar
from evolu.core.exceptions import EmptyFrontException
from evolu.util.comparator import (Comparator, EqualSolutionsComparator, IdenticalSolutionsComparator,
                                         DominanceWithConstraintsComparator, SolutionAttributeComparator)
from evolu.util.density_estimator import CrowdingDistance, DensityEstimator

from evolu.core.solution import Solution

S = TypeVar("S", bound=Solution)
"""
module:: archive
synopsis: Archive implementation.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Archive(Generic[S], ABC):
    """Base class for solution archives.
    
    Archives are data structures used to store and manage collections of solutions,
    typically non-dominated solutions in multi-objective optimization. They provide
    mechanisms for adding solutions while maintaining specific properties (e.g.,
    non-dominance, diversity, size limits).
    
    Attributes:
        solution_list (List[S]): List of solutions stored in the archive.
    
    Note:
        Subclasses must implement the add() method to define how solutions are
        added and managed.
    """
    
    def __init__(self) -> None:
        """Initialize an empty archive."""
        self.solution_list: List[S] = []

    @abstractmethod
    def add(self, solution: S) -> bool:
        """Add a solution to the archive.
        
        Args:
            solution (S): Solution to add to the archive.
        
        Returns:
            bool: True if the solution was successfully added, False otherwise.
                A solution might not be added if it's dominated, duplicate, or
                doesn't meet other archive-specific criteria.
        
        Note:
            The implementation may modify the archive by removing dominated
            solutions or managing archive size limits.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `add`.")

    def get(self, index: int) -> S:
        """Get a solution from the archive by index.
        
        Args:
            index (int): Index of the solution to retrieve.
        
        Returns:
            S: Solution at the specified index.
        """
        return self.solution_list[index]

    def size(self) -> int:
        """Get the number of solutions in the archive.
        
        Returns:
            int: Number of solutions currently stored in the archive.
        """
        return len(self.solution_list)

    def get_name(self) -> str:
        """Get the name of the archive implementation.
        
        Returns:
            str: Class name of the archive.
        """
        return self.__class__.__name__


class NonDominatedSolutionsArchive(Archive[S]):
    """Archive that maintains only non-dominated solutions.
    
    This archive automatically maintains the Pareto-optimal set by removing
    any solutions that become dominated when new solutions are added. When
    adding a solution:
    - If it dominates existing solutions, those are removed.
    - If it's dominated by any existing solution, it's not added.
    - If it's non-dominated, it's added and kept sorted.
    
    Attributes:
        comparator (Comparator[S]): Comparator used to determine dominance
            relationships between solutions.
    
    Note:
        Solutions are kept sorted by objective values for efficient management.
        This archive has no size limit and can grow as more non-dominated
        solutions are discovered.
    """
    
    def __init__(self, dominance_comparator: Comparator[S] = DominanceWithConstraintsComparator()) -> None:
        """Initialize non-dominated solutions archive with a comparator.
        
        Args:
            dominance_comparator (Comparator[S], optional): Comparator for
                dominance checking. Defaults to DominanceWithConstraintsComparator().
        """
        super(NonDominatedSolutionsArchive, self).__init__()
        self.comparator = dominance_comparator

    def add(self, solution: S) -> bool:
        is_dominated = False
        is_contained = False
        if len(self.solution_list) == 0:
            self.solution_list.append(copy.deepcopy(solution))
            return True
        else:
            number_of_deleted_solutions = 0
            # New copy of list and enumerate
            for index, current_solution in enumerate(list(self.solution_list)):
                is_dominated_flag = self.comparator.compare(solution, current_solution)
                if is_dominated_flag == -1:
                    del self.solution_list[index - number_of_deleted_solutions]
                    number_of_deleted_solutions += 1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break
                elif is_dominated_flag == 0:
                    if solution.objectives == current_solution.objectives:
                        is_contained = True
                        break
        if not is_dominated and not is_contained:
            if len(self.solution_list) == 0:
                self.solution_list.append(copy.deepcopy(solution))
                return True
            else:
                for index in range(0, len(self.solution_list)):
                    for i in range(0, solution.number_of_objectives):
                        if solution.objectives[i] <= self.solution_list[index].objectives[i]:
                            if solution.objectives[i] < self.solution_list[index].objectives[i]:
                                self.solution_list.insert(index, copy.deepcopy(solution))
                                return True
                        else:
                            break
            self.solution_list.append(copy.deepcopy(solution))
            return True
        return False


class ModifiedNonDominatedSolutionsArchive(Archive[S]):
    def __init__(self, dominance_comparator: Comparator = DominanceWithConstraintsComparator()):
        super(ModifiedNonDominatedSolutionsArchive, self).__init__()
        self.comparator = dominance_comparator
        self.equal_solutions_comparator = EqualSolutionsComparator()
        self.identical_solutions_comparator = IdenticalSolutionsComparator()

    def add(self, solution: S) -> bool:
        is_dominated = False
        is_contained = False
        if len(self.solution_list) == 0:
            self.solution_list.append(copy.deepcopy(solution))
            self.solution_list[len(self.solution_list) - 1].survive_time = 0
            return True
        else:
            number_of_deleted_solutions = 0
            # New copy of list and enumerate
            for index, current_solution in enumerate(list(self.solution_list)):
                is_dominated_flag = self.comparator.compare(solution, current_solution)
                if is_dominated_flag == -1:
                    del self.solution_list[index - number_of_deleted_solutions]
                    number_of_deleted_solutions += 1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break
                elif is_dominated_flag == 0:
                    if self.equal_solutions_comparator.compare(solution, current_solution) == 0:
                        if self.identical_solutions_comparator.compare(solution, current_solution) != 0:
                            survive_time = self.solution_list[index].survive_time
                            self.solution_list[index] = copy.deepcopy(solution)
                            self.solution_list[index].survive_time = survive_time
                        is_contained = True
                        break
        if not is_dominated and not is_contained:
            if len(self.solution_list) == 0:
                self.solution_list.append(copy.deepcopy(solution))
                self.solution_list[len(self.solution_list) - 1].survive_time = 0
                return True
            else:
                for index in range(0, len(self.solution_list)):
                    for i in range(0, solution.number_of_objectives):
                        if solution.objectives[i] <= self.solution_list[index].objectives[i]:
                            if solution.objectives[i] < self.solution_list[index].objectives[i]:
                                self.solution_list.insert(index, copy.deepcopy(solution))
                                self.solution_list[index].survive_time = 0
                                return True
                        else:
                            break
            self.solution_list.append(copy.deepcopy(solution))
            self.solution_list[len(self.solution_list) - 1].survive_time = 0
            return True
        return False


class BoundedArchive(Archive[S]):
    def __init__(self, maximum_size: int, dominance_comparator: Comparator, comparator: Comparator[S] = None,
                 density_estimator: DensityEstimator = None):
        super(BoundedArchive, self).__init__()
        self.maximum_size = maximum_size
        self.dominance_comparator = dominance_comparator
        self.comparator = comparator
        self.density_estimator = density_estimator
        self.non_dominated_solution_archive = NonDominatedSolutionsArchive(dominance_comparator)
        self.solution_list = self.non_dominated_solution_archive.solution_list

    def compute_density_estimator(self):
        self.density_estimator.compute_density_estimator(self.solution_list)

    def add(self, solution: S) -> bool:
        success = self.non_dominated_solution_archive.add(solution)
        if success:
            if self.size() > self.maximum_size:
                self.compute_density_estimator()
                worst_solution, index_to_remove = self.__find_worst_solution(self.solution_list)
                self.solution_list.pop(index_to_remove)
        return success

    def __find_worst_solution(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise EmptyFrontException("The solution list is None")
        elif len(solution_list) == 0:
            raise EmptyFrontException("The solution list is empty")
        worst_solution = solution_list[0]
        index_to_remove = 0
        for solution_index, solution in enumerate(solution_list[1:]):
            if self.comparator.compare(worst_solution, solution) < 0:
                worst_solution = solution
                index_to_remove = solution_index + 1
        return worst_solution, index_to_remove


class BoundedCrowdingDistanceArchive(BoundedArchive[S]):
    """Bounded archive using crowding distance for size control.
    
    A bounded archive that maintains a maximum number of non-dominated
    solutions. When the archive exceeds maximum_size, solutions with the
    smallest crowding distance (most crowded) are removed first.
    
    Uses:
    - DominanceWithConstraintsComparator for dominance checking
    - CrowdingDistance for density estimation
    - SolutionAttributeComparator on "crowding_distance" for sorting
    
    This archive is commonly used in algorithms like NSGA-II for maintaining
    diversity while limiting archive size.
    
    Args:
        maximum_size (int): Maximum number of solutions to maintain in archive.
    """
    
    def __init__(self, maximum_size: int):
        super(BoundedCrowdingDistanceArchive, self).__init__(
            maximum_size=maximum_size,
            dominance_comparator=DominanceWithConstraintsComparator(),
            comparator=SolutionAttributeComparator("crowding_distance", lowest_is_best=False),
            density_estimator=CrowdingDistance(),
        )


class ArchiveWithReferencePoint(BoundedArchive[S]):
    """Bounded archive that filters solutions based on a reference point.
    
    This archive extends BoundedArchive by adding reference point-based filtering.
    Solutions that are non-dominated with the reference point (equivalent) may
    be filtered or accepted probabilistically. This is useful for reference point
    based algorithms like R-NSGA-II.
    
    The archive maintains solutions that dominate or are incomparable with the
    reference point. Solutions equivalent to the reference point are handled
    with a small probability (5%) to maintain diversity.
    
    Attributes:
        maximum_size (int): Maximum number of solutions in archive.
        __reference_point (List[float]): Reference point in objective space.
        lock (Lock): Thread lock for thread-safe operations.
    
    Note:
        - Thread-safe implementation using locks
        - Solutions non-dominated with reference point accepted with 5% probability
        - Reference point can be updated dynamically
    """
    
    def __init__(self,
                 maximum_size: int,
                 reference_point: List[float],
                 comparator: Comparator[S],
                 density_estimator: DensityEstimator,
                 ):
        """Initialize archive with reference point.
        
        Args:
            maximum_size (int): Maximum number of solutions to maintain.
            reference_point (List[float]): Reference point in objective space
                for filtering solutions.
            comparator (Comparator[S]): Comparator for solution comparison.
            density_estimator (DensityEstimator): Density estimator for size control.
        """
        super(ArchiveWithReferencePoint, self).__init__(maximum_size, comparator, density_estimator)
        self.__reference_point = reference_point
        self.__comparator = comparator
        self.__density_estimator = density_estimator
        self.lock = Lock()

    def add(self, solution: S) -> bool:
        """Add solution to archive with reference point filtering.
        
        Adds solution if it dominates or is incomparable with the reference point.
        Solutions equivalent to the reference point are accepted with 5% probability.
        Uses thread-safe locking for concurrent access.
        
        Args:
            solution (S): Solution to add.
        
        Returns:
            bool: True if solution was added, False otherwise.
        
        Note:
            - Thread-safe operation
            - Filters solutions based on dominance relation with reference point
            - Maintains archive size using density estimator when full
        """
        with self.lock:
            dominated_solution = None
            if self.__dominance_test(solution.objectives, self.__reference_point) == 0:
                if len(self.solution_list) == 0:
                    result = True
                else:
                    if random.uniform(0.0, 1.0) < 0.05:
                        result = True
                        dominated_solution = solution
                    else:
                        result = False
            else:
                result = True
            if result:
                result = super(ArchiveWithReferencePoint, self).add(solution)
            if result and dominated_solution is not None and len(self.solution_list) > 1:
                if dominated_solution in self.solution_list:
                    self.solution_list.remove(dominated_solution)
            if result and len(self.solution_list) > self.maximum_size:
                self.compute_density_estimator()
        return result

    def filter(self):
        # In case of having at least a solution which is non-dominated with the reference point, filter it
        if len(self.solution_list) > 1:
            self.solution_list[:] = [
                sol for sol in self.solution_list if self.__dominance_test(sol.objectives, self.__reference_point) != 0
            ]

    def update_reference_point(self, new_reference_point) -> None:
        with self.lock:
            self.__reference_point = new_reference_point
            first_solution = copy.deepcopy(self.solution_list[0])
            self.filter()
            if len(self.solution_list) == 0:
                self.solution_list.append(first_solution)

    def get_reference_point(self) -> List[float]:
        with self.lock:
            return self.__reference_point

    def __dominance_test(self, vector1: List[float], vector2: List[float]) -> int:
        best_is_one = 0
        best_is_two = 0
        for value1, value2 in zip(vector1, vector2):
            if value1 != value2:
                if value1 < value2:
                    best_is_one = 1
                if value2 < value1:
                    best_is_two = 1
        if best_is_one > best_is_two:
            result = -1
        elif best_is_two > best_is_one:
            result = 1
        else:
            result = 0
        return result


class CrowdingDistanceArchiveWithReferencePoint(ArchiveWithReferencePoint[S]):
    """Bounded archive with reference point using crowding distance.
    
    A specialized archive that combines reference point filtering with
    crowding distance-based size control. When archive size exceeds
    maximum_size, solutions with smallest crowding distance are removed.
    
    This archive is ideal for reference point based algorithms that need
    to maintain diversity while focusing solutions around a reference point.
    
    Args:
        maximum_size (int): Maximum number of solutions to maintain.
        reference_point (List[float]): Reference point in objective space.
    
    Note:
        - Uses CrowdingDistance for density estimation
        - Uses SolutionAttributeComparator on "crowding_distance"
        - Extends ArchiveWithReferencePoint with crowding distance
    """
    
    def __init__(self, maximum_size: int, reference_point: List[float]):
        super(CrowdingDistanceArchiveWithReferencePoint, self).__init__(
            maximum_size=maximum_size,
            reference_point=reference_point,
            comparator=SolutionAttributeComparator("crowding_distance", lowest_is_best=False),
            density_estimator=CrowdingDistance(),
        )
