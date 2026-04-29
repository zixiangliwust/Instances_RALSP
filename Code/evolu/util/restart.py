# -*- coding: utf-8 -*-
import copy
from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from evolu.core.problem import Problem
from evolu.util.comparator import (Comparator, ObjectiveComparator, IdenticalSolutionsComparator, MultiComparator)

from evolu.core.solution import Solution

S = TypeVar("S", bound=Solution)


class Restart(Generic[S], ABC):
    """Base class for restart operators.
    
    Restart operators handle population diversity maintenance by detecting and
    handling duplicated solutions. They are used to prevent premature convergence
    and maintain exploration in evolutionary algorithms.
    
    Note:
        Subclasses must implement the execute() method to define the restart strategy.
    """
    
    def __init__(self) -> None:
        """Initialize restart operator."""
        pass

    @abstractmethod
    def execute(self, solution_list: List[S]) -> List[S]:
        """Execute restart operation on the solution list.
        
        Args:
            solution_list (List[S]): List of solutions to process.
        
        Returns:
            List[S]: Modified solution list with duplicated solutions handled
                according to the restart strategy.
        
        Note:
            Different restart strategies may:
            - Mark duplicated solutions as invalid (e.g., set objectives to large values)
            - Replace duplicated solutions with new random solutions
            - Remove duplicated solutions
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `execute`.")

    def get_name(self) -> str:
        """Get the name of the restart operator.
        
        Returns:
            str: Class name of the restart operator.
        """
        return self.__class__.__name__


class SimpleReplaceDuplicatedSolution(Restart[S]):
    """Restart operator that marks duplicated solutions as invalid.
    
    This operator detects identical solutions and marks the duplicates by setting
    their objective values to a large value (1.0e30), effectively making them
    the worst solutions. This ensures they are unlikely to be selected in future
    operations.
    
    Attributes:
        identical_solutions_comparator (IdenticalSolutionsComparator): Comparator
            used to detect identical solutions.
    
    Note:
        The original solution in a duplicate pair is kept, while duplicates are
        marked as invalid. This preserves the first occurrence while discouraging
        duplicate selection.
    """
    
    def __init__(self) -> None:
        """Initialize simple replace duplicated solution restart operator."""
        super(SimpleReplaceDuplicatedSolution, self).__init__()
        self.identical_solutions_comparator = IdenticalSolutionsComparator()

    def execute(self, solution_list: List[S]) -> List[S]:
        """Mark duplicated solutions as invalid.
        
        Args:
            solution_list (List[S]): List of solutions to process.
        
        Returns:
            List[S]: Modified solution list where duplicated solutions have
                their objectives set to 1.0e30 (effectively worst possible).
        """
        result_list = copy.deepcopy(solution_list)
        for j in range(0, len(result_list) - 1):
            for k in range(j + 1, len(result_list)):
                if self.identical_solutions_comparator.compare(result_list[j], result_list[k]) == 0:
                    for i in range(0, result_list[k].number_of_objectives):
                        result_list[k].objectives[i] = 1.0e30
        return result_list

    def get_name(self) -> str:
        """Get the name of the restart operator.
        
        Returns:
            str: "SimpleReplaceDuplicatedSolution"
        """
        return self.__class__.__name__


class ReplaceDuplicatedSolutionWithNewSolution(Restart[S]):
    """Restart operator that replaces duplicated solutions with new random solutions.
    
    This operator detects identical solutions and replaces duplicates with newly
    generated random solutions from the problem. This maintains population size
    while introducing diversity.
    
    Attributes:
        identical_solutions_comparator (IdenticalSolutionsComparator): Comparator
            used to detect identical solutions.
        problem (Problem[S]): Problem instance for creating new solutions.
    
    Note:
        The original solution in a duplicate pair is kept, while duplicates are
        replaced with new random solutions. This strategy is more aggressive than
        SimpleReplaceDuplicatedSolution as it introduces new genetic material.
    """
    
    def __init__(self, problem: Problem[S]) -> None:
        """Initialize replace duplicated solution with new solution restart operator.
        
        Args:
            problem (Problem[S]): Problem instance for creating new solutions
                to replace duplicates.
        """
        super(ReplaceDuplicatedSolutionWithNewSolution, self).__init__()
        self.identical_solutions_comparator = IdenticalSolutionsComparator()
        self.problem = problem

    def execute(self, solution_list: List[S]) -> List[S]:
        """Replace duplicated solutions with new random solutions.
        
        Args:
            solution_list (List[S]): List of solutions to process.
        
        Returns:
            List[S]: Modified solution list where duplicated solutions have been
                replaced with new random solutions from the problem.
        """
        result_list = copy.deepcopy(solution_list)
        for j in range(0, len(result_list) - 1):
            for k in range(j + 1, len(result_list)):
                if self.identical_solutions_comparator.compare(result_list[j], result_list[k]) == 0:
                    # Create new solution and evaluate it
                    new_solution = self.problem.create_solution()
                    new_solution = self.problem.evaluate_solution(new_solution)
                    result_list[k] = new_solution
        return result_list

    def get_name(self) -> str:
        """Get the name of the restart operator.
        
        Returns:
            str: "ReplaceDuplicatedSolutionWithNewSolution"
        """
        return self.__class__.__name__
