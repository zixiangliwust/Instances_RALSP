# -*- coding: utf-8 -*-
import copy
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from evolu.core.problem import Problem
from evolu.core.solution import Solution

S = TypeVar("S", bound=Solution)
R = TypeVar("R", bound=Solution)

"""
module:: generator
synopsis: Population generators implementation.
moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Generator(Generic[R], ABC):
    """Base class for solution generators.
    
    Generators create initial solutions or populations for optimization algorithms.
    Different generators may use different strategies: random generation, seeding
    with known good solutions, or domain-specific heuristics.
    
    Note:
        Subclasses must implement create_solution() method.
    """
    
    @abstractmethod
    def create_solution(self, problem: Problem[R]) -> R:
        """Create a single solution.
        
        Args:
            problem (Problem[R]): Problem instance for which to create a solution.
        
        Returns:
            R: A newly created solution instance.
        
        Note:
            The solution is typically not evaluated. Call problem.evaluate_solution()
            separately if evaluation is needed.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `create_solution`.")

    def create_solutions(self, problem: Problem[R], population_size: int) -> List[R]:
        """Create multiple solutions (population).
        
        Default implementation calls create_solution() repeatedly.
        Subclasses may override for more efficient batch generation.
        
        Args:
            problem (Problem[R]): Problem instance for which to create solutions.
            population_size (int): Number of solutions to create.
        
        Returns:
            List[R]: List of newly created solution instances.
        """
        return [self.create_solution(problem) for _ in range(population_size)]


class RandomGenerator(Generator[R]):
    """Random solution generator.
    
    This generator creates solutions by calling problem.create_solution(),
    which typically generates random feasible solutions according to the
    problem's encoding (e.g., random permutation, random float vector).
    
    This is the most commonly used generator for initializing populations
    in evolutionary algorithms.
    """
    
    def create_solution(self, problem: Problem[R]) -> R:
        """Create a random solution.
        
        Args:
            problem (Problem[R]): Problem instance.
        
        Returns:
            R: Randomly generated solution.
        """
        return problem.create_solution()

    def create_solutions(self, problem: Problem[R], population_size: int) -> List[R]:
        """Create multiple random solutions.
        
        Args:
            problem (Problem[R]): Problem instance.
            population_size (int): Number of solutions to create.
        
        Returns:
            List[R]: List of randomly generated solutions.
        """
        return [problem.create_solution() for _ in range(0, population_size)]


class InjectorGenerator(Generator[R]):
    """Solution generator that injects pre-existing solutions.
    
    This generator first returns solutions from a provided list (seed solutions),
    then falls back to random generation once the seed solutions are exhausted.
    Useful for:
    - Seeding algorithms with known good solutions
    - Hybrid algorithms that combine heuristic and random initialization
    - Warm-starting optimization from previous runs
    
    Attributes:
        population (List[R]): List of seed solutions to inject. Solutions are
            removed from this list as they are returned.
    """
    
    def __init__(self, solutions: List[R]) -> None:
        """Initialize with a list of solutions to inject.
        
        Args:
            solutions (List[R]): Seed solutions to inject. A deep copy is made
                so the original list is not modified.
        """
        super(InjectorGenerator, self).__init__()
        self.population = copy.deepcopy(solutions)

    def create_solution(self, problem: Problem[R]) -> R:
        """Create a solution by injecting from the list or generating a new one.
        
        If seed solutions are available, returns one and removes it from the list.
        Once all seed solutions are exhausted, generates random solutions.
        
        Args:
            problem (Problem[R]): Problem instance (used only if seed solutions
                are exhausted).
        
        Returns:
            R: Seed solution (if available) or randomly generated solution.
        """
        if len(self.population) > 0:
            # If we have more solutions to inject, return one from the list
            return self.population.pop()
        else:
            # Otherwise generate a new solution
            solution = problem.create_solution()
        return solution
