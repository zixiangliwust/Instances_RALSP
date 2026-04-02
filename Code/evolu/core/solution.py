# -*- coding: utf-8 -*-
from abc import ABC
from typing import Generic, List, TypeVar

from evolu.util.checking import Check

BitSet = List[bool]
S = TypeVar("S")


class Solution(Generic[S], ABC):
    """Base class representing candidate solutions.

    This class stores decision variables, objective values, constraint
    violations and additional attributes common to all solution types.

    Attributes:
        number_of_variables (int): Number of decision variables.
        number_of_objectives (int): Number of objective functions.
        number_of_constraints (int): Number of constraint functions.
        variables (List[List]): Decision variable values for each variable.
        objectives (List[float]): Objective function values.
        fitness (float | None): Scalar fitness value, if used by the algorithm.
        constraints (List[float]): Constraint violation values.
        attributes (dict): Dictionary for storing arbitrary metadata.
        solution_name (str): Human-readable name of the solution.
        solution_type (str): Encoding type (e.g., ``\"Float\"``, ``\"Permutation\"``).
        survive_time (int): Number of iterations the solution has survived.
    """

    def __init__(
        self,
        number_of_variables: int = 1,
        number_of_objectives: int = 1,
        number_of_constraints: int = 0,
    ):
        """Initialize a generic solution.

        Args:
            number_of_variables (int): Number of decision variables.
            number_of_objectives (int): Number of objective functions.
            number_of_constraints (int): Number of constraint functions.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.fitness = None
        self.constraints = [0.0 for _ in range(self.number_of_constraints)]
        self.attributes = {}
        self.solution_name = ""
        self.solution_type = ""
        self.survive_time = 0

    def __eq__(self, solution) -> bool:
        """Return True if two solutions have the same decision variables."""
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        """Return a string representation of the solution."""
        return "Solution(variables={},objectives={},constraints={})".format(
            self.variables, self.objectives, self.constraints
        )


class FloatSolution(Solution[float]):
    """Solution with continuous (float) decision variables.

    Attributes:
        lower_bound (List[float]): Lower bounds for each variable.
        upper_bound (List[float]): Upper bounds for each variable.
    """

    def __init__(
        self,
        lower_bound: List[float] = [0.0],
        upper_bound: List[float] = [1.0],
        number_of_variables: int = 1,
        number_of_objectives: int = 1,
        number_of_constraints: int = 0,
    ):
        super(FloatSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.solution_type = "Float"

    def __copy__(self):
        """Create a shallow copy of the float solution."""
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.fitness = self.fitness
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()
        return new_solution


class PermutationSolution(Solution):
    """Solution with permutation-based decision variables."""

    def __init__(
        self,
        number_of_variables: int = 1,
        number_of_objectives: int = 1,
        number_of_constraints: int = 0,
    ):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.solution_type = "Permutation"

    def __copy__(self):
        """Create a shallow copy of the permutation solution."""
        new_solution = PermutationSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.fitness = self.fitness
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.attributes = self.attributes.copy()
        return new_solution


class IntegerSolution(Solution[int]):
    """Solution with integer decision variables.

    Attributes:
        lower_bound (List[float]): Lower bounds for each variable.
        upper_bound (List[float]): Upper bounds for each variable.
    """

    def __init__(
        self,
        lower_bound: List[float] = [0.0],
        upper_bound: List[float] = [1.0],
        number_of_variables: int = 1,
        number_of_objectives: int = 1,
        number_of_constraints: int = 0,
    ):
        super(IntegerSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.solution_type = "Integer"

    def __copy__(self):
        """Create a shallow copy of the integer solution."""
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.fitness = self.fitness
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()
        return new_solution


class BinaryArraySolution(Solution[BitSet]):
    """Solution with binary array decision variables."""

    def __init__(
        self,
        number_of_variables: int = 1,
        number_of_objectives: int = 1,
        number_of_constraints: int = 0,
    ):
        super(BinaryArraySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.solution_type = "BinaryArray"
        self.bits_per_variable = []

    def __copy__(self):
        """Create a shallow copy of the binary array solution."""
        new_solution = BinaryArraySolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
        )
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.attributes = self.attributes.copy()
        new_solution.bits_per_variable = self.bits_per_variable
        return new_solution

    def get_total_number_of_bits(self) -> int:
        """Return the total number of bits across all variables."""
        total = 0
        for var in self.variables:
            total += len(var)
        return total

    def get_binary_string(self) -> str:
        """Return a string representation of the first binary variable."""
        string = ""
        for bit in self.variables[0]:
            string += "1" if bit else "0"
        return string

    def cardinality(self, variable_index: int) -> int:
        """Return the number of set bits in the specified variable."""
        return sum(1 for _ in self.variables[variable_index] if _)


class CompositeSolution(Solution):
    """Solution composed of a list of sub-solutions.

    Each decision variable can itself be a solution of any supported encoding
    (float, permutation, integer, binary, etc.), enabling mixed encodings.
    All sub-solutions must share the same number of objectives and constraints.

    Attributes:
        sub_solutions (List[Solution]): List of sub-solutions composing this solution.
    """

    def __init__(self, solutions: List[Solution]):
        super(CompositeSolution, self).__init__(
            len(solutions), solutions[0].number_of_objectives, solutions[0].number_of_constraints
        )
        Check.is_not_none(solutions)
        Check.collection_is_not_empty(solutions)
        for solution in solutions:
            Check.that(
                solution.number_of_objectives == solutions[0].number_of_objectives,
                "The solutions in the list must have the same number of objectives: "
                + str(solutions[0].number_of_objectives),
            )
            Check.that(
                solution.number_of_constraints == solutions[0].number_of_constraints,
                "The solutions in the list must have the same number of constraints: "
                + str(solutions[0].number_of_constraints),
            )
        self.sub_solutions = solutions

    def __copy__(self):
        new_solution = CompositeSolution(self.sub_solutions)
        new_solution.fitness = self.fitness
        new_solution.objectives = self.objectives[:]
        new_solution.constraints = self.constraints[:]
        new_solution.attributes = self.attributes.copy()
        return new_solution
