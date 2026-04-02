from typing import List
from evolu.core.solution import Solution
from evolu.util.checking import Check


def is_feasible(solution: Solution) -> bool:
    """Check if a solution is feasible.
    
    A solution is considered feasible if it satisfies all constraints.
    In evolu, constraints are stored as a list where negative values indicate
    constraint violations. A solution is feasible when all constraint values
    are non-negative.
    
    Args:
        solution (Solution): Solution to check for feasibility.
    
    Returns:
        bool: True if the solution is feasible (no violated constraints),
            False otherwise.
    
    Example:
        >>> solution.constraints = [0.0, 0.0, 0.0]  # All constraints satisfied
        >>> is_feasible(solution)
        True
        
        >>> solution.constraints = [0.0, -1.5, 0.0]  # One constraint violated
        >>> is_feasible(solution)
        False
    """
    return number_of_violated_constraints(solution) == 0


def number_of_violated_constraints(solution: Solution) -> int:
    """Count the number of violated constraints in a solution.
    
    A constraint is considered violated if its value is negative. This function
    counts how many constraints are violated.
    
    Args:
        solution (Solution): Solution to check.
    
    Returns:
        int: Number of violated constraints (non-negative integer).
    
    Example:
        >>> solution.constraints = [0.0, -1.5, 0.0, -0.3]
        >>> number_of_violated_constraints(solution)
        2
    """
    return sum([1 for _ in solution.constraints if _ < 0])


def overall_constraint_violation_degree(solution: Solution) -> float:
    """Calculate the overall constraint violation degree of a solution.
    
    The constraint violation degree is the sum of all negative constraint values
    (i.e., the magnitude of violations). This provides a scalar measure of how
    much a solution violates constraints, useful for constraint-handling techniques
    like penalty methods.
    
    Args:
        solution (Solution): Solution to check.
    
    Returns:
        float: Sum of all negative constraint values. Returns 0.0 if the solution
            is feasible (no negative constraints), or a negative value indicating
            the total violation magnitude.
    
    Example:
        >>> solution.constraints = [0.0, -1.5, 0.0, -0.3]
        >>> overall_constraint_violation_degree(solution)
        -1.8
    """
    return sum([value for value in solution.constraints if value < 0])


def feasibility_ratio(solutions: List[Solution]) -> float:
    """Calculate the ratio of feasible solutions in a population.
    
    This function computes what fraction of the given solutions are feasible,
    providing a measure of population feasibility during optimization.
    
    Args:
        solutions (List[Solution]): List of solutions to check.
    
    Returns:
        float: Ratio of feasible solutions, ranging from 0.0 (no feasible solutions)
            to 1.0 (all solutions feasible).
    
    Raises:
        ValueError: If the solutions list is empty.
    
    Example:
        >>> population = [sol1, sol2, sol3, sol4]
        >>> # Assume sol1 and sol3 are feasible
        >>> feasibility_ratio(population)
        0.5
    """
    Check.that(len(solutions) > 0, "The solution list is empty")
    return sum(1 for solution in solutions if is_feasible(solution)) / len(solutions)
