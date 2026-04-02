"""Example script for running multi-objective optimization algorithms on MOReconfigurableALBP with integer representation.

This script demonstrates how to use multi-objective optimization algorithms
to solve Multi-Objective Reconfigurable Assembly Line Balancing Problem
(MOReconfigurableALBP) using integer solution representation.

Features:
- Runs multiple MOEAs on MOReconfigurableALBP problems
- Uses FloatPermutationIntegerSwapMutation and RepeatedPermutationOnePointCrossover
  operators for integer permutation solutions
- Progress bar observer for console output
- Real-time visualization of Pareto front evolution
- Static and interactive plots of final Pareto fronts
- Saves results to files

The integer representation directly uses product indices in the sequence, making
it more suitable for permutation-based operators that maintain valid sequences.
"""
import sys
import os
from typing import List
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.algorithm import Algorithm
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.solution import (get_non_dominated_solutions, print_function_values_to_file,
                                       print_variables_to_file)
from evolu.util.termination_criterion import StoppingByEvaluations
from evolureconfigurablealbp.problem.multiobjective.MO_ReconfigurableALS_problem import *


def select_and_run_algorithm(algorithm_name: str, file_path: str, problem_name: str) -> None:
    """Select and run a multi-objective optimization algorithm on the given MOReconfigurableALBP problem.
    
    This function creates an IntegerMOReconfigurableALBP problem instance,
    configures a multi-objective algorithm with permutation operators, runs it,
    and saves the Pareto front to files.
    
    Args:
        algorithm_name (str): Name of the algorithm to run. Supported algorithms:
            Local search: "MORS", "MSAA", "MRSA"
            Population-based: "MOABC", "IMOABC", "NSGAII", "NSGAIII", "SPEA2",
            "OMOPSO", "SMPSO", "SMPSORP"
        file_path (str): Path to the directory containing the problem instance file.
        problem_name (str): Name of the problem instance file to load.
    
    Note:
        The algorithm is configured with:
        - Maximum of 1000 function evaluations
        - FloatPermutationIntegerSwapMutation for permutation mutation
        - RepeatedPermutationOnePointCrossover for permutation crossover
        - Results (Pareto front) are saved to FUN.* and VAR.* files
        - Observers provide progress tracking and visualization
    """
    max_evaluations: int = 1000
    problem: IntegerMOReconfigurableALSP = IntegerMOReconfigurableALSP(file_path, problem_name)
    """
    Local search algorithms
    """
    if algorithm_name == "MORS":
        from evolu.optimizers.multiobjective.MORS import MORS
        algorithm = MORS(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        )
    if algorithm_name == "MSAA":
        from evolu.optimizers.multiobjective.MOSA import MSAA
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = MSAA(
            problem=problem,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MRSA":
        from evolu.optimizers.multiobjective.MOSA import MRSA
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = MRSA(
            problem=problem,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    """
    Population-based algorithms
    """
    if algorithm_name == "MOABC":
        from evolu.optimizers.multiobjective.MOABC import MOABC
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = MOABC(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "IMOABC":
        from evolu.optimizers.multiobjective.MOABC import IMOABC
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import RepeatedPermutationOnePointCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = IMOABC(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=RepeatedPermutationOnePointCrossover(probability=1.0),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "NSGAII":
        from evolu.optimizers.multiobjective import NSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import RepeatedPermutationOnePointCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=RepeatedPermutationOnePointCrossover(probability=1.0),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            dominance_comparator=DominanceWithConstraintsComparator(),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )

    algorithm.run()
    front: List[Solution] = algorithm.get_result()
    front = get_non_dominated_solutions(front)
    print_function_values_to_file(front, "results/FUN." + algorithm.label)
    print_variables_to_file(front, "results/VAR." + algorithm.label)
    print(len(front))
    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    """Main execution block.
    
    This block demonstrates how to run multiple multi-objective algorithms on
    MOReconfigurableALBP problem instances with integer representation. The script
    loads problem instances from a directory and runs algorithms, generating
    plots and saving results.
    
    Example:
        To run different algorithms or problems, modify the `algorithm_name_list`
        and `file_path` variables. Adjust num_of_runs for multiple independent
        runs for statistical analysis.
    """
    file_path: str = "D:/GitHubInstances/Instances_RALSP/Instances/"
    # Get all entries in the directory
    entries: List[str] = os.listdir(file_path)
    # Filter out files
    problem_name_list: List[str] = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]
    # problem_name_list = problem_name_list[0:10]

    algorithm_name_list: List[str] = [
        "MORS", "MSAA", "MRSA",
        "MOABC", "IMOABC", "NSGAII", 
    ]
    num_of_runs: int = 1
    for algorithm_name in algorithm_name_list:
        for problem_index in range(0, len(problem_name_list)):
            for num_run in range(0, num_of_runs):
                select_and_run_algorithm(algorithm_name, file_path, problem_name_list[problem_index])
    print("Execution completed")
