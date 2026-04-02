"""Example script for running multi-objective optimization algorithms on MOReconfigurableALBP with float representation.

This script demonstrates how to use multi-objective optimization algorithms
to solve Multi-Objective Reconfigurable Assembly Line Balancing Problem
(MOReconfigurableALBP) using float solution representation.

Features:
- Runs multiple MOEAs on MOReconfigurableALBP problems
- Uses FloatPolynomialMutation and FloatSimulatedBinaryCrossover operators
- Progress bar observer for console output
- Real-time visualization of Pareto front evolution
- Static and interactive plots of final Pareto fronts
- Saves results to files

The float representation uses continuous variables that are decoded to determine
the product sequence through sorting, allowing gradient-based mutation operators.
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
    
    This function creates a FloatMOReconfigurableALBP problem instance, configures
    a multi-objective algorithm with float operators, runs it, and saves the
    Pareto front to files.
    
    Args:
        algorithm_name (str): Name of the algorithm to run. Supported algorithms:
            Local search: "MORS", "MSAA", "MRSA"
            Population-based: "MOABC", "IMOABC", "GDE3", "MOEAD", "NSGAII",
            "NSGAIII", "SPEA2", "OMOPSO", "SMPSO", "SMPSORP"
        file_path (str): Path to the directory containing the problem instance file.
        problem_name (str): Name of the problem instance file to load.
    
    Note:
        The algorithm is configured with:
        - Maximum of 1000 function evaluations
        - FloatPolynomialMutation for continuous variables
        - FloatSimulatedBinaryCrossover for algorithms that support it
        - Results (Pareto front) are saved to FUN.* and VAR.* files
        - Observers provide progress tracking and visualization
    """
    max_evaluations: int = 1000
    problem: FloatMOReconfigurableALSP = FloatMOReconfigurableALSP(file_path, problem_name)
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
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = MSAA(
            problem=problem,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MRSA":
        from evolu.optimizers.multiobjective.MOSA import MRSA
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = MRSA(
            problem=problem,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    """
    Population-based algorithms
    """
    if algorithm_name == "MOABC":
        from evolu.optimizers.multiobjective.MOABC import MOABC
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = MOABC(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "IMOABC":
        from evolu.optimizers.multiobjective.MOABC import IMOABC
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = IMOABC(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "GDE3":
        from evolu.optimizers.multiobjective.MODE import GDE3
        from evolu.operator.selection import DifferentialEvolutionSelection
        from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
        algorithm = GDE3(
            problem=problem,
            population_size=100,
            selection=DifferentialEvolutionSelection(),
            crossover=FloatDifferentialEvolutionCrossover(CR=0.5, F=0.5, K=0.5),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOEAD":
        from evolu.optimizers.multiobjective.MOEAD import MOEAD
        from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.aggregation_function import Tschebycheff
        algorithm = MOEAD(
            problem=problem,
            population_size=1000,
            crossover=FloatDifferentialEvolutionCrossover(CR=1.0, F=0.5),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            aggregation_function=Tschebycheff(dimension=problem.number_of_objectives),
            neighbor_size=20,
            neighborhood_selection_probability=0.9,
            max_number_of_replaced_solutions=2,
            weight_files_path="D:/GitHubCreated/IISO_EvoSuite/python/resources/MOEAD_weights",
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "NSGAII":
        from evolu.optimizers.multiobjective import NSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
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
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            dominance_comparator=DominanceWithConstraintsComparator(),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "OMOPSO":
        from evolu.optimizers.multiobjective.MOPSO import OMOPSO
        from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        population_size = 100
        mutation_probability = 1.0 / problem.number_of_variables
        algorithm = OMOPSO(
            problem=problem,
            population_size=population_size,
            uniform_mutation=FloatUniformMutation(probability=mutation_probability, perturbation=0.5),
            non_uniform_mutation=FloatNonUniformMutation(
                mutation_probability, perturbation=0.5, max_iterations=int(max_evaluations / population_size)
            ),
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            epsilon=0.0075,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "SMPSO":
        from evolu.optimizers.multiobjective.MOPSO import SMPSO
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = SMPSO(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "NSGAIII":
        from evolu.optimizers.multiobjective.MOGAIII import NSGAIII, UniformReferenceDirectionFactory
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = NSGAIII(
            problem=problem,
            population_size=92,
            reference_directions=UniformReferenceDirectionFactory(3, n_points=91),
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=30),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
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
    MOReconfigurableALBP problem instances with float representation. The script
    loads problem instances from a directory and runs algorithms, generating
    plots and saving results.
    
    Example:
        To run different algorithms or problems, modify the `algorithm_name_list`
        and `file_path` variables. Adjust num_of_runs for multiple independent
        runs for statistical analysis.
    """
    # Path to the directory containing problem instance files
    file_path: str = "D:/GitHubInstances/Instances_RALSP/Instances/"
    
    # Get all files from the directory
    entries: List[str] = os.listdir(file_path)
    problem_name_list: List[str] = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]
    # Uncomment to limit the number of problems: problem_name_list = problem_name_list[0:10]

    # Define the algorithms to test
    algorithm_name_list: List[str] = [
        "NSGAII",
        # "MORS", "MSAA", "MRSA",
        # "MOABC", "IMOABC", "GDE3", "MOEAD",
        # "NSGAII", "OMOPSO", "SMPSO", "NSGAIII",
    ]
    
    # Number of independent runs for each algorithm-problem combination
    num_of_runs: int = 1
    
    # Run all algorithm-problem combinations
    for algorithm_name in algorithm_name_list:
        for problem_index in range(0, len(problem_name_list)):
            for num_run in range(0, num_of_runs):
                select_and_run_algorithm(algorithm_name, file_path, problem_name_list[problem_index])
    
    print("Execution completed")
