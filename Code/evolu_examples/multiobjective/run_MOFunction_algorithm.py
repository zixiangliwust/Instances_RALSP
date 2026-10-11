"""Example script for running multi-objective optimization algorithms with visualization.

This script demonstrates how to use multi-objective optimization algorithms
with quality indicators, observers, and visualization capabilities.

Features:
- Runs multi-objective algorithms on benchmark problems
- Computes quality indicators (Hypervolume, Epsilon)
- Uses observers for progress tracking and visualization
- Saves results to files
- Interactive visualization of Pareto fronts

This is a more advanced example showing the full capabilities of the framework
for multi-objective optimization analysis.
"""
import sys
import os
import winsound
from typing import List
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.algorithm import Algorithm
from evolu.core.problem import FloatProblem
from evolu.core.solution import Solution
from evolu.core.quality_indicator import HyperVolume, EpsilonIndicator
from evolu.lab.visualization import InteractivePlot, Plot
from evolu.problem.multi_objective.float_problems import *
from evolu.util.observer import ProgressBarObserver, VisualizerObserver
from evolu.util.solution import (get_non_dominated_solutions, print_function_values_to_file,
                                       print_variables_to_file, read_solutions)
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem) -> None:
    """Select and run a multi-objective optimization algorithm with visualization.
    
    This function creates a multi-objective algorithm instance, configures observers
    for progress tracking and visualization, runs the algorithm, computes quality
    indicators, and saves results.
    
    Args:
        algorithm_name (str): Name of the algorithm to run. See available algorithms
            in the function body for supported algorithm names.
        problem (FloatProblem): The multi-objective optimization problem to solve.
    
    Note:
        The algorithm is configured with:
        - Maximum of 10000 function evaluations
        - Progress bar observer for console output
        - Visualizer observer for real-time visualization
        - Quality indicator computation (Hypervolume, Epsilon)
    """
    max_evaluations = 10000
    
    # Local search algorithms
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
    if algorithm_name == "MOAOA":
        from evolu.optimizers.multiobjective.MOAOA import MOAOA
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOAOA(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
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
            population_size=300,
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
    if algorithm_name == "MOGWO":
        from evolu.optimizers.multiobjective.MOGWO import MOGWO
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOGWO(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOJAYA":
        from evolu.optimizers.multiobjective.MOJAYA import MOJAYA
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOJAYA(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOMBO":
        from evolu.optimizers.multiobjective.MOMBO import MOMBO
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import DominanceWithConstraintsComparator, MultiComparator, \
            SolutionAttributeComparator
        algorithm = MOMBO(
            problem=problem,
            population_size=20,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOMFO":
        from evolu.optimizers.multiobjective.MOMFO import MOMFO
        algorithm = MOMFO(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MONMRA":
        from evolu.optimizers.multiobjective.MONMRA import MONMRA
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MONMRA(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
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
    if algorithm_name == "MOSCA":
        from evolu.optimizers.multiobjective.MOSCA import MOSCA
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOSCA(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOSSO":
        from evolu.optimizers.multiobjective.MOSSO import MOSSO
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOSSO(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOTLBO":
        from evolu.optimizers.multiobjective.MOTLBO import MOTLBO
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOTLBO(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOWOA":
        from evolu.optimizers.multiobjective.MOWOA import MOWOA
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        algorithm = MOWOA(
            problem=problem,
            population_size=100,
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOCell":
        from evolu.optimizers.multiobjective.MOCell import MOCell
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        from evolu.util.neighborhood import C9
        algorithm = MOCell(
            problem=problem,
            population_size=100,
            neighborhood=C9(10, 10),
            archive=BoundedCrowdingDistanceArchive(100),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "MOEAD_DRA":
        from evolu.optimizers.multiobjective.MOEAD import MOEAD_DRA
        from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.aggregation_function import Tschebycheff
        algorithm = MOEAD_DRA(
            problem=problem,
            population_size=600,
            crossover=FloatDifferentialEvolutionCrossover(CR=1.0, F=0.5),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            aggregation_function=Tschebycheff(dimension=problem.number_of_objectives),
            neighbor_size=20,
            neighborhood_selection_probability=0.9,
            max_number_of_replaced_solutions=2,
            weight_files_path="D:/GitHubCreated/IISO_EvoSuite/python/resources/MOEAD_weights",
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "IBEA":
        from evolu.optimizers.multiobjective.IBEA import IBEA
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = IBEA(
            problem=problem,
            kappa=1.0,
            population_size=100,
            offspring_population_size=100,
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
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
    if algorithm_name == "SPEA2":
        from evolu.optimizers.multiobjective.SPEA2 import SPEA2
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = SPEA2(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=100)
    )
    algorithm.run()
    front = algorithm.get_result()
    front = get_non_dominated_solutions(front)
    hypervolume = HyperVolume([1, 1])
    value = hypervolume.compute([front[i].objectives for i in range(len(front))])
    print(f"Value of HyperVolume is: {value}")
    ei = EpsilonIndicator(
        reference_front=[problem.reference_front[i].objectives for i in range(len(problem.reference_front))])
    value = ei.compute([front[i].objectives for i in range(len(front))])
    print(f"Value of EpsilonIndicator is: {value}")
    # Plot front
    plot_front = Plot(
        title="Pareto front approximation. Problem: " + problem.get_name(),
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(front, label=algorithm.label, file_name=algorithm.get_name())
    # Plot interactive front
    plot_front = InteractivePlot(
        title="Pareto front approximation. Problem: " + problem.get_name(),
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(front, label=algorithm.label, file_name=algorithm.get_name())
    # Save results to file
    print_function_values_to_file(front, "results/FUN." + algorithm.label)
    print_variables_to_file(front, "results/VAR." + algorithm.label)
    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    problems = []
    problem = ZDT1()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/ZDT1.pf")
    problems.append(problem)
    problem = ZDT2()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/ZDT2.pf")
    problems.append(problem)
    problem = ZDT3()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/ZDT3.pf")
    problems.append(problem)
    problem = ZDT4()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/ZDT4.pf")
    problems.append(problem)
    problem = ZDT6()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/ZDT6.pf")
    problems.append(problem)

    # NOTE: Additional test problems are available but commented out.
    # Uncomment as needed: DTLZ2, Srinivas, LZ09_F2, LIRCMOP2, UF1, Schaffer
    # See other example files for usage patterns.

    algorithm_name_list = [
        "OMOPSO",
        # "MORS",
        # "MSAA",
        # "MRSA",
        # "MOABC",
        # "IMOABC",
        # "MOAOA",
        # "GDE3",
        # "MOEAD",
        # "NSGAII",
        # "MOGWO",
        # "MOJAYA",
        # "MOMBO",
        # "MOMFO",
        # "MONMRA",
        # "SMPSO",
        # "MOSCA",
        # "MOSSO",
        # "MOTLBO",
        # "MOWOA",
        # "MOEAD_DRA",
        # "MOCell",
        # "IBEA",
        # "NSGAIII",
        # "SPEA2",
    ]
    num_of_runs = 1
    for problem in problems:
        for num_run in range(0, num_of_runs):
            for algorithm_name in algorithm_name_list:
                if problem.number_of_objectives <= 2 and algorithm_name == "NSGAIII":
                    continue
                select_and_run_algorithm(algorithm_name, problem)
    print("Execution completed")
    # Play beep sound when execution completes (Windows only)
    try:
        winsound.Beep(440, 10000)  # Frequency 440Hz, duration 10000ms
        print("End of beep.")
    except Exception:
        pass  # Ignore if winsound is not available on this platform
