"""Example script for running multi-objective algorithms with reference fronts.

This script demonstrates how to use multi-objective optimization algorithms
with reference Pareto fronts for quality assessment. It loads reference fronts,
runs algorithms, and visualizes the obtained fronts against reference fronts.

Features:
- Load reference fronts from files
- Run multi-objective algorithms
- Interactive visualization of Pareto fronts
- Compare obtained fronts with reference fronts
"""
import sys
import os
import winsound
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.problem import FloatProblem
from evolu.lab.visualization import InteractivePlot, Plot
from evolu.problem.multi_objective.float_problems import *
from evolu.util.observer import VisualizerObserver
from evolu.util.solution import (print_function_values_to_file,
                                       print_variables_to_file, read_solutions)
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem) -> None:
    """Select and run a multi-objective algorithm with reference front comparison.
    
    This function creates a multi-objective algorithm instance, sets up a reference
    front for the problem (if available), configures visualization observers,
    runs the algorithm, and displays results.
    
    Args:
        algorithm_name (str): Name of the algorithm to run (e.g., "IBEA", "GDE3",
            "MOCell", "MOEAD", "NSGAII", "OMOPSO", "SPEA2", "SMPSO").
        problem (FloatProblem): The multi-objective optimization problem to solve.
            The problem should have a reference_front attribute set if available.
    
    Note:
        The algorithm is configured with a maximum of 1000 function evaluations
        and uses VisualizerObserver for real-time visualization of the Pareto front.
    """
    max_evaluations = 1000
    if algorithm_name == "IBEA":
        from evolu.optimizers.multiobjective.IBEA import IBEA
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = IBEA(
            problem=problem,
            kappa=1.0,
            population_size=100,
            offspring_population_size=100,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "GDE3":
        from evolu.optimizers.multiobjective.MODE import GDE3
        from evolu.operator.selection import DifferentialEvolutionSelection
        from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
        from evolu.util.comparator import GDominanceComparator
        algorithm = GDE3(
            problem=problem,
            population_size=100,
            selection=DifferentialEvolutionSelection(),
            crossover=FloatDifferentialEvolutionCrossover(CR=0.5, F=0.5, K=0.5),
            dominance_comparator=GDominanceComparator(problem.reference_point),
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
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "NSGAII":
        from evolu.optimizers.multiobjective import NSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import GDominanceComparator
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
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
            dominance_comparator=GDominanceComparator(problem.reference_point),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "NSGAIII":
        from evolu.optimizers.multiobjective.MOGAIII import NSGAIII, UniformReferenceDirectionFactory
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
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
    if algorithm_name == "MORS":
        from evolu.optimizers.multiobjective.MORS import MORS
        algorithm = MORS(
            problem=problem,
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
    if algorithm_name == "OMOPSO":
        from evolu.optimizers.multiobjective.MOPSO import OMOPSO
        from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        population_size = 100
        algorithm = OMOPSO(
            problem=problem,
            population_size=100,
            uniform_mutation=FloatUniformMutation(probability=1.0 / problem.number_of_variables, perturbation=0.5),
            non_uniform_mutation=FloatNonUniformMutation(
                probability=1.0 / problem.number_of_variables, perturbation=0.5,
                max_iterations=int(max_evaluations / population_size)),
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            epsilon=0.0075,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "SPEA2":
        from evolu.optimizers.multiobjective.SPEA2 import SPEA2
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import GDominanceComparator
        algorithm = SPEA2(
            problem=problem,
            population_size=40,
            offspring_population_size=40,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            dominance_comparator=GDominanceComparator(problem.reference_point),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=problem.reference_point)
    )
    algorithm.run()
    front = algorithm.get_result()
    # Plot front
    plot_front = Plot(
        title="Pareto front approximation", reference_front=problem.reference_front, axis_labels=problem.obj_labels
    )
    plot_front.plot(front, label=algorithm.label, file_name=algorithm.get_name())
    # Plot interactive front
    plot_front = InteractivePlot(
        title="Pareto front approximation", reference_front=problem.reference_front, axis_labels=problem.obj_labels
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
    problem.reference_point = [0.4, 0.6]
    problems.append(problem)
    problem = ZDT2()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/{}.pf".format(problem.get_name()))
    problem.reference_point = [0.2, 0.5]
    problems.append(problem)
    # NOTE: Additional test problems available but commented out.
    # Uncomment as needed: ZDT4, DTLZ2, Srinivas

    algorithm_name_list = ["MORS", "GDE3", "NSGAII", "SPEA2", ]
    num_of_runs = 1
    for algorithm_name in algorithm_name_list:
        for num_run in range(0, num_of_runs):
            for problem in problems:
                if problem.number_of_objectives <= 2 and algorithm_name == "NSGAIII":
                    continue
                select_and_run_algorithm(algorithm_name, problem)
    print("Execution completed")
