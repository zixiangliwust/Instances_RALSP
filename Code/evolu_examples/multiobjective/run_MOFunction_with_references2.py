"""Example script for running HYPE algorithm with reference points.

This script demonstrates how to use the HYPE (Hypervolume-based) algorithm
with reference points for multi-objective optimization.

Features:
- Uses HYPE algorithm with reference point
- Interactive visualization of Pareto fronts
- Saves results to files
"""
import sys
import os
import winsound
from typing import List
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.problem import FloatProblem
from evolu.core.solution import FloatSolution
from evolu.lab.visualization import InteractivePlot, Plot
from evolu.problem.multi_objective.float_problems import *
from evolu.util.observer import VisualizerObserver
from evolu.util.solution import (print_function_values_to_file,
                                       print_variables_to_file, read_solutions)
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem, reference_point: List[float]) -> None:
    """Select and run HYPE algorithm with reference point.
    
    This function creates a HYPE algorithm instance configured with a reference
    point, runs the algorithm, and displays/saves results.
    
    Args:
        algorithm_name (str): Name of the algorithm ("HYPE").
        problem (FloatProblem): The multi-objective optimization problem to solve.
        reference_point (List[float]): Reference point for hypervolume computation.
    
    Note:
        The algorithm is configured with 1000 maximum evaluations and uses
        visualization observers for real-time display.
    """
    max_evaluations = 1000
    if algorithm_name == "HYPE":
        from evolu.optimizers.multiobjective.HYPE import HYPE
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = HYPE(
            problem=problem,
            reference_point=reference_point,
            population_size=100,
            offspring_population_size=100,
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front)
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
    reference_points = []
    problem = ZDT1()
    problem.reference_front = read_solutions(
        file_name="D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front/ZDT1.pf")
    reference_point = FloatSolution([0], [1], problem.number_of_objectives, )
    reference_point.objectives = [1.0, 1.0]  # Mandatory for HYPE
    problems.append(problem)
    reference_points.append(reference_point)
    algorithm_name_list = ["HYPE"]
    num_of_runs = 1
    for algorithm_name in algorithm_name_list:
        for num_run in range(0, num_of_runs):
            for p in range(len(problems)):
                problem = problems[p]
                reference_point = reference_points[p]
                if problem.number_of_objectives <= 2 and algorithm_name == "NSGAIII":
                    continue
                select_and_run_algorithm(algorithm_name, problem, reference_point)
    print("Execution completed")
