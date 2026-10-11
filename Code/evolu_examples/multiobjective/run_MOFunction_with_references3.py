"""Example script for running SMPSORP algorithm with multiple reference points.

This script demonstrates how to use the SMPSORP (SMPSO with Reference Points)
algorithm with multiple reference points for multi-objective optimization.

Features:
- Uses SMPSORP algorithm with multiple reference points
- Creates separate archives for each reference point
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
from evolu.lab.visualization import InteractivePlot, Plot
from evolu.problem.multi_objective.float_problems import *
from evolu.util.observer import VisualizerObserver
from evolu.util.solution import (print_function_values_to_file,
                                       print_variables_to_file, read_solutions)
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem, reference_point: List[float]) -> None:
    """Select and run SMPSORP algorithm with multiple reference points.
    
    This function creates a SMPSORP algorithm instance configured with multiple
    reference points, where each reference point has its own archive. This allows
    the algorithm to focus on different regions of the Pareto front.
    
    Args:
        algorithm_name (str): Name of the algorithm ("SMPSORP").
        problem (FloatProblem): The multi-objective optimization problem to solve.
        reference_point (List[float]): List of reference points. The algorithm
            creates separate archives for each reference point.
    
    Note:
        The population is divided equally among the reference points, with each
        archive receiving population_size / len(reference_point) solutions.
    """
    max_evaluations = 1000
    if algorithm_name == "SMPSORP":
        from evolu.optimizers.multiobjective.MOPSO import SMPSORP
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.archive import CrowdingDistanceArchiveWithReferencePoint
        population_size = 100
        archives_with_reference_points = []
        for point in reference_point:
            archives_with_reference_points.append(
                CrowdingDistanceArchiveWithReferencePoint(int(population_size / len(reference_point)), point)
            )
        algorithm = SMPSORP(
            problem=problem,
            population_size=population_size,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            reference_points=reference_point,
            leaders_archive=archives_with_reference_points,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, reference_point=reference_point)
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
    reference_point = [[0.1, 0.8], [0.6, 0.1]]
    problems.append(problem)
    reference_points.append(reference_point)
    algorithm_name_list = ["SMPSORP"]
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
    # Play beep sound when execution completes (Windows only)
    try:
        winsound.Beep(440, 10000)  # Frequency 440Hz, duration 10000ms
        print("End of beep.")
    except Exception:
        pass  # Ignore if winsound is not available on this platform
