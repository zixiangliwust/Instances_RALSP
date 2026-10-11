"""Example script for running dynamic multi-objective optimization algorithms.

This script demonstrates how to use dynamic optimization algorithms that can
adapt to time-varying problems where the problem definition changes over time.

Features:
- Dynamic GDE3 algorithm for dynamic MOO
- Dynamic NSGA-II algorithm
- TimeCounter for triggering problem changes
- Observers for saving fronts at different time steps

This is useful for problems where the Pareto front changes over time,
requiring algorithms that can track and adapt to these changes.
"""
import sys
import os
import winsound
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.problem import FloatProblem
from evolu.problem.multi_objective.float_problems import *
from evolu.util.observable import TimeCounter
from evolu.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem) -> None:
    """Select and run a dynamic multi-objective optimization algorithm.
    
    This function creates a dynamic algorithm instance that can adapt to
    problem changes over time. It sets up a TimeCounter to trigger problem
    changes and observers to save fronts at different time steps.
    
    Args:
        algorithm_name (str): Name of the dynamic algorithm to run.
            Options include:
            - "DynamicGDE3": Dynamic version of GDE3
            - "DynamicNSGAII": Dynamic version of NSGA-II
        problem (FloatProblem): The dynamic multi-objective optimization
            problem to solve. The problem should implement change detection.
    
    Note:
        The TimeCounter is configured to trigger problem changes every second.
        Observers save the current front to files at each change event.
    """
    max_evaluations = 500
    if algorithm_name == "DynamicGDE3":
        from evolu.optimizers.multiobjective.MODE import DynamicGDE3
        from evolu.operator.selection import DifferentialEvolutionSelection
        from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
        time_counter = TimeCounter(delay=1)
        time_counter.observable.register(problem)
        time_counter.start()
        algorithm = DynamicGDE3(
            problem=problem,
            population_size=100,
            selection=DifferentialEvolutionSelection(),
            crossover=FloatDifferentialEvolutionCrossover(CR=0.5, F=0.5, K=0.5),
            termination_criterion=StoppingByEvaluations(max_evaluations=500),
        )
    if algorithm_name == "DynamicNSGAII":
        from evolu.optimizers.multiobjective.MOGAII import DynamicNSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
        time_counter = TimeCounter(delay=1)
        time_counter.observable.register(problem)
        time_counter.start()
        max_evaluations = 25000
        algorithm = DynamicNSGAII(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "DynamicSMPSO":
        from evolu.optimizers.multiobjective.MOPSO import DynamicSMPSO
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.archive import BoundedCrowdingDistanceArchive
        time_counter = TimeCounter(delay=1)
        time_counter.observable.register(problem)
        time_counter.start()
        algorithm = DynamicSMPSO(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            leaders_archive=BoundedCrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    algorithm.observable.register(observer=PlotFrontToFileObserver("dynamic_front_vis"))
    algorithm.observable.register(observer=WriteFrontToFileObserver("dynamic_front"))
    algorithm.run()
    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    problems = []
    problem = FDA2()
    problems.append(problem)
    algorithm_name_list = ["DynamicGDE3", "DynamicNSGAII", "DynamicSMPSO"]
    algorithm_name_list = ["DynamicGDE3", ]
    num_of_runs = 1
    for algorithm_name in algorithm_name_list:
        for num_run in range(0, num_of_runs):
            for problem in problems:
                if problem.number_of_objectives <= 2 and algorithm_name == "NSGAIII":
                    continue
                select_and_run_algorithm(algorithm_name, problem)
    print("Execution completed")
