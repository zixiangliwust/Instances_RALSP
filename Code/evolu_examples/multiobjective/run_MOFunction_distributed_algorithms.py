"""Example script for running distributed multi-objective optimization algorithms.

This script demonstrates how to use distributed computing capabilities of the
evolu framework with Dask or Spark for parallel evaluation of solutions.

Features:
- Distributed NSGA-II using Dask LocalCluster
- Dask evaluator for parallel solution evaluation
- Spark-based distributed algorithms (if available)

This is useful for computationally expensive problems where parallel evaluation
can significantly speed up optimization.
"""
import sys
import os
import winsound
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.problem import FloatProblem
from evolu.problem.multi_objective.float_problems import *
from evolu.util.solution import (print_function_values_to_file,
                                      print_variables_to_file)
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem) -> None:
    """Select and run a distributed multi-objective optimization algorithm.
    
    This function creates a distributed algorithm instance using Dask or Spark
    for parallel evaluation. It sets up a compute cluster and configures the
    algorithm to use distributed evaluation.
    
    Args:
        algorithm_name (str): Name of the distributed algorithm variant to run.
            Options include:
            - "distributed_nsgaii_with_dask": NSGA-II with Dask LocalCluster
            - "distributed_nsgaii_with_dask_evaluator": NSGA-II with DaskEvaluator
            - "distributed_nsgaii_with_spark": NSGA-II with Spark (if available)
        problem (FloatProblem): The multi-objective optimization problem to solve.
    
    Note:
        This example uses a small number of evaluations (100) for demonstration.
        Dask LocalCluster will be created with 24 workers. Adjust as needed
        for your system configuration.
    """
    max_evaluations = 100
    if algorithm_name == "distributed_nsgaii_with_dask":
        from dask.distributed import Client
        from distributed import LocalCluster
        from evolu.optimizers.multiobjective.MOGAII import DistributedNSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
        client = Client(LocalCluster(n_workers=24))
        ncores = sum(client.ncores().values())
        print(f"{ncores} cores available")
        # creates the algorithm
        algorithm = DistributedNSGAII(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            number_of_cores=ncores,
            client=client,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "distributed_nsgaii_with_dask_evaluator":
        from evolu.optimizers.multiobjective.MOGAII import NSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
        from evolu.util.evaluator import DaskEvaluator
        algorithm = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            population_evaluator=DaskEvaluator(),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "distributed_nsgaii_with_spark_evaluator":
        from evolu.optimizers.multiobjective.MOGAII import NSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.evaluator import SparkEvaluator
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
        algorithm = NSGAII(
            problem=problem,
            population_size=10,
            offspring_population_size=10,
            selection=BinaryTournamentSelection(
                MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                 SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
            ),
            crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
            population_evaluator=SparkEvaluator(),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "parallel_nsgaii_with_multiprocess_evaluator":
        from evolu.optimizers.multiobjective.MOGAII import NSGAII
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        from evolu.util.evaluator import MultiprocessEvaluator
        from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
        algorithm = NSGAII(
            population_evaluator=MultiprocessEvaluator(8),
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
    algorithm.run()
    front = algorithm.get_result()
    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)
    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    problems = []
    problem = ZDT1Modified()
    problems.append(problem)
    algorithm_name_list = [
        "distributed_nsgaii_with_dask",
        "distributed_nsgaii_with_dask_evaluator",
        "distributed_nsgaii_with_spark_evaluator",
        "parallel_nsgaii_with_multiprocess_evaluator"
    ]
    num_of_runs = 1
    for algorithm_name in algorithm_name_list:
        for num_run in range(0, num_of_runs):
            for problem in problems:
                if problem.number_of_objectives <= 2 and algorithm_name == "NSGAIII":
                    continue
                select_and_run_algorithm(algorithm_name, problem)
    print("Execution completed")
