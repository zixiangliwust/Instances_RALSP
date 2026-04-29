"""Example script for running single-objective optimization algorithms.

This script demonstrates how to use various single-objective optimization
algorithms from the evolu framework to solve continuous optimization problems.

It includes examples of:
- Local search algorithms (Random Search, Hill Climbing, Simulated Annealing)
- Population-based algorithms (Genetic Algorithm, Particle Swarm Optimization,
  Differential Evolution, etc.)

The script runs multiple algorithms on benchmark problems and displays the results.
"""
import sys
import os
import winsound
from typing import List, Union
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.algorithm import Algorithm
from evolu.core.problem import FloatProblem
from evolu.core.solution import Solution
from evolu.problem.single_objective.float_problems.float_problems import *
from evolu.util.termination_criterion import StoppingByEvaluations

def select_and_run_algorithm(algorithm_name: str, problem: FloatProblem) -> None:
    """Select and run a specified optimization algorithm on the given problem.
    
    This function creates an algorithm instance based on the algorithm name,
    configures it with appropriate operators and parameters, runs the algorithm,
    and displays the results.
    
    Args:
        algorithm_name (str): Name of the algorithm to run. 
    
    Note:
        The algorithm is configured with a maximum of 10000 function evaluations.
        Results are printed to console, including the best solution found and
        computing time.
    """
    max_evaluations = 10000
    
    # Local search algorithms
    if algorithm_name == "FloatRSBase":
        from evolu.optimizers.singleobjective.RS import RSBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = RSBase(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatHCBase":
        from evolu.optimizers.singleobjective.HC import HCBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = HCBase(
            problem=problem,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatLAHCBase":
        from evolu.optimizers.singleobjective.HC import LAHCBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = LAHCBase(
            problem=problem,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatLSBase":
        from evolu.optimizers.singleobjective.LS import LSBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = LSBase(
            problem=problem,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatSABase":
        from evolu.optimizers.singleobjective.SA import SABase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = SABase(
            problem=problem,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatRSA":
        from evolu.optimizers.singleobjective.SA import RSA
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = RSA(
            problem=problem,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    # Population-based algorithms
    if algorithm_name == "FloatABCBase":
        from evolu.optimizers.singleobjective.ABC import ABCBase
        from evolu.operator.selection import RouletteWheelSelection
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = ABCBase(
            problem=problem,
            population_size=100,
            selection=RouletteWheelSelection(),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatAOABase":
        from evolu.optimizers.singleobjective.AOA import AOABase
        algorithm = AOABase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatDEBase":
        from evolu.optimizers.singleobjective.DE import DEBase
        from evolu.operator.selection import DifferentialEvolutionSelection
        from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
        algorithm = DEBase(
            problem=problem,
            population_size=100,
            selection=DifferentialEvolutionSelection(),
            crossover=FloatDifferentialEvolutionCrossover(),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatESBase":
        from evolu.optimizers.singleobjective.ES import ESBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = ESBase(
            problem=problem,
            population_size=100,
            offspring_population_size=10,
            elitist=True,
            mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatGABase":
        from evolu.optimizers.singleobjective.GA import GABase
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = GABase(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(),
            crossover=FloatSimulatedBinaryCrossover(0.9, 5.0),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatGASteady":
        from evolu.optimizers.singleobjective.GA import GABase
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = GABase(
            problem=problem,
            population_size=100,
            offspring_population_size=1,
            selection=BinaryTournamentSelection(),
            crossover=FloatSimulatedBinaryCrossover(0.9, 5.0),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
        algorithm.algorithm_name = "GASteady"
    if algorithm_name == "FloatGAElite":
        from evolu.optimizers.singleobjective.GA import GAElite
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = GAElite(
            problem=problem,
            population_size=100,
            offspring_population_size=int(100 * 0.9),
            selection=BinaryTournamentSelection(),
            crossover=FloatSimulatedBinaryCrossover(0.9, 5.0),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatGWOBase":
        from evolu.optimizers.singleobjective.GWO import GWOBase
        algorithm = GWOBase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatGWORW":
        from evolu.optimizers.singleobjective.GWO import GWORW
        algorithm = GWORW(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatJAYABase":
        from evolu.optimizers.singleobjective.JAYA import JAYABase
        algorithm = JAYABase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatMFOBase":
        from evolu.optimizers.singleobjective.MFO import MFOBase
        algorithm = MFOBase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatNMRABase":
        from evolu.optimizers.singleobjective.NMRA import NMRABase
        algorithm = NMRABase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatPSOBase":
        from evolu.optimizers.singleobjective.PSO import PSOBase
        algorithm = PSOBase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatPSO2007":
        from evolu.optimizers.singleobjective.PSO import PSO2007
        algorithm = PSO2007(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatPSO2011":
        from evolu.optimizers.singleobjective.PSO import PSO2011
        algorithm = PSO2011(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatPSOPhasor":
        from evolu.optimizers.singleobjective.PSO import PSOPhasor
        algorithm = PSOPhasor(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatSAPopulation":
        from evolu.optimizers.singleobjective.SA import SAPopulation
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = SAPopulation(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatSCABase":
        from evolu.optimizers.singleobjective.SCA import SCABase
        algorithm = SCABase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatSSOBase":
        from evolu.optimizers.singleobjective.SSO import SSOBase
        algorithm = SSOBase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatTLBOBase":
        from evolu.optimizers.singleobjective.TLBO import TLBOBase
        algorithm = TLBOBase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatWOABase":
        from evolu.optimizers.singleobjective.WOA import WOABase
        algorithm = WOABase(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatWOAHI":
        from evolu.optimizers.singleobjective.WOA import WOAHI
        algorithm = WOAHI(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    algorithm.run()
    result = algorithm.get_result()
    # Save results to file
    # print_function_values_to_file(result, "results/FUN." + algorithm.get_name() + "." + problem.get_name())
    # print_variables_to_file(result, "results/VAR." + algorithm.get_name() + "." + problem.get_name())
    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.get_name()}")
    if isinstance(result, list):
        print(f"Solution: {result[0].variables}")
        print(f"Fitness: {result[0].objectives}")
    else:
        print(f"Solution: {result.variables}")
        print(f"Fitness: {result.objectives}")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    """Main execution block.
    
    This block demonstrates how to run multiple algorithms on multiple problems
    with multiple independent runs for statistical analysis.
    
    Example:
        To run different algorithms, modify the `algorithm_name_list` and `problems`
        lists. The script will execute each algorithm on each problem for the
        specified number of runs.
    
    Note:
        The script will beep when execution completes (Windows only).
    """
    # Define the problems to solve
    problems = [
        Sphere(),
        # Rastrigin(10),
    ]
    
    # Define the algorithms to test
    algorithm_name_list = [
        "FloatGABase",
        "FloatPSOBase",
        # "FloatRSBase",
        # "FloatHCBase",
        # "FloatLAHCBase",
        # "FloatLSBase",
        # "FloatSABase",
        # "FloatRSA",
        # "FloatABCBase",
        # "FloatAOABase",
        # "FloatDEBase",
        # "FloatESBase",
        # "FloatGABase",
        # "FloatGASteady",
        # "FloatGAElite",
        # "FloatGWOBase",
        # "FloatGWORW",
        # "FloatJAYABase",
        # "FloatMFOBase",
        # "FloatNMRABase",
        # "FloatPSOBase",
        # "FloatPSO2007",
        # "FloatPSO2011",
        # "FloatPSOPhasor",
        # "FloatSAPopulation",
        # "FloatSCABase",
        # "FloatSSOBase",
        # "FloatTLBOBase",
        # "FloatWOABase",
        # "FloatWOAHI",
    ]
    
    # Number of independent runs for each algorithm-problem combination
    num_of_runs = 10
    
    # Run all algorithm-problem combinations
    for algorithm_name in algorithm_name_list:
        for problem in problems:
            for num_run in range(0, num_of_runs):
                select_and_run_algorithm(algorithm_name, problem)
    
    print("Execution completed")
    # Play beep sound when execution completes (Windows only)
    try:
        winsound.Beep(440, 10000)  # Frequency 440Hz, duration 10000ms
    except Exception:
        pass  # Ignore if winsound is not available on this platform