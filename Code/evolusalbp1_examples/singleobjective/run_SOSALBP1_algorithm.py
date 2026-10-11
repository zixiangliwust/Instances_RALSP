"""Example script for running single-objective optimization algorithms on SALBP1/UALBP1 with both float and permutation representations.

This script demonstrates how to use various single-objective optimization
algorithms to solve Simple Assembly Line Balancing Problem Type 1 (SALBP1)
using float solution representation.

It includes examples of:
- Local search algorithms (Random Search, Local Search, Simulated Annealing, VNS)
- Population-based algorithms (ABC variants, DE, ES, GA, PSO, etc.)

The script runs multiple algorithms on SALBP1 problem instances and displays
the results, showing how to configure algorithms with FloatPolynomialMutation
operators for continuous solution representation in assembly line balancing scenarios.
"""
import sys
import os
import winsound
from typing import List, Union
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.algorithm import Algorithm
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.termination_criterion import StoppingByEvaluations
from evolusalbp1.problem.singleobjective.SALBP1_problems import *
from evolusalbp1.problem.singleobjective.UALBP1_problems import *

def select_and_run_algorithm(algorithm_name: str, file_path: str, problem_name: str, problem_type: str = "SALBP1") -> None:
    """Select and run a specified optimization algorithm on the given SALBP1/UALBP1 problem.
    
    This function creates a FloatSALBP1, PermutationSALBP1, FloatUALBP1, PermutationUALBP1, 
    or AntSALBP1/AntUALBP1 problem instance based on the algorithm type and problem_type parameter, 
    configures an algorithm with appropriate operators, runs the algorithm,
    and displays results.
    
    Args:
        algorithm_name (str): Name of the algorithm to run. 
        file_path (str): Path to the directory containing the problem instance file.
        problem_name (str): Name of the problem instance file to load.
        problem_type (str): Type of problem - "SALBP1" or "UALBP1" (default: "SALBP1")
    
    Note:
        The algorithm is configured with:
        - Maximum of 1000 function evaluations
        - FloatPolynomialMutation for continuous variables
        - Results are printed to console, including the best solution found
          and computing time.
        - UALBP1 allows bidirectional task assignment on U-shaped lines.
    """
    problem: FloatSALBP1 = FloatSALBP1(file_path, problem_name)
    max_evaluations: int = 1000
    
    # Determine problem type based on problem_type parameter
    if problem_type == "UALBP1":
        if algorithm_name.startswith("Float"):
            problem: FloatUALBP1 = FloatUALBP1(file_path, problem_name)
        elif algorithm_name in ["PermutationUALBP1PermutationACO", "PermutationUALBP1IntegerACO"]:
            problem = AntUALBP1(file_path, problem_name)
        else:
            problem = PermutationUALBP1(file_path, problem_name)
    else:  # SALBP1
        if algorithm_name.startswith("Float"):
            problem: FloatSALBP1 = FloatSALBP1(file_path, problem_name)
        elif algorithm_name in ["PermutationSALBP1PermutationACO", "PermutationSALBP1IntegerACO"]:
            problem = AntSALBP1(file_path, problem_name)
        else:
            problem = PermutationSALBP1(file_path, problem_name)
    """
    Local search algorithms
    """
    if algorithm_name == "FloatRSBase":
        from evolu.optimizers.singleobjective.RS import RSBase
        algorithm = RSBase(
            problem=problem,
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
    if algorithm_name == "FloatVNSBase":
        from evolu.optimizers.singleobjective.VNS import VNSBase
        from evolu.operator.mutation import (FloatPermutationIntegerSwapMutation,
                                                   FloatPermutationIntegerInsertionMutation, FloatPolynomialMutation)
        algorithm = VNSBase(
            problem=problem,
            mutation_operator_list=[
                FloatPermutationIntegerSwapMutation(probability=1.0 / problem.number_of_variables),
                FloatPermutationIntegerInsertionMutation(probability=1.0 / problem.number_of_variables),
                FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            ],
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatVNSSA":
        from evolu.optimizers.singleobjective.VNS import VNSSA
        from evolu.operator.mutation import (FloatPermutationIntegerSwapMutation,
                                                   FloatPermutationIntegerInsertionMutation, FloatPolynomialMutation)
        algorithm = VNSSA(
            problem=problem,
            mutation_operator_list=[
                FloatPermutationIntegerSwapMutation(probability=1.0 / problem.number_of_variables),
                FloatPermutationIntegerInsertionMutation(probability=1.0 / problem.number_of_variables),
                FloatPolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20.0),
            ],
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    """
    Population-based algorithms
    """
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
    if algorithm_name == "FloatABC1":
        from evolu.optimizers.singleobjective.ABC import ABC1
        from evolu.operator.selection import RouletteWheelSelection
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = ABC1(
            problem=problem,
            population_size=100,
            selection=RouletteWheelSelection(),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatABC2":
        from evolu.optimizers.singleobjective.ABC import ABC2
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = ABC2(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatIABC":
        from evolu.optimizers.singleobjective.ABC import IABC
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = IABC(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatBABase":
        from evolu.optimizers.singleobjective.BA import BABase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = BABase(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatIBA":
        from evolu.optimizers.singleobjective.BA import IBA
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = IBA(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatCSBase":
        from evolu.optimizers.singleobjective.CS import CSBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = CSBase(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatDCS":
        from evolu.optimizers.singleobjective.CS import DCS
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = DCS(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatICS":
        from evolu.optimizers.singleobjective.CS import ICS
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = ICS(
            problem=problem,
            population_size=100,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
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
            offspring_population_size=100,
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
        algorithm.algorithm_name = "FloatGASteady"
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
    if algorithm_name == "FloatIGA":
        from evolu.optimizers.singleobjective.GA import IGA
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = IGA(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
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
    if algorithm_name == "FloatMABase":
        from evolu.optimizers.singleobjective.MA import MABase
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import FloatSimulatedBinaryCrossover
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = MABase(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(),
            crossover=FloatSimulatedBinaryCrossover(0.9, 5.0),
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatMBOBase":
        from evolu.optimizers.singleobjective.MBO import MBOBase
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = MBOBase(
            problem=problem,
            population_size=121,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatIMBO":
        from evolu.optimizers.singleobjective.MBO import IMBO
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = IMBO(
            problem=problem,
            population_size=5,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatMBOSA":
        from evolu.optimizers.singleobjective.MBO import MBOSA
        from evolu.operator.mutation import FloatPolynomialMutation
        algorithm = MBOSA(
            problem=problem,
            population_size=5,
            mutation=FloatPolynomialMutation(1.0 / problem.number_of_variables, 20.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "FloatMFOBase":
        from evolu.optimizers.singleobjective.MFO import MFOBase
        algorithm = MFOBase(
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

    problem: Union[AntSALBP1, PermutationSALBP1]
    if algorithm_name == "SALBP1PermutationACO" or algorithm_name == "SALBP2ntergerACO":
        problem = AntSALBP1(file_path, problem_name)
    else:
        problem = PermutationSALBP1(file_path, problem_name)
    """
    Local search algorithms
    """
    if algorithm_name == "PermutationRSBase":
        from evolu.optimizers.singleobjective.RS import RSBase
        algorithm = RSBase(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSALBP1RandSearch":
        from evolusalbp1.optimizers.singleobjective.SALBP1RS import SALBP1RandSearch
        algorithm = SALBP1RandSearch(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSALBP1RandomPriority":
        from evolusalbp1.optimizers.singleobjective.SALBP1RS import SALBP1RandomPriority
        algorithm = SALBP1RandomPriority(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationLSBase":
        from evolu.optimizers.singleobjective.LS import LSBase
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = LSBase(
            problem=problem,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSABase":
        from evolu.optimizers.singleobjective.SA import SABase
        from evolu.operator.mutation import FloatPermutationIntegerScrambleMutation
        algorithm = SABase(
            problem=problem,
            mutation=FloatPermutationIntegerScrambleMutation(probability=1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationVNSBase":
        from evolu.optimizers.singleobjective.VNS import VNSBase
        from evolu.operator.mutation import (FloatPermutationIntegerSwapMutation,
                                                   FloatPermutationIntegerInsertionMutation,
                                                   FloatPermutationIntegerInversionMutation)
        algorithm = VNSBase(
            problem=problem,
            mutation_operator_list=[
                FloatPermutationIntegerSwapMutation(probability=1.0 / problem.number_of_variables),
                FloatPermutationIntegerInsertionMutation(probability=1.0 / problem.number_of_variables),
                FloatPermutationIntegerInversionMutation(probability=1.0 / problem.number_of_variables),
            ],
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationVNSSA":
        from evolu.optimizers.singleobjective.VNS import VNSSA
        from evolu.operator.mutation import (FloatPermutationIntegerSwapMutation,
                                                   FloatPermutationIntegerInsertionMutation,
                                                   FloatPermutationIntegerInversionMutation)
        algorithm = VNSSA(
            problem=problem,
            mutation_operator_list=[
                FloatPermutationIntegerSwapMutation(probability=1.0 / problem.number_of_variables),
                FloatPermutationIntegerInsertionMutation(probability=1.0 / problem.number_of_variables),
                FloatPermutationIntegerInversionMutation(probability=1.0 / problem.number_of_variables),
            ],
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationIG":
        from evolu.optimizers.singleobjective.PermutationILS import PermutationIG
        from evolu.operator.mutation import FloatPermutationIntegerScrambleMutation
        algorithm = PermutationIG(
            problem=problem,
            mutation=FloatPermutationIntegerScrambleMutation(probability=1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationILS":
        from evolu.optimizers.singleobjective.PermutationILS import PermutationILS
        from evolu.operator.mutation import FloatPermutationIntegerScrambleMutation
        algorithm = PermutationILS(
            problem=problem,
            mutation=FloatPermutationIntegerScrambleMutation(probability=1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSALBP2ntergerSA":
        from evolusalbp1.optimizers.singleobjective.SALBP1SA import SALBP2ntergerSA
        algorithm = SALBP2ntergerSA(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationTS":
        from evolu.optimizers.singleobjective.PermutationTS import PermutationTS
        algorithm = PermutationTS(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSALBP2ntergerTS":
        from evolusalbp1.optimizers.singleobjective.SALBP1TS import SALBP2ntergerTS
        algorithm = SALBP2ntergerTS(
            problem=problem,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    """ 
    Population-based algorithms
    """
    if algorithm_name == "PermutationABCBase":
        from evolu.optimizers.singleobjective.ABC import ABCBase
        from evolu.operator.selection import RouletteWheelSelection
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = ABCBase(
            problem=problem,
            population_size=100,
            selection=RouletteWheelSelection(),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationABC1":
        from evolu.optimizers.singleobjective.ABC import ABC1
        from evolu.operator.selection import RouletteWheelSelection
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = ABC1(
            problem=problem,
            population_size=100,
            selection=RouletteWheelSelection(),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationABC2":
        from evolu.optimizers.singleobjective.ABC import ABC2
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = ABC2(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationIABC":
        from evolu.optimizers.singleobjective.ABC import IABC
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = IABC(
            problem=problem,
            population_size=100,
            selection=BinaryTournamentSelection(),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationBABase":
        from evolu.optimizers.singleobjective.BA import BABase
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = BABase(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationIBA":
        from evolu.optimizers.singleobjective.BA import IBA
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = IBA(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationCSBase":
        from evolu.optimizers.singleobjective.CS import CSBase
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = CSBase(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationDCS":
        from evolu.optimizers.singleobjective.CS import DCS
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = DCS(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationICS":
        from evolu.optimizers.singleobjective.CS import ICS
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = ICS(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationESBase":
        from evolu.optimizers.singleobjective.ES import ESBase
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = ESBase(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            elitist=True,
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationFOABase":
        from evolu.optimizers.singleobjective.FOA import FOABase
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = FOABase(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationIFOA":
        from evolu.optimizers.singleobjective.FOA import IFOA
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = IFOA(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationGABase":
        from evolu.optimizers.singleobjective.GA import GABase
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import PermutationPartiallyMatchedCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = GABase(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(),
            crossover=PermutationTwoPointCrossover(0.8),
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationGASteady":
        from evolu.optimizers.singleobjective.GA import GABase
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import PermutationPartiallyMatchedCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = GABase(
            problem=problem,
            population_size=100,
            offspring_population_size=1,
            selection=BinaryTournamentSelection(),
            crossover=PermutationPartiallyMatchedCrossover(0.8),
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
        algorithm.algorithm_name = "PermutationGASteady"
    if algorithm_name == "PermutationGAElite":
        from evolu.optimizers.singleobjective.GA import GAElite
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import PermutationPartiallyMatchedCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = GAElite(
            problem=problem,
            population_size=100,
            offspring_population_size=int(100 * 0.9),
            selection=BinaryTournamentSelection(),
            crossover=PermutationPartiallyMatchedCrossover(0.8),
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationIGA":
        from evolu.optimizers.singleobjective.GA import IGA
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import PermutationPartiallyMatchedCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = IGA(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(),
            crossover=PermutationPartiallyMatchedCrossover(0.8),
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationMABase":
        from evolu.optimizers.singleobjective.MA import MABase
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import PermutationPartiallyMatchedCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = MABase(
            problem=problem,
            population_size=100,
            offspring_population_size=100,
            selection=BinaryTournamentSelection(),
            crossover=PermutationPartiallyMatchedCrossover(0.8),
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationMBOBase":
        from evolu.optimizers.singleobjective.MBO import MBOBase
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = MBOBase(
            problem=problem,
            population_size=121,
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationIMBO":
        from evolu.optimizers.singleobjective.MBO import IMBO
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = IMBO(
            problem=problem,
            population_size=5,
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationMBOSA":
        from evolu.optimizers.singleobjective.MBO import MBOSA
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = MBOSA(
            problem=problem,
            population_size=5,
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationDiscretePSO":
        from evolu.optimizers.singleobjective.PSO import DiscretePSO
        from evolu.operator.selection import BinaryTournamentSelection
        from evolu.operator.crossover import PermutationPartiallyMatchedCrossover
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = DiscretePSO(
            problem=problem,
            population_size=100,
            crossover=PermutationPartiallyMatchedCrossover(0.8),
            mutation=FloatPermutationIntegerSwapMutation(1.0 / problem.number_of_variables),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSAPopulation":
        from evolu.optimizers.singleobjective.SA import SAPopulation
        from evolu.operator.mutation import FloatPermutationIntegerSwapMutation
        algorithm = SAPopulation(
            problem=problem,
            population_size=100,
            mutation=FloatPermutationIntegerSwapMutation(probability=1.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSALBP1PermutationACO":
        from evolusalbp1.optimizers.singleobjective.SALBP1ACO import SALBP1PermutationACO
        algorithm = SALBP1PermutationACO(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationSALBP2ntergerACO":
        from evolusalbp1.optimizers.singleobjective.SALBP1ACO import SALBP2ntergerACO
        algorithm = SALBP2ntergerACO(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    
    # UALBP1-specific ACO algorithms
    if algorithm_name == "PermutationUALBP1PermutationACO":
        from evolusalbp1.optimizers.singleobjective.SALBP1ACO import SALBP1PermutationACO
        algorithm = SALBP1PermutationACO(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )
    if algorithm_name == "PermutationUALBP1IntegerACO":
        from evolusalbp1.optimizers.singleobjective.SALBP1ACO import SALBP2ntergerACO
        algorithm = SALBP2ntergerACO(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        )

    algorithm.run()
    result: Union[Solution, List[Solution]] = algorithm.get_result()
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
    file_path: str = "D:/GitHubInstances/Instances_SALBP1/Instances/"
    entries: List[str] = os.listdir(file_path)
    problem_name_list: List[str] = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]

    algorithm_name_list: List[str] = [
        # Float algorithms
        "FloatRSBase",
        "FloatLSBase",
        "FloatSABase",
        "FloatVNSBase",
        "FloatVNSSA",
        "FloatABCBase",
        "FloatABC1",
        "FloatABC2",
        "FloatIABC",
        "FloatBABase",
        "FloatIBA",
        "FloatCSBase",
        "FloatDCS",
        "FloatICS",
        "FloatDEBase",
        "FloatESBase",
        "FloatGABase",
        "FloatGASteady",
        "FloatGAElite",
        "FloatIGA",
        "FloatGWOBase",
        "FloatGWORW",
        "FloatJAYABase",
        "FloatMABase",
        "FloatMBOBase",
        "FloatIMBO",
        "FloatMBOSA",
        "FloatMFOBase",
        "FloatPSOBase",
        "FloatPSOPhasor",
        "FloatSAPopulation",
        "FloatTLBOBase",
        "FloatWOABase",
        # Permutation algorithms - SALBP1
        "PermutationRSBase",
        "PermutationSALBP1RandSearch",
        "PermutationSALBP1RandomPriority",
        "PermutationLSBase",
        "PermutationSABase",
        "PermutationVNSBase",
        "PermutationVNSSA",
        "PermutationIG",
        "PermutationILS",
        "PermutationSALBP2ntergerSA",
        "PermutationTS",
        "PermutationSALBP2ntergerTS",
        "PermutationABCBase",
        "PermutationABC1",
        "PermutationABC2",
        "PermutationIABC",
        "PermutationBABase",
        "PermutationIBA",
        "PermutationCSBase",
        "PermutationDCS",
        "PermutationICS",
        "PermutationESBase",
        "PermutationGABase",
        "PermutationGASteady",
        "PermutationGAElite",
        "PermutationIGA",
        "PermutationMABase",
        "PermutationMBOBase",
        "PermutationIMBO",
        "PermutationMBOSA",
        "PermutationDiscretePSO",
        "PermutationSAPopulation",
        "PermutationSALBP1PermutationACO",
        "PermutationSALBP2ntergerACO",
        # UALBP1 ACO algorithms
        "PermutationUALBP1PermutationACO",
        "PermutationUALBP1IntegerACO",
    ]
    num_of_runs: int = 1
    for algorithm_name in algorithm_name_list:
        for problem_index in range(0, len(problem_name_list)):
            for num_run in range(0, num_of_runs):
                select_and_run_algorithm(algorithm_name, file_path, problem_name_list[problem_index])
    print("Execution completed")