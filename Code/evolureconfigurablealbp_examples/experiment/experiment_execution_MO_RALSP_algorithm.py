"""Example script for multi-objective experimental execution on MOReconfigurableALBP with float representation.

This script demonstrates how to set up and run systematic experimental studies
comparing multiple multi-objective optimization algorithms on reconfigurable
assembly line balancing problems with float solution representation.

Features:
- Configures multiple MOEAs with standardized settings
- Uses FloatPolynomialMutation and FloatSimulatedBinaryCrossover operators
- Runs experiments with multiple independent runs
- Generates quality indicator summaries
- Creates statistical reports for algorithm comparison

This is used for comprehensive algorithm evaluation and comparison studies on
MOReconfigurableALBP problems.
"""
import sys
import os
import winsound
import time
from typing import Dict, List
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolu.core.algorithm import Algorithm
from evolu.core.problem import Problem
from evolu.lab.experiment import ExperimentExecution, ExperimentReport
from evolu.util.termination_criterion import StoppingByEvaluations
from evolureconfigurablealbp.problem.multiobjective.MO_ReconfigurableALS_problem import *

class ExperimentExecutionMO(ExperimentExecution):
    """Multi-objective experiment execution configuration for MOReconfigurableALBP with float representation.
    
    This class extends ExperimentExecution to configure multi-objective
    optimization algorithms for experimental studies on reconfigurable assembly
    line balancing problems with float solution representation.
    
    Attributes:
        m_workers (int): Number of parallel workers for experiment execution.
            Defaults to 6.
    """
    
    def __init__(self) -> None:
        """Initialize experiment execution for multi-objective problems.
        
        Sets up the experiment execution with 6 parallel workers for
        concurrent algorithm runs.
        """
        super(ExperimentExecutionMO, self).__init__(m_workers=28)

    def algorithm_settings(self, problem: Problem, algorithm_name: str = "") -> Algorithm:
        """Configure algorithm settings based on algorithm name.
        
        Creates and configures an algorithm instance with standardized settings
        for experimental comparison. Supports various multi-objective algorithms
        with float operators for continuous solution representation.
        
        Args:
            problem (Problem): The optimization problem to solve.
            algorithm_name (str, optional): Name of the algorithm to configure.
                Supported algorithms: "MSAA", "MRSA", "MOABC", "IMOABC", "GDE3",
                "MOEAD", "NSGAII", "NSGAIII", "SPEA2", "OMOPSO", "SMPSO", "MODQNHH".
                Defaults to "".
        
        Returns:
            Algorithm: Configured algorithm instance ready to run.
        
        Note:
            All algorithms are configured with:
            - Maximum of 10000 function evaluations
            - FloatPolynomialMutation for continuous variables
            - FloatSimulatedBinaryCrossover for algorithms that support it
            - Standard comparator configurations for multi-objective selection
        """
        max_evaluations: int = 100000
        if algorithm_name == "MSAA":
            from evolu.optimizers.multiobjective.MOSA import MSAA
            from evolu.operator.mutation import FloatPolynomialMutation
            algorithm = MSAA(
                problem=problem,
                mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                 distribution_index=20.0),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            )
        if algorithm_name == "MRSA":
            from evolu.optimizers.multiobjective.MOSA import MRSA
            from evolu.operator.mutation import FloatPolynomialMutation
            algorithm = MRSA(
                problem=problem,
                mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                 distribution_index=20.0),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            )

        if algorithm_name == "MOABC":
            from evolu.optimizers.multiobjective.MOABC import MOABC
            from evolu.operator.selection import BinaryTournamentSelection
            from evolu.operator.mutation import FloatPolynomialMutation
            from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
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
            from evolu.util.comparator import MultiComparator, SolutionAttributeComparator
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
                population_size=100,
                crossover=FloatDifferentialEvolutionCrossover(CR=1.0, F=0.5),
                mutation=FloatPolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                 distribution_index=20),
                aggregation_function=Tschebycheff(dimension=problem.number_of_objectives),
                neighbor_size=20,
                neighborhood_selection_probability=0.9,
                max_number_of_replaced_solutions=2,
                weight_files_path="D:/GitHubCreated/IISO_EvoSuite/python/resources/MOEAD_weights",
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            )
        if algorithm_name == "NSGAII":
            from evolu.optimizers.multiobjective.MOGAII import NSGAII
            from evolu.operator.selection import BinaryTournamentSelection
            from evolu.operator.crossover import FloatSimulatedBinaryCrossover
            from evolu.operator.mutation import FloatPolynomialMutation
            from evolu.util.ranking import FastNonDominatedRanking
            from evolu.util.density_estimator import CrowdingDistance
            from evolu.util.comparator import MultiComparator
            algorithm = NSGAII(
                problem=problem,
                population_size=100,
                offspring_population_size=100,
                selection=BinaryTournamentSelection(
                    MultiComparator(
                        [FastNonDominatedRanking.get_comparator(), CrowdingDistance.get_comparator()])
                ),
                crossover=FloatSimulatedBinaryCrossover(probability=1.0, distribution_index=20),
                mutation=FloatPolynomialMutation(
                    probability=1.0 / problem.number_of_variables, distribution_index=20),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            )
        if algorithm_name == "OMOPSO":
            from evolu.optimizers.multiobjective.MOPSO import OMOPSO
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import BoundedCrowdingDistanceArchive
            algorithm = OMOPSO(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
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
                mutation=FloatPolynomialMutation(
                    probability=1.0 / problem.number_of_variables, distribution_index=20
                ),
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
        if algorithm_name == "IMOTLBO":
            from evolu.optimizers.multiobjective.MOTLBO import IMOTLBO
            from evolu.operator.mutation import FloatPolynomialMutation
            from evolu.util.archive import BoundedCrowdingDistanceArchive
            algorithm = IMOTLBO(
                problem=problem,
                population_size=100,
                mutation=FloatPolynomialMutation(
                    probability=1.0 / problem.number_of_variables, distribution_index=20
                ),
                leaders_archive=BoundedCrowdingDistanceArchive(100),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            )
        if algorithm_name == "DQNMOHH":
            from evolu.optimizers.multiobjective.MOHH import DQNMOHH
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import ModifiedNonDominatedSolutionsArchive
            algorithm = DQNMOHH(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
                ),
                leaders_archive=ModifiedNonDominatedSolutionsArchive(),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        if algorithm_name == "QMOHH":
            from evolu.optimizers.multiobjective.MOHH import QMOHH
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import ModifiedNonDominatedSolutionsArchive
            algorithm = QMOHH(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
                ),
                leaders_archive=ModifiedNonDominatedSolutionsArchive(),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        if algorithm_name == "MOHHv1":
            from evolu.optimizers.multiobjective.MOHH import MOHHv1
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import ModifiedNonDominatedSolutionsArchive
            algorithm = MOHHv1(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
                ),
                leaders_archive=ModifiedNonDominatedSolutionsArchive(),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        if algorithm_name == "MOHHv2":
            from evolu.optimizers.multiobjective.MOHH import MOHHv2
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import ModifiedNonDominatedSolutionsArchive
            algorithm = MOHHv2(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
                ),
                leaders_archive=ModifiedNonDominatedSolutionsArchive(),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        if algorithm_name == "MOHHv3":
            from evolu.optimizers.multiobjective.MOHH import MOHHv3
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import ModifiedNonDominatedSolutionsArchive
            algorithm = MOHHv3(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
                ),
                leaders_archive=ModifiedNonDominatedSolutionsArchive(),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        if algorithm_name == "MOHHv4":
            from evolu.optimizers.multiobjective.MOHH import MOHHv4
            from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
            from evolu.util.archive import BoundedCrowdingDistanceArchive
            algorithm = MOHHv4(
                problem=problem,
                population_size=100,
                uniform_mutation=FloatUniformMutation(
                    probability=1.0 / problem.number_of_variables, perturbation=0.5),
                non_uniform_mutation=FloatNonUniformMutation(
                    1.0 / problem.number_of_variables, perturbation=0.5,
                    max_iterations=int(max_evaluations / 100)
                ),
                leaders_archive=BoundedCrowdingDistanceArchive(100),
                epsilon=0.0075,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        return algorithm


if __name__ == "__main__":
    # Record start time
    start_time = time.time()
    print(f"Starting experiment: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configure the experiment
    file_path: str = "D:/GitHubInstances/Instances_RALSP/Instances/"
    # Get all entries in the directory
    entries: List[str] = os.listdir(file_path)
    # Filter out files
    problem_name_list: List[str] = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]
    # Remove example_of_paper.txt from the list
    problem_name_list = [name for name in problem_name_list if name != "example_of_paper.txt" and name != "example_of_english_paper.txt"]

    problems: Dict[str, FloatMOReconfigurableALSP] = {}
    for problem_name in problem_name_list:
        problems[problem_name] = FloatMOReconfigurableALSP(file_path, problem_name)

    algorithm_name_list: List[str] = [
        "MSAA",
        "MRSA",
        "MOABC",
        "IMOABC",
        "GDE3",
        "MOEAD",
        "NSGAII",
        "OMOPSO",
        "SMPSO",
        "QMOHH",
    ]
    output_directory: str = "ExperimentExecutionMO" + "/data"
    # experiment_execution = ExperimentExecutionMO()
    # experiment_execution.problems = problems
    # experiment_execution.algorithm_name_list = algorithm_name_list
    # experiment_execution.num_of_runs = 10
    # experiment_execution.output_dir = output_directory
    # experiment_execution.run_experiment()
    # Generate summary file
    algorithm_name_list: List[str] = [
        "MSAA",
        "MRSA",
        "MOABC",
        "IMOABC",
        "GDE3",
        "MOEAD",
        "NSGAII",
        "OMOPSO",
        "SMPSO",
        "QMOHH",
    ]
    indicator_name_list = ["GD", "IGD", "EPSILON", "HV", "NHV", ]
    pareto_front_file = []
    experiment_report = ExperimentReport()
    experiment_report.input_dir = output_directory
    experiment_report.problem_name_list = list(problems.keys())
    experiment_report.algorithm_name_list = algorithm_name_list
    experiment_report.indicator_name_list = indicator_name_list
    experiment_report.pareto_front_file = pareto_front_file
    experiment_report.pareto_front_directory = "D:/GitHubCreated/IISO_EvoSuite/python/resources/pareto_front"
    experiment_report.reference_front_directory = "D:/GitHubCreated/IISO_EvoSuite/python/resources/reference_front"
    experiment_report.reference_point_directory = "D:/GitHubCreated/IISO_EvoSuite/python/resources/reference_point"
    experiment_report.num_of_runs = 10
    experiment_report.is_single_objective = False
    experiment_report.generate_quality_indicators()

    # Output program execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
    # Play beep sound when execution completes (Windows only)
    try:
        winsound.Beep(440, 10000)  # Frequency 440Hz, duration 10000ms
        print("End of beep.")
    except Exception:
        pass  # Ignore if winsound is not available on this platform