import copy
import io
import os
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, TypeVar

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu, iqr, ks_2samp

from evolu.core.algorithm import Algorithm
from evolu.core.problem import Problem
from evolu.core.quality_indicator import *
from evolu.logger import get_logger
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.solution import print_function_values_to_file, print_variables_to_file, read_solutions

S = TypeVar("S")

logger = get_logger(__name__)
"""
module:: laboratory
synopsis: Run experiment. WIP!
moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class ExperimentIndividual:
    """Individual experiment task for algorithm execution.
    
    This class represents a single experiment run, encapsulating:
    - An algorithm instance configured for a specific problem
    - Algorithm and problem names for identification
    - Run number for multiple independent runs
    
    Used by ExperimentExecution to organize and execute multiple experimental runs.
    
    Attributes:
        algorithm (Algorithm[S, S]): Configured algorithm instance to execute.
        algorithm_name (str): Name of the algorithm (for identification and output organization).
        problem_name (str): Name of the problem (for identification and output organization).
        num_run (int): Run number (for independent runs with same configuration).
    """
    
    def __init__(self, algorithm: Algorithm[S, S], algorithm_name: str, problem_name: str, num_run: int) -> None:
        """Initialize experiment individual.
        
        Args:
            algorithm (Algorithm[S, S]): Configured algorithm instance ready to run.
            algorithm_name (str): Name identifier for the algorithm.
            problem_name (str): Name identifier for the problem.
            num_run (int): Run number (typically 0 to num_of_runs-1).
        """
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        self.problem_name = problem_name
        self.num_run = num_run

    def execute(self, output_path: str = "") -> None:
        """Execute the algorithm and save results to files.
        
        Runs the algorithm and optionally saves results to files:
        - FUN.{num_run}: Objective function values
        - VAR.{num_run}: Decision variable values
        - TIME.{num_run}: Total computing time
        
        Args:
            output_path (str, optional): Directory path where results should be saved.
                If empty string, results are not saved. Defaults to "".
        
        Note:
            Files are saved in the specified output_path directory. Subdirectories
            are not created automatically.
        """
        self.algorithm.run()
        if output_path:
            file_name = os.path.join(output_path, "FUN.{}".format(self.num_run))
            print_function_values_to_file(self.algorithm.get_result(), file_name=file_name)
            file_name = os.path.join(output_path, "VAR.{}".format(self.num_run))
            print_variables_to_file(self.algorithm.get_result(), file_name=file_name)
            file_name = os.path.join(output_path, "TIME.{}".format(self.num_run))
            with open(file_name, "w+") as of:
                of.write(str(self.algorithm.total_computing_time))

    def get_algorithm_data(self) -> dict:
        """Get algorithm observable data.
        
        Retrieves data collected by the algorithm's observable system during execution.
        
        Returns:
            dict: Dictionary containing observable data (evaluations, iterations,
                solutions, computing time, etc.) collected during algorithm execution.
        """
        return self.algorithm.get_observable_data()


class ExperimentExecution(ABC):
    """Base class for executing experimental studies.
    
    This class provides infrastructure for running systematic experimental studies
    comparing multiple algorithms on multiple problems across multiple independent
    runs. Experiments are executed in parallel using ProcessPoolExecutor.
    
    The experiment structure:
    - Multiple problems (self.problems)
    - Multiple algorithms (self.algorithm_name_list)
    - Multiple independent runs (self.num_of_runs)
    
    Results are saved in a structured directory hierarchy:
        output_dir/
            algorithm_name/
                problem_name/
                    FUN.0, FUN.1, ... (objective values)
                    VAR.0, VAR.1, ... (decision variables)
                    TIME.0, TIME.1, ... (computing times)
    
    Attributes:
        m_workers (int): Maximum number of parallel workers for execution.
        output_dir (Optional[str]): Output directory for experiment results.
        problems (Dict[str, Problem[S]]): Dictionary mapping problem names to
            problem instances.
        algorithm_name_list (List[str]): List of algorithm names to test.
        num_of_runs (int): Number of independent runs per algorithm-problem pair.
        experiment_individual_list (List[ExperimentIndividual]): List of all
            experiment tasks to execute.
        job_data (List): Additional job metadata (if needed).
    
    Note:
        Subclasses must implement algorithm_settings() to configure algorithms
        for each problem.
    """
    
    def __init__(self, m_workers: int = 6) -> None:
        """Initialize experiment execution framework.
        
        Args:
            m_workers (int, optional): Maximum number of parallel workers for
                executing experiments. Defaults to 6.
        """
        self.m_workers: int = m_workers
        self.output_dir: Optional[str] = None
        self.problems: Dict[str, Problem[S]] = {}
        self.algorithm_name_list: List[str] = []
        self.num_of_runs: int = 10
        self.experiment_individual_list: List[ExperimentIndividual] = []
        self.job_data: List = []

    @abstractmethod
    def algorithm_settings(self, problem: Problem[S], algorithm_name: str = "") -> Algorithm[S, S]:
        """Configure algorithm for the given problem.
        
        This method must create and configure an algorithm instance for the
        specified problem. Different algorithm names should result in different
        algorithm configurations.
        
        Args:
            problem (Problem[S]): Problem instance for which to configure the algorithm.
            algorithm_name (str, optional): Name of the algorithm to configure.
                Different names should result in different algorithm types or configurations.
                Defaults to "".
        
        Returns:
            Algorithm[S, S]: Fully configured algorithm instance ready to run.
        
        Note:
            Subclasses must implement this method to define how algorithms are
            configured for each problem. This is where algorithm-specific settings
            (population size, operators, termination criteria, etc.) are specified.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `algorithm_settings`.")

    def run_experiment(self) -> None:
        for algorithm_name in self.algorithm_name_list:
            for problem_name, problem in self.problems.items():
                for num_run in range(self.num_of_runs):
                    algorithm = self.algorithm_settings(problem, algorithm_name)
                    self.experiment_individual_list.append(
                        ExperimentIndividual(
                            algorithm=algorithm,
                            algorithm_name=algorithm_name,
                            problem_name=problem_name,
                            num_run=num_run,
                        )
                    )
        with ProcessPoolExecutor(max_workers=self.m_workers) as executor:
            for job in self.experiment_individual_list:
                output_path = os.path.join(self.output_dir, job.algorithm_name, job.problem_name)
                executor.submit(job.execute, output_path)


class ExperimentReport:
    """Generate comprehensive quality indicator reports from experiment results.
    
    This class processes experiment results generated by ExperimentExecution
    and computes quality indicators for each algorithm-problem-run combination.
    Results are saved to CSV files for further analysis.
    
    The input directory structure (automatically generated by ExperimentExecution):
        input_dir/
            algorithm_name/
                problem_name/
                    FUN.0, FUN.1, ... (objective values)
                    VAR.0, VAR.1, ... (decision variables)
                    TIME.0, TIME.1, ... (computing times)
    
    Attributes:
        input_dir (str): Directory containing experiment results.
        problem_name_list (str): List of problem names to process.
        algorithm_name_list (str): List of algorithm names to process.
        indicator_name_list (List[str]): List of quality indicator names to compute.
        pareto_front_file (List[str]): List of reference Pareto front file names.
        pareto_front_directory (str): Directory containing reference Pareto fronts.
        reference_front_directory (str): Directory for reference fronts.
        reference_point_directory (str): Directory for reference points.
        num_of_runs (int): Number of independent runs per algorithm-problem pair.
        is_single_objective (bool): Whether problems are single-objective.
    
    Example:
        >>> report = ExperimentReport()
        >>> report.input_dir = "results/"
        >>> report.problem_name_list = ["ZDT1", "ZDT2"]
        >>> report.algorithm_name_list = ["NSGA-II", "MOEA/D"]
        >>> report.indicator_name_list = ["IGD", "HV"]
        >>> report.generate_quality_indicators()
    """
    
    __EPS = 1.0e-14

    def __init__(self):
        """Initialize experiment report generator.
        
        Creates an ExperimentReport instance with default settings. Configure
        the attributes before calling generate_quality_indicators().
        
        The input data directory must match the structure generated by
        ExperimentExecution class.
        """
        self.input_dir: str = ""
        self.problem_name_list: str = ""
        self.algorithm_name_list: str = ""
        self.indicator_name_list: List[str] = []
        self.pareto_front_file: List[str] = []
        self.pareto_front_directory: str = ""
        self.reference_front_directory: str = ""
        self.reference_point_directory: str = ""
        self.num_of_runs: int = 10
        self.is_single_objective: bool = False

    def generate_quality_indicators(self):
        if not self.is_single_objective and len(self.pareto_front_file) == 0:
            self.generate_reference_fronts()
        if not self.is_single_objective:
            self.generate_reference_point()
        
        # Determine output directory for the summary file
        # Use input_dir parent directory or default to current directory with 'QualityIndicatorSummary.csv'
        input_path = Path(self.input_dir)
        # Use input_dir as the base and place summary file in its parent directory
        summary_file_path = input_path.parent / "QualityIndicatorSummary.csv"
        
        with open(summary_file_path, "w+") as of:
            of.write("Algorithm,Problem,ExecutionId,IndicatorName,IndicatorValue\n")
        for algorithm_name in self.algorithm_name_list:
            for i in range(len(self.problem_name_list)):
                problem_name = self.problem_name_list[i]
                if not self.is_single_objective:
                    if len(self.pareto_front_file) == 0:
                        pareto_front_path = self.reference_front_directory + '/' + problem_name + ".rf"
                    else:
                        pareto_front_path = os.path.join(self.pareto_front_directory, self.pareto_front_file[i])
                    reference_point_path = self.reference_point_directory + '/' + problem_name + ".rf"
                for num_run in range(self.num_of_runs):
                    front_solutions_time_file = self.input_dir + '/' + algorithm_name + '/' + \
                                                problem_name + '/' + 'TIME.' + str(num_run)
                    with open(front_solutions_time_file, "r") as content_file:
                        content = content_file.read()
                    with open(summary_file_path, "a+") as of:
                        of.write(",".join([algorithm_name, problem_name, str(num_run), "Time", str(content)]))
                        of.write("\n")
                for indicator_name in self.indicator_name_list:
                    for num_run in range(self.num_of_runs):
                        front_solutions_file = self.input_dir + '/' + algorithm_name + '/' + \
                                               problem_name + '/' + 'FUN.' + str(num_run)
                        front_solutions = read_solutions(front_solutions_file)
                        if self.is_single_objective:
                            if indicator_name == "FIT":
                                indicator = FitnessValue()
                                # Get all fitness values from the first solution
                                fitness_values = front_solutions[0].objectives
                                
                                # Output multiple fitness values for single-objective with hierarchical objectives
                                if len(fitness_values) > 1:
                                    for fit_idx, fit_value in enumerate(fitness_values):
                                        with open(summary_file_path, "a+") as of:
                                            of.write(",".join([algorithm_name, problem_name, str(num_run), f"Fit{fit_idx + 1}", str(fit_value)]))
                                            of.write("\n")
                                else:
                                    result = fitness_values[0]
                                    with open(summary_file_path, "a+") as of:
                                        of.write(",".join([algorithm_name, problem_name, str(num_run), "Fit", str(result)]))
                                        of.write("\n")
                        else:
                            if Path(pareto_front_path).is_file():
                                reference_front = []
                                with open(pareto_front_path) as file:
                                    for line in file:
                                        reference_front.append([float(x) for x in line.split()])
                                reference_points = []
                                with open(reference_point_path) as file:
                                    for line in file:
                                        reference_points.append([float(x) for x in line.split()])
                                reference_point = reference_points[0]
                            else:
                                logger.warning("Reference front not found at", pareto_front_path)
                            if indicator_name == "GD":
                                indicator = GenerationalDistance(reference_front=reference_front)
                            elif indicator_name == "IGD":
                                indicator = InvertedGenerationalDistance(reference_front=reference_front)
                            elif indicator_name == "SPREAD":
                                indicator = SpreadIndicator(reference_front=reference_front)
                            elif indicator_name == "EPSILON":
                                indicator = EpsilonIndicator(reference_front=reference_front)
                            elif indicator_name == "HV":
                                # For HV, we need to use the reference point (nadir point) not the reference front
                                indicator = HyperVolume(reference_point=reference_point)
                            elif indicator_name == "NHV":
                                indicator = NormalizedHyperVolume(reference_point=reference_point,
                                                                  reference_front=reference_front)
                            else:
                                logger.warning(f"Unknown indicator {indicator_name} for {problem_name}")
                                continue
                            result = indicator.compute(
                                [front_solutions[i].objectives for i in range(len(front_solutions))])                            
                            # Output quality indicator to summary file
                            with open(summary_file_path, "a+") as of:
                                of.write(",".join([algorithm_name, problem_name, str(num_run), indicator_name,
                                                  str(result)]))
                                of.write("\n")


    def generate_reference_fronts(self):
        for problem_name in self.problem_name_list:
            non_dominated_solutions_archive = NonDominatedSolutionsArchive()
            pareto_front_path = self.reference_front_directory + "/" + problem_name + ".rf"
            for algorithm_name in self.algorithm_name_list:
                for num_run in range(self.num_of_runs):
                    front_solutions_file = self.input_dir + '/' + algorithm_name + '/' + \
                                           problem_name + '/' + 'FUN.' + str(num_run)
                    front_solutions = read_solutions(front_solutions_file)
                    for solution in front_solutions:
                        non_dominated_solutions_archive.add(solution)
            print_function_values_to_file(non_dominated_solutions_archive.solution_list, file_name=pareto_front_path)

    def generate_reference_point(self):
        for problem_name in self.problem_name_list:
            reference_solution = None
            reference_point_path = self.reference_point_directory + "/" + problem_name + ".rf"
            for algorithm_name in self.algorithm_name_list:
                for num_run in range(self.num_of_runs):
                    front_solutions_file = self.input_dir + '/' + algorithm_name + '/' + \
                                           problem_name + '/' + 'FUN.' + str(num_run)
                    front_solutions = read_solutions(front_solutions_file)
                    if front_solutions:  # Check if solutions list is not empty
                        if reference_solution is None:
                            reference_solution = copy.deepcopy(front_solutions[0])
                            # Initialize reference_solution objectives with minimum values
                            for i in range(front_solutions[0].number_of_objectives):
                                reference_solution.objectives[i] = float('-inf')
                        for solution in front_solutions:
                            for i in range(solution.number_of_objectives):
                                reference_solution.objectives[i] = max(reference_solution.objectives[i], solution.objectives[i])
            if reference_solution is not None:
                # Add small epsilon to ensure reference point is truly dominated by all solutions
                for i in range(reference_solution.number_of_objectives):
                    reference_solution.objectives[i] += self.__EPS
                print_function_values_to_file(reference_solution, file_name=reference_point_path)
            else:
                logger.warning(f"No solutions found for problem {problem_name}, cannot generate reference point")

    def check_minimization(self, indicator) -> bool:
        if indicator == "HV":
            return False
        else:
            return True
