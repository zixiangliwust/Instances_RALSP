# -*- coding: utf-8 -*-
import copy
import random
import time
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic ant colony optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP1PermutationACO(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic ant colony optimization
    References:
    [1] Li, Zixiang, Ibrahim Kucukkoc, and Qiuhua Tang. 2017. "New MILP model and station-oriented ant colony optimization algorithm for balancing U-type assembly lines."
    Computers & Industrial Engineering 112:107-21. doi: http://dx.doi.org/10.1016/j.cie.2017.07.005.
    """

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ) -> None:
        """Initialize SALBP1 permutation ACO algorithm."""
        super(SALBP1PermutationACO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name: str = "Ant colony optimization"
        self.population_generator: Generator = population_generator
        self.population_evaluator: Evaluator = population_evaluator
        self.termination_criterion: TerminationCriterion = termination_criterion
        self.observable.register(termination_criterion)
        self.t0: float = 5.0
        self.Q: int = 50
        self.rho: float = 0.1
        self.first_task_prob: List[float] = [self.t0 for _ in range(0, self.problem.number_of_variables)]
        self.task_task_prob: List[List[float]] = []
        for i in range(0, self.problem.number_of_variables):
            self.task_task_prob.append([self.t0 for _ in range(0, self.problem.number_of_variables)])

    def initialization(self) -> None:
        self.first_task_prob = [self.t0 for _ in range(0, self.problem.number_of_variables)]
        self.task_task_prob = []
        for i in range(0, self.problem.number_of_variables):
            self.task_task_prob.append([self.t0 for _ in range(0, self.problem.number_of_variables)])
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        for j in range(0, self.population_size):
            new_solution = self.problem.create_solution1(self.first_task_prob, self.task_task_prob)
            self.solutions.append(new_solution)

    def create_initial_solutions(self):
        self.solutions.clear()
        for j in range(0, self.population_size):
            solution = self.problem.create_solution1(self.first_task_prob, self.task_task_prob)
            self.solutions.append(solution)

    def update_probability_matrix(self):
        for i in range(0, self.problem.number_of_variables):
            self.first_task_prob[i] = (1 - self.rho) * self.first_task_prob[i] + self.rho * self.t0
        m = self.g_best.variables[0]
        self.first_task_prob[m] = (1 - self.rho) * self.first_task_prob[m] + self.rho * self.Q / self.g_best.objectives[
            1]
        for i in range(0, self.problem.number_of_variables):
            for h in range(0, self.problem.number_of_variables):
                if i != h:
                    self.task_task_prob[i][h] = (1 - self.rho) * self.task_task_prob[i][h] + self.rho * self.t0
        for i in range(1, self.problem.number_of_variables):
            m = self.g_best.variables[i - 1]
            n = self.g_best.variables[i]
            self.task_task_prob[m][n] = (1 - self.rho) * self.task_task_prob[m][n] + self.rho * self.Q / \
                                        self.g_best.objectives[1]

    def evolve(self):
        self.update_probability_matrix()
        self.create_initial_solutions()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name



"""
Module: Basic ant colony optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP2ntergerACO(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic ant colony optimization
    References:
    [1] Li, Zixiang, Ibrahim Kucukkoc, and Qiuhua Tang. 2017. "New MILP model and station-oriented ant colony optimization algorithm for balancing U-type assembly lines."
    Computers & Industrial Engineering 112:107-21. doi: http://dx.doi.org/10.1016/j.cie.2017.07.005.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(SALBP2ntergerACO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Ant colony optimization"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.tau_min = 0.01
        self.tau_max = 0.99
        self.rho = 0.1
        self.task_station_prob = []
        for i in range(0, self.problem.number_of_variables):
            self.task_station_prob.append([0.5 for _ in range(0, self.problem.number_of_variables)])

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        for j in range(0, self.population_size):
            new_solution = self.problem.create_solution2(self.task_station_prob)
            self.solutions.append(new_solution)

    def create_initial_solutions(self) -> None:
        """Create initial solutions for the population."""
        self.solutions.clear()
        for j in range(0, self.population_size):
            solution = self.problem.create_solution2(self.task_station_prob)
            self.solutions.append(solution)

    def update_probability_matrix(self) -> None:
        """Update probability matrix based on best solution."""
        if len(self.solutions) > 0:
            self.p_best = self.best_solution_selection.execute(self.solutions)
        else:
            self.p_best = copy.deepcopy(self.g_best)
        n_stations = int(self.p_best.objectives[0])
        for i in range(0, self.problem.number_of_variables):
            for j in range(0, n_stations):
                if self.p_best.variables[i] == j:
                    self.task_station_prob[i][j] = min(
                        max(self.tau_min, self.task_station_prob[i][j] + self.rho * (1 - self.task_station_prob[i][j])),
                        self.tau_max)
                else:
                    self.task_station_prob[i][j] = min(
                        max(self.tau_min, self.task_station_prob[i][j] - self.rho * self.task_station_prob[i][j]),
                        self.tau_max)
        convergence_factor = 0.0
        for i in range(0, self.problem.number_of_variables):
            for j in range(0, n_stations):
                convergence_factor = convergence_factor + min(self.tau_max - self.task_station_prob[i][j],
                                                              self.task_station_prob[i][j] - self.tau_min)
        convergence_factor = 2 * convergence_factor / (
                self.problem.number_of_variables * n_stations * (self.tau_max - self.tau_min))
        if convergence_factor < 0.05:
            for i in range(0, self.problem.number_of_variables):
                for j in range(0, n_stations):
                    self.task_station_prob[i][j] = 0.5

    def evolve(self):
        self.update_probability_matrix()
        self.create_initial_solutions()
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
