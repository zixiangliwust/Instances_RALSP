import copy
import time
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Random search algorithm (Simple random search algorithms)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class RSBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Random search algorithm
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy;
    Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    """

    def __init__(self,
                 problem: Problem[S],
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        super(RSBase, self).__init__(problem=problem)
        self.algorithm_name = "Random Search"
        self.problem = problem
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.solution = None
        self.offspring_population_size = 1

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def evolve(self) -> None:
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        if self.comparator.compare(self.solution, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.solution)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
