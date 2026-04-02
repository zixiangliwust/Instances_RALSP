import copy
import random
import time
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.util.evaluator import Evaluator
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Hill climbing algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class HCBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Hill climbing algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] Mitchell, Melanie, John H. Holland, and Stephanie Forrest. 1993. "When will a genetic algorithm outperform hill climbing?"
    In Proceedings of the 6th International Conference on Neural Information Processing Systems, 51â€?. Denver, Colorado: Morgan Kaufmann Publishers Inc.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(HCBase, self).__init__(problem=problem)
        self.algorithm_name = "Hill climbing algorithm"
        self.neighbor_size = 10
        self.neighbors = []
        self.offspring_population_size = self.neighbor_size
        self.mutation_operator = mutation
        self.comparator = comparator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.solution = None

    def initialization(self, population_size=None):
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

    def evolve(self):
        self.neighbors = []
        for _ in range(0, self.neighbor_size):
            new_solution: Solution = self.mutation_operator.execute(self.solution)
            self.neighbors.append(new_solution)
        self.neighbors = self.evaluate(self.neighbors)
        best_solution = self.best_solution_selection.execute(self.neighbors)
        if self.comparator.compare(self.solution, best_solution) == 1:
            self.solution = copy.deepcopy(best_solution)

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        if self.comparator.compare(self.solution, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.solution)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Late acceptance hill climbing algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class LAHCBase(HCBase[S, R]):
    """
    Late acceptance hill climbing algorithm
    References:
    [1] Yuan, Biao, Chaoyong Zhang, and Xinyu Shao. 2015. "A late acceptance hill-climbing algorithm for balancing two-sided assembly lines with multiple constraints."
    Journal of Intelligent Manufacturing 26 (1):159-68. doi: 10.1007/s10845-013-0770-x.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(LAHCBase, self).__init__(problem=problem, mutation=mutation, comparator=comparator,
                                       population_evaluator=population_evaluator,
                                       termination_criterion=termination_criterion)
        self.algorithm_name = "Late acceptance hill climbing algorithm"

    def initialization(self, population_size=None):
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)
        self.neighbors = [copy.deepcopy(self.solution) for _ in range(0, self.neighbor_size)]

    def evolve(self):
        for j in range(0, self.neighbor_size):
            new_solution: Solution = self.mutation_operator.execute(self.solution)
            new_solution = self.evaluate_solution(new_solution)
            if self.comparator.compare(new_solution, self.neighbors[j], ) < 1:
                self.neighbors[j] = copy.deepcopy(new_solution)
                self.solution = copy.deepcopy(new_solution)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
