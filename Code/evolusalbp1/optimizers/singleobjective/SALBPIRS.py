import copy
import time
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger
from evolu.optimizers.singleobjective.RS import RSBase


logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")

"""
Module: Random search[see 2.13. Random Search in Pape 2015]
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP1RandSearch(RSBase[S, R]):
    """
    Random search[see 2.13. Random Search in Pape 2015];
    Random Search assigns the next task from the list of available tasks randomly;
    References:
    [1] Pape, Tom. 2015. "Heuristics and lower bounds for the simple assembly line balancing problem type 1: Overview, computational tests and improvements."
    European Journal of Operational Research 240 (1):32-42. doi: http://dx.doi.org/10.1016/j.ejor.2014.06.023.
    """

    def __init__(self,
                 problem: Problem[S],
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        super(SALBP1RandSearch, self).__init__(problem=problem, comparator=comparator,
                                               termination_criterion=termination_criterion)
        self.algorithm_name = "Random Search"

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.problem.create_evaluate_random_solution()

    def evolve(self) -> None:
        self.solution = self.problem.create_evaluate_random_solution()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Random task priority search[see 2.14. Random Task Priority Search in Pape 2015]
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP1RandomPriority(RSBase[S, R]):
    """
    Random task priority search[see 2.14. Random Task Priority Search in Pape 2015];
    Random Task Priority Search apply a roulette wheel selection according to the task priorities raised to the power;
    References:
    [1] Pape, Tom. 2015. "Heuristics and lower bounds for the simple assembly line balancing problem type 1: Overview, computational tests and improvements."
    European Journal of Operational Research 240 (1):32-42. doi: http://dx.doi.org/10.1016/j.ejor.2014.06.023.
    """

    def __init__(self,
                 problem: Problem[S],
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        super(SALBP1RandomPriority, self).__init__(problem=problem, comparator=comparator,
                                                   termination_criterion=termination_criterion)
        self.algorithm_name = "Random task priority search"

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.problem.create_evaluate_heuristic_solution()

    def evolve(self) -> None:
        self.solution = self.problem.create_evaluate_heuristic_solution()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name

