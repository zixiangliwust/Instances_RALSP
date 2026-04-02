# -*- coding: utf-8 -*-
import copy
import random
import threading
import time
from typing import List, TypeVar
import numpy
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.generator import Generator
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.util.evaluator import Evaluator, SequentialEvaluator
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Variable neighborhood search
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class VNSBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Variable neighborhood search
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation_operator_list: list,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(VNSBase, self).__init__(problem=problem)
        self.algorithm_name = "Variable neighborhood search"
        self.mutation_operator_list = mutation_operator_list
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.sub_max_iteration = 5
        self.solution = None

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
        self.offspring_population_size = 0
        for k in range(0, len(self.mutation_operator_list)):
            j = 0
            while True:
                new_solution: Solution = self.mutation_operator_list[k].execute(self.solution)
                new_solution = self.evaluate_solution(new_solution)
                self.offspring_population_size = self.offspring_population_size + 1
                if self.comparator.compare(new_solution, self.solution) == -1:
                    j = 0
                    self.solution = new_solution
                    if self.comparator.compare(self.solution, self.g_best) == -1:
                        self.g_best = copy.deepcopy(self.solution)
                j = j + 1
                if j >= self.sub_max_iteration:
                    break

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        pass

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name



"""
Module: Variable neighborhood search with simulated annealing_based acceptance
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class VNSSA(SingleObjectiveSwarmRoot[S, R]):
    """
    Variable neighborhood search with simulated annealing_based acceptance
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation_operator_list: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(VNSSA, self).__init__(problem=problem)
        self.algorithm_name = "Variable neighborhood search with simulated annealing_based acceptance"
        self.mutation_operator_list = mutation_operator_list
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.offspring_population_size = self.sub_max_iteration * len(self.mutation_operator_list)
        self.solution = None

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
        for k in range(0, len(self.mutation_operator_list)):
            for _ in range(0, self.sub_max_iteration):
                new_solution: Solution = self.mutation_operator_list[k].execute(self.solution)
                new_solution = self.evaluate_solution(new_solution)
                acceptance_probability = self.compute_acceptance_probability(
                    self.solution.objectives[0], new_solution.objectives[0], self.temperature
                )
                if self.comparator.compare(new_solution, self.solution) == -1:
                    self.solution = new_solution
                    if self.comparator.compare(self.solution, self.g_best) == -1:
                        self.g_best = copy.deepcopy(self.solution)
                elif acceptance_probability > random.random():
                    self.solution = new_solution
        self.temperature *= self.cooling_rate

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        pass

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
