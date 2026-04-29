# -*- coding: utf-8 -*-
import copy
import time
import random
import math
import numpy as np
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.operator.replacement import JoinPopulationSelectionReplacement
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic moth-flame optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MFOBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic moth-flame optimization algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Mirjalili, Seyedali. 2015. "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm."
    Knowledge-Based Systems 89:228-49. doi: https://doi.org/10.1016/j.knosys.2015.07.006.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Input parameters:
        population_size (int): number of population size, default = 100; [2, 10000]
        """
        super(MFOBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Moth-flame optimization"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.replacement_operator = JoinPopulationSelectionReplacement(self.comparator)
        self.local_best_solutions = None

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = self.sort_population.execute(self.solutions)

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(
            self.population_size - (self.iterations + 1) * ((self.population_size - 1) / self.max_iterations))
        # Ensure num_flame is within valid range [0, population_size - 1]
        num_flame = max(0, min(num_flame, self.population_size - 1))
        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + (self.iterations + 1) * ((-1) / self.max_iterations)
        for j in range(0, self.population_size):
            for i in range(self.problem.number_of_variables):
                #   D in Eq.(3.13)
                distance_to_flame = abs(self.local_best_solutions[j].variables[i] - self.solutions[j].variables[i])
                t = (a - 1) * random.uniform(0.0, 1.0) + 1
                b = 1
                if j <= num_flame:  # Update the variables of the moth with respect to its corresponding flame
                    # Eq.(3.12)
                    offsprings[j].variables[i] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + \
                                                 self.local_best_solutions[j].variables[i]
                else:  # Update the variables of the moth with respect to one flame
                    # Eq.(3.12).
                    offsprings[j].variables[i] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + \
                                                 self.local_best_solutions[num_flame].variables[i]
        return offsprings

    def evolve(self):
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = copy.deepcopy(offsprings)
        self.local_best_solutions = self.replacement(self.local_best_solutions, offsprings)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name