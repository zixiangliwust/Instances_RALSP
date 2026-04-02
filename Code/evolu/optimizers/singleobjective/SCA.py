# -*- coding: utf-8 -*-
import copy
import math
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.evaluator import Evaluator, SequentialEvaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic sine cosine algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SCABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic sine cosine algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] https://www.mathworks.com/matlabcentral/fileexchange/54948-sca-a-sine-cosine-algorithm
    [4] Mirjalili, Seyedali. 2016. "SCA: A Sine Cosine Algorithm for solving optimization problems."
    Knowledge-Based Systems 96:120-33. doi: https://doi.org/10.1016/j.knosys.2015.12.022.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(SCABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Sine cosine algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)


    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        a = 2.0
        r1 = a - (self.iterations + 1) * a / float(self.max_iterations)
        for j in range(0, self.population_size):
            # Eq 3.4, r1 decreases linearly from a to 0
            for i in range(self.problem.number_of_variables):
                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = 2 * math.pi * random.random()
                r3 = 2 * random.random()
                r4 = random.random()
                # Eq. 3.3, 3.1 and 3.2
                if r4 < 0.5:
                    offsprings[j].variables[i] = population[j].variables[i] + r1 * math.sin(r2) * abs(
                        r3 * self.g_best.variables[i] - population[j].variables[i])
                else:
                    offsprings[j].variables[i] = population[j].variables[i] + r1 * math.cos(r2) * abs(
                        r3 * self.g_best.variables[i] - population[j].variables[i])
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return copy.deepcopy(offsprings)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name

