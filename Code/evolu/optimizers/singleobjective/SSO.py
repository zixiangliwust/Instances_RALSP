# -*- coding: utf-8 -*-
import copy
import random
import math
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic salp swarm optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SSOBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic salp swarm optimization
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Mirjalili, Seyedali, Amir H. Gandomi, Seyedeh Zahra Mirjalili, Shahrzad Saremi, Hossam Faris, and Seyed Mohammad Mirjalili. 2017.
    "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems."  Advances in Engineering Software 114:163-91.
    doi: https://doi.org/10.1016/j.advengsoft.2017.07.002.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(SSOBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Salp swarm optimization"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)


    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        # Eq. (3.2) in the paper
        c1 = 2 * math.exp(-((4 * (self.iterations + 1) / self.max_iterations) ** 2))
        for j in range(0, self.population_size):
            if j < self.population_size / 2:
                for i in range(0, self.problem.number_of_variables):
                    c2 = random.random()
                    if random.random() < 0.5:
                        offsprings[j].variables[i] = self.g_best.variables[i] + c1 * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * c2 +
                                self.problem.lower_bound[i])
                    else:
                        offsprings[j].variables[i] = self.g_best.variables[i] - c1 * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * c2 +
                                self.problem.lower_bound[i])
            else:
                # Eq. (3.4) in the paper
                for i in range(0, self.problem.number_of_variables):
                    offsprings[j].variables[i] = (self.solutions[j].variables[i] + self.solutions[j - 1].variables[
                        i]) / 2
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return copy.deepcopy(offsprings)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name

