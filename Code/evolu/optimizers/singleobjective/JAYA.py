# -*- coding: utf-8 -*-
import copy
import random
import numpy as np
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.operator.selection import BestSolutionSelection, WorstSolutionSelection
from evolu.util.comparator import ObjectiveComparator
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic Jaya algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class JAYABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic Jaya algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Venkata Rao, R. 2016. "Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems."
    International Journal of Industrial Engineering Computations 7:19-34. doi: 10.5267/j.ijiec.2015.8.004.
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
        super(JAYABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Jaya algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.p_worst = None
        self.worst_solution_selection = WorstSolutionSelection(comparator=ObjectiveComparator(0))

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            for i in range(0, self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0, 1) * (
                        self.g_best.variables[i] - abs(population[j].variables[i])) - random.uniform(0, 1) * (
                                                     self.p_worst.variables[i] - abs(population[j].variables[i]))
        return offsprings

    def after_initialization(self):
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        self.g_best = copy.deepcopy(self.p_best)
        self.p_worst = copy.deepcopy(self.worst_solution_selection.execute(self.solutions))

    def after_evolve(self):
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        self.p_worst = copy.deepcopy(self.worst_solution_selection.execute(self.solutions))

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name