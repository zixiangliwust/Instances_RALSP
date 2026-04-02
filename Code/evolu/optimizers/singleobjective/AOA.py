# -*- coding: utf-8 -*-
import copy
import numpy as np
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
Module: Basic arithmetic optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class AOABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic arithmetic optimization algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://www.mathworks.com/matlabcentral/fileexchange/84742-the-arithmetic-optimization-algorithm-aoa
    [3] https://seyedalimirjalili.com/projects
    [4] Abualigah, Laith, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz, and Amir H. Gandomi. 2021. "The Arithmetic Optimization Algorithm."
    Computer Methods in Applied Mechanics and Engineering 376:113609. doi: https://doi.org/10.1016/j.cma.2020.113609.
    """
    __EPS = 1.0e-14

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
        alpha (int): fixed parameter, sensitive exploitation parameter, Default: 5; [2, 10]
        miu (float): fixed parameter, control parameter to adjust the search process, Default: 0.5; [0.1, 2.0]
        moa_min (float): range min of Math SwarmRoot Accelerated, Default: 0.2; (0, 0.41)
        moa_max (float): range max of Math SwarmRoot Accelerated, Default: 0.9; (0.41, 1.0)
        """
        super(AOABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Arithmetic optimization algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.alpha = 5
        self.miu = 0.5
        self.moa_min = 0.2
        self.moa_max = 0.9


    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        moa = self.moa_min + (self.iterations + 1) * ((self.moa_max - self.moa_min) / self.max_iterations)  # Eq. 2
        mop = 1 - ((self.iterations + 1) ** (1.0 / self.alpha)) / (self.max_iterations ** (1.0 / self.alpha))  # Eq. 4
        for j in range(0, self.population_size):
            for i in range(0, self.problem.number_of_variables):
                r1, r2, r3 = np.random.rand(3)
                if r1 > moa:  # Exploration phase
                    if r2 < 0.5:
                        offsprings[j].variables[i] = self.g_best.variables[i] / (mop + self.__EPS) * \
                                                     ((self.problem.upper_bound[i] - self.problem.lower_bound[
                                                         i]) * self.miu +
                                                      self.problem.lower_bound[i])
                    else:
                        offsprings[j].variables[i] = self.g_best.variables[i] * mop * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * self.miu +
                                self.problem.lower_bound[i])
                else:  # Exploitation phase
                    if r3 < 0.5:
                        offsprings[j].variables[i] = self.g_best.variables[i] - mop * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * self.miu +
                                self.problem.lower_bound[i])
                    else:
                        offsprings[j].variables[i] = self.g_best.variables[i] + mop * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * self.miu +
                                self.problem.lower_bound[i])
        return offsprings

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
