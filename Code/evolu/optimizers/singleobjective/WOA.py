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
Module: Basic whale optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class WOABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic whale optimization algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Mirjalili, Seyedali, and Andrew Lewis. 2016. "The Whale Optimization Algorithm."
    Advances in Engineering Software 95:51-67. doi: https://doi.org/10.1016/j.advengsoft.2016.01.008.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(WOABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Whale optimization algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)


    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            r1 = random.uniform(0.0, 1.0)  # r1 is a random number in [0,1]
            r2 = random.uniform(0.0, 1.0)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            for i in range(self.problem.number_of_variables):
                if random.uniform(0.0, 1.0) < 0.5:
                    if abs(a) < 1:
                        d_leader = abs(c * self.g_best.variables[i] - self.solutions[j].variables[i])
                        offsprings[j].variables[i] = self.g_best.variables[i] - a * d_leader
                    elif abs(a) >= 1:
                        idx = random.randint(0, self.population_size - 1)
                        d_idx = abs(c * self.solutions[idx].variables[i] - self.solutions[j].variables[i])
                        offsprings[j].variables[i] = self.solutions[idx].variables[i] - a * d_idx
                else:
                    l = - 2 * random.uniform(0.0, 1.0) + 1  # parameters in Eq. (2.5)
                    d_leader2 = abs(self.g_best.variables[i] - self.solutions[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2 * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + self.g_best.variables[i]
        return offsprings

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Hybrid improved whale optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class WOAHI(WOABase):
    """
    Hybrid improved whale optimization algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] Tang, C., W. Sun, W. Wu, and M. Xue. 2019. A hybrid improved whale optimization algorithm. 
    Paper presented at the 2019 IEEE 15th International Conference on Control and Automation (ICCA), 16-19 July 2019.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(WOAHI, self).__init__(problem=problem, population_size=population_size,
                                    population_generator=population_generator,
                                    population_evaluator=population_evaluator,
                                    termination_criterion=termination_criterion)
        self.algorithm_name = "Hybrid improved whale optimization algorithm"
        self.non_improvement_time_limit = 20
        self.non_improvement_time = 0
        self.number_of_new_solutions = int(self.population_size / 2)

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 + 2 * math.cos(math.pi / 2 * (1 + self.iterations / self.max_iterations))  # Eq. 8
        w = 0.5 + 0.5 * pow(self.iterations / self.max_iterations, 2)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            r1 = random.uniform(0.0, 1.0)  # r1 is a random number in [0,1]
            r2 = random.uniform(0.0, 1.0)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            for i in range(self.problem.number_of_variables):
                if random.uniform(0.0, 1.0) < 0.5:
                    if abs(a) < 1:
                        d_leader = abs(c * self.g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = w * self.g_best.variables[i] - a * d_leader
                    elif abs(a) >= 1:
                        idx = random.randint(0, self.population_size - 1)
                        d_idx = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx
                else:
                    l = - 2 * random.uniform(0.0, 1.0) + 1  # parameters in Eq. (2.5)
                    d_leader2 = abs(self.g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = w * self.g_best.variables[i] + d_leader2 * math.exp(b * l) * math.cos(
                        l * 2 * math.pi)
        return offsprings

    def restart(self):
        self.offspring_population_size = self.population_size
        if self.non_improvement_time >= self.non_improvement_time_limit:
            self.non_improvement_time = 0
            idx_list = random.sample(range(0, self.population_size), self.number_of_new_solutions)
            for idx in idx_list:
                self.solutions[idx] = self.create_solution()
                self.solutions[idx] = self.evaluate_solution(self.solutions[idx])
            self.offspring_population_size += self.number_of_new_solutions

    def evolve(self) -> None:
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.restart()

    def after_evolve(self) -> None:
        self.non_improvement_time += 1
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.non_improvement_time = 0
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
