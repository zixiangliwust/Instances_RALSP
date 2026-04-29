# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic teaching–learning-based optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class TLBOBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic teaching–learning-based optimization
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/andaviaco/tblo
    [3] Rao, R. V., V. J. Savsani, and D. P. Vakharia. 2011. "Teaching–learning-based optimization: A novel method for constrained mechanical design optimization problems."
    Computer-Aided Design 43 (3):303-15. doi: https://doi.org/10.1016/j.cad.2010.12.015.
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
        super(TLBOBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Teaching-learning-based optimization"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.offspring_population_size = 2 * self.population_size


    def teacher_phase_reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        mean_position = [0.0] * self.problem.number_of_variables
        for j in range(0, self.population_size):
            for i in range(self.problem.number_of_variables):
                mean_position[i] += population[j].variables[i] / (self.population_size * 1.0)
        for j in range(0, self.population_size):
            # Teaching Phrase
            TF = random.randint(1, 2)  # 1 or 2 (never 3)
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0.0, 1.0) * (
                        self.g_best.variables[i] - TF * mean_position[i])
        return offsprings

    def learner_phase_reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            idx = random.choice(list(set(range(0, self.population_size)) - {j}))
            if self.comparator.compare(population[j], population[idx]) == 1:
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[idx].variables[i] - population[j].variables[i])
            else:
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[j].variables[i] - population[idx].variables[i])
        return offsprings

    def evolve(self):
        selected_solutions = self.selection(self.solutions)
        offsprings = self.teacher_phase_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        selected_solutions = self.selection(self.solutions)
        offsprings = self.learner_phase_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
