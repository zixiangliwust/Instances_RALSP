# -*- coding: utf-8 -*-
import copy
import random
import numpy as np
import math
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
Module: Basic naked mole-rat algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class NMRABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic naked mole-rat algorithm
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/rohitsalgotra/Naked-Mole-Rat-Algorithm
    [3] Salgotra, Rohit, and Urvinder Singh. 2019. "The naked mole-rat algorithm."
    Neural Computing and Applications 31 (12):8837-57. doi: 10.1007/s00521-019-04464-7.
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
        pb (float): probability of breeding, default = 0.75; (0, 1.0)
        """
        super(NMRABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Naked mole-rat algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.breeder_size = int(self.population_size / 5)
        self.pb = 0.5


    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(0, self.breeder_size):
            if random.uniform(0.0, 1.0) > self.pb:
                alpha = random.uniform(0.0, 1.0)
                for i in range(0, self.problem.number_of_variables):
                    offsprings[j].variables[i] = (1 - alpha) * population[j].variables[i] + alpha * (
                            self.g_best.variables[i] - population[j].variables[i])
        for j in range(self.breeder_size, self.population_size):
            idx1 = random.randint(self.breeder_size, self.population_size - 1)
            idx2 = random.randint(self.breeder_size, self.population_size - 1)
            while idx1 == idx2:
                idx1 = random.randint(self.breeder_size, self.population_size - 1)
                idx2 = random.randint(self.breeder_size, self.population_size - 1)
            alpha = random.uniform(0.0, 1.0)
            for i in range(0, self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + alpha * (
                        population[idx1].variables[i] - population[idx2].variables[i])

        return offsprings

    def evolve(self) -> None:
        self.solutions = self.sort_population.execute(self.solutions)
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name