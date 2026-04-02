# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy.stats import cauchy
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Crossover, Selection
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic differential evolution
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class DEBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic differential evolution
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Storn, Rainer, and Kenneth Price. 1997. "Differential Evolution â€?A Simple and Efficient Heuristic for global Optimization over Continuous Spaces."
    Journal of Global Optimization 11 (4):341-59. doi: 10.1023/A:1008202821328.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 crossover: Crossover,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Input parameters:
        population_size (int): number of population size, default = 100; [2, 10000]
        """
        super(DEBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Differential evolution"
        self.selection_operator = selection
        self.crossover_operator = crossover
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def selection(self, population: List[S]) -> List[S]:
        mating_pool = []
        for i in range(0,self.offspring_population_size):
            self.selection_operator.set_index_to_exclude(i)
            selected_solution = self.selection_operator.execute(self.solutions)
            # If the selection operator returns a list, extend mating_pool with it
            if isinstance(selected_solution, list):
                mating_pool.extend(selected_solution)
            else:
                # Otherwise, it returns a single solution, add it directly
                mating_pool.append(selected_solution)
        return mating_pool

    def reproduction(self, mating_pool: List[S]) -> List[S]:
        number_of_parents = self.crossover_operator.get_number_of_parents()        
        offsprings = []
        for i in range(0,self.offspring_population_size):
            parents = []
            for j in range(number_of_parents):
                parents.append(population[i*number_of_parents + j])
            self.crossover_operator.current_individual = self.solutions[i]
            new_solutions = self.crossover_operator.execute(parents)
            for solution in new_solutions:
                offsprings.append(solution)
                if len(offsprings) >= self.offspring_population_size:
                    break
        return offsprings

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
