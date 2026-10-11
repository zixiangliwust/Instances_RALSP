# -*- coding: utf-8 -*-
import copy
import random
import math
import numpy as np
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
Module: Basic grey wolf optimizer
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class GWOBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic grey wolf optimizer
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] https://www.mathworks.com/matlabcentral/fileexchange/44974-grey-wolf-optimizer-gwo?s_tid=FX_rc3_behav
    [4] Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. 2014. "Grey Wolf Optimizer."
    Advances in Engineering Software 69:46-61. doi: https://doi.org/10.1016/j.advengsoft.2013.12.007.
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
        super(GWOBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Grey wolf optimizer"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.alpha_solution = None
        self.beta_solution = None
        self.delta_solution = None

    def init_progress(self) -> None:
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.solutions = self.sort_population.execute(self.solutions)
        self.alpha_solution, self.beta_solution, self.delta_solution = copy.deepcopy(self.solutions[0:3])

    def update_leaders(self):
        for j in range(0, self.population_size):
            if self.comparator.compare(self.solutions[j], self.alpha_solution) == -1:
                self.delta_solution = copy.deepcopy(self.beta_solution)
                self.beta_solution = copy.deepcopy(self.alpha_solution)
                self.alpha_solution = copy.deepcopy(self.solutions[j])
            elif self.comparator.compare(self.solutions[j], self.beta_solution) == -1:
                self.delta_solution = copy.deepcopy(self.beta_solution)
                self.beta_solution = copy.deepcopy(self.solutions[j])
            elif self.comparator.compare(self.solutions[j], self.delta_solution) == -1:
                self.delta_solution = copy.deepcopy(self.solutions[j])

    def reproduction(self, population: List[S]) -> List[S]:
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            for i in range(0, self.problem.number_of_variables):
                a1 = a * (2 * random.uniform(0.0, 1.0) - 1)
                a2 = a * (2 * random.uniform(0.0, 1.0) - 1)
                a3 = a * (2 * random.uniform(0.0, 1.0) - 1)
                c1 = 2 * random.uniform(0.0, 1.0)
                c2 = 2 * random.uniform(0.0, 1.0)
                c3 = 2 * random.uniform(0.0, 1.0)
                # Equation (3.5)-part 1
                d_alpha = abs(c1 * self.alpha_solution.variables[i] - population[j].variables[i])
                # Equation (3.6)-part 1
                x1 = self.alpha_solution.variables[i] - a1 * d_alpha
                # Equation (3.5)-part 2
                d_beta = abs(c2 * self.beta_solution.variables[i] - population[j].variables[i])
                # Equation (3.6)-part 2
                x2 = self.beta_solution.variables[i] - a2 * d_beta
                # Equation (3.5)-part 3
                d_delta = abs(c3 * self.delta_solution.variables[i] - population[j].variables[i])
                # Equation (3.5)-part 3
                x3 = self.delta_solution.variables[i] - a3 * d_delta
                # Equation (3.7)
                offsprings[j].variables[i] = (x1 + x2 + x3) / 3
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return copy.deepcopy(offsprings)

    def evolve(self):
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.update_leaders()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Random walk grey wolf optimizer
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class GWORW(GWOBase[S, R]):
    """
    Random walk grey wolf optimizer
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] Gupta, Shubham, and Kusum Deep. 2019. "A novel Random Walk Grey Wolf Optimizer."
    Swarm and Evolutionary Computation 44:101-12. doi: https://doi.org/10.1016/j.swevo.2018.01.001.
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
        super(GWORW, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Random walk grey wolf optimizer"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.alpha_solution = None
        self.beta_solution = None
        self.delta_solution = None
        self.offspring_population_size = 3 + self.population_size

    def random_walk(self):
        a = 2 - self.iterations * (2 / self.max_iterations)
        leaders = [self.alpha_solution, self.beta_solution, self.delta_solution]
        offspring_leaders = copy.deepcopy(leaders)
        for j in range(0, len(leaders)):
            for i in range(0, self.problem.number_of_variables):
                offspring_leaders[j].variables[i] = leaders[j].variables[i] + a * np.random.standard_cauchy()
        return offspring_leaders
    
    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.replacement_operator.replace(population, offsprings)

    def evolve(self):
        offspring_leaders = self.random_walk()
        offspring_leaders = self.evaluate(offspring_leaders)
        self.alpha_solution, self.beta_solution, self.delta_solution = self.replacement_operator.replace(
            [self.alpha_solution, self.beta_solution, self.delta_solution], offspring_leaders)
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.update_leaders()
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name