# -*- coding: utf-8 -*-
import copy
import random
import time
import math
from typing import List, TypeVar
import numpy
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.generator import Generator
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.util.evaluator import Evaluator, SequentialEvaluator
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic simulated annealing
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic simulated annealing
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] van Laarhoven, Peter J. M., and Emile H. L. Aarts. 1987. "Simulated annealing."
    In Simulated Annealing: Theory and Applications, edited by Peter J. M. van Laarhoven and Emile H. L. Aarts, 7-15. Dordrecht: Springer Netherlands.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(SABase, self).__init__(problem=problem)
        self.algorithm_name = "Simulated annealing"
        self.mutation_operator = mutation
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.offspring_population_size = self.sub_max_iteration
        self.solution = None

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def evolve(self) -> None:
        for _ in range(0, self.sub_max_iteration):
            new_solution: Solution = self.mutation_operator.execute(self.solution)
            new_solution = self.evaluate_solution(new_solution)
            acceptance_probability = self.compute_acceptance_probability(
                self.solution.objectives[0], new_solution.objectives[0], self.temperature
            )
            if self.comparator.compare(new_solution, self.solution) == -1:
                self.solution = new_solution
                if self.comparator.compare(self.solution, self.g_best) == -1:
                    self.g_best = copy.deepcopy(self.solution)
                    self.problem.g_best = self.g_best
            elif acceptance_probability > random.random():
                self.solution = new_solution
        self.temperature *= self.cooling_rate

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        pass

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Restarted simulated annealing
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class RSA(SABase[S, R]):
    """
    Restarted simulated annealing
    References:
    [1] Li, Zixiang, Mukund Nilakantan Janardhanan, Peter Nielsen, and Qiuhua Tang. 2018. "Mathematical models and simulated annealing algorithms for the robotic assembly line balancing problem."
    Assembly Automation 38 (4):420-36. doi: 10.1108/Aa-09-2017-115.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(RSA, self).__init__(problem=problem, mutation=mutation, comparator=comparator,
                                  termination_criterion=termination_criterion)
        self.algorithm_name = "Restarted simulated annealing"
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.temperature_restart = 0.001
        self.non_improvement_time_limit = 100
        self.non_improvement_time = 0
        self.offspring_population_size = self.sub_max_iteration

    def evolve(self) -> None:
        for _ in range(0, self.sub_max_iteration):
            new_solution: Solution = self.mutation_operator.execute(self.solution)
            new_solution = self.evaluate_solution(new_solution)
            acceptance_probability = self.compute_acceptance_probability(
                self.solution.objectives[0], new_solution.objectives[0], self.temperature)
            if self.comparator.compare(new_solution, self.solution) == -1:
                self.solution = new_solution
                if self.comparator.compare(self.solution, self.g_best) == -1:
                    self.g_best = copy.deepcopy(self.solution)
                    self.problem.g_best = self.g_best
                    self.non_improvement_time = 0
            elif acceptance_probability > random.random():
                self.solution = new_solution
        self.temperature *= self.cooling_rate

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def after_evolve(self):
        self.non_improvement_time += 1
        if self.non_improvement_time >= self.non_improvement_time_limit:
            self.non_improvement_time = 0
            if self.temperature < self.temperature_restart:
                self.temperature = self.temperature_restart

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Population simulated annealing
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SAPopulation(SingleObjectiveSwarmRoot[S, R]):
    """
    Population simulated annealing
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [3] van Laarhoven, Peter J. M., and Emile H. L. Aarts. 1987. "Simulated annealing."
    In Simulated Annealing: Theory and Applications, edited by Peter J. M. van Laarhoven and Emile H. L. Aarts, 7-15. Dordrecht: Springer Netherlands.
    """

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Input parameters:
        population_size (int): number of population size, default = 100; [2, 10000]
        """
        super(SAPopulation, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Population simulated annealing"
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.neighbor_num = 2
        self.mutation_operator = mutation
        self.comparator = comparator
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.offspring_population_size = self.sub_max_iteration * self.population_size * self.neighbor_num


    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        for j in range(0, self.population_size):
            for k in range(0, self.neighbor_num):
                new_solution = self.mutation_operator.execute(population[j])
                offsprings.append(new_solution)
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        new_solutions = []
        offsprings = self.sort_population.execute(offsprings)[:self.population_size]
        for j in range(0, self.population_size):
            acceptance_probability = self.compute_acceptance_probability(
                population[j].objectives[0], offsprings[j].objectives[0], self.temperature
            )
            if self.comparator.compare(offsprings[j], population[j]) == -1:
                new_solutions.append(offsprings[j])
            elif acceptance_probability > random.random():
                new_solutions.append(offsprings[j])
            else:
                new_solutions.append(population[j])
        return new_solutions

    def evolve(self) -> None:
        for _ in range(0, self.sub_max_iteration):
            selected_solutions = self.selection(self.solutions)
            offsprings = self.reproduction(selected_solutions)
            offsprings = self.evaluate(offsprings)
            self.solutions = self.replacement(self.solutions, offsprings)
            
            # Sort the population
            self.solutions = self.sort_population.execute(self.solutions)
            
            # Remove duplicates in the population
            for j in range(0,self.population_size - 1):
                for k in range(j + 1, self.population_size):
                    if self.comparator.compare(self.solutions[j], self.solutions[k]) == 0:
                        self.solutions[k] = self.create_solution()
                        self.solutions[k] = self.evaluate_solution(self.solutions[k])
                        self.offspring_population_size += 1
        self.temperature *= self.cooling_rate

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)
