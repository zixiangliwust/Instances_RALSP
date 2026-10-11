# -*- coding: utf-8 -*-
import copy
import random
import numpy
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic fruit fly optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class FOABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic fruit fly optimization algorithm
    References:
    [1] Pan, Wen-Tsao. 2012. "A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example."
    Knowledge-Based Systems 26:69-74. doi: https://doi.org/10.1016/j.knosys.2011.07.001.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(FOABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Fruit fly optimization algorithm"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.para_k = 5
        self.offspring_population_size = population_size * self.para_k
        self.leader_neighbors = []

    def olfactory_phase(self):
        self.leader_neighbors = [[] for _ in range(self.population_size)]
        for j in range(self.population_size):
            for k in range(self.para_k):
                new_solution = self.mutation_operator.execute(self.solutions[j])
                new_solution = self.evaluate_solution(new_solution)
                self.leader_neighbors[j].append(new_solution)

    def visual_phase(self):
        for j in range(self.population_size):
            self.leader_neighbors[j] = self.sort_population.execute(self.leader_neighbors[j])
            if self.comparator.compare(self.leader_neighbors[j][0], self.solutions[j]) == -1:
                self.solutions[j] = self.leader_neighbors[j][0]
            self.leader_neighbors[j].clear()

    def evolve(self):
        self.olfactory_phase()
        self.visual_phase()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved friuit fly optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class IFOA(SingleObjectiveSwarmRoot[S, R]):
    """
    Improved friuit fly optimization
    References:
    [1] Pan, Wen-Tsao. 2012. "A new Fruit Fly Optimization Algorithm: Taking the financial distress model as an example."
    Knowledge-Based Systems 26:69-74. doi: https://doi.org/10.1016/j.knosys.2011.07.001.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IFOA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Improved fruit fly optimization"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.para_k = 5
        self.scout_time = 10
        self.neighbor_size = 10
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.offspring_population_size = population_size * self.para_k
        self.leader_neighbors = []

    def olfactory_phase(self):
        self.leader_neighbors = [[] for _ in range(self.population_size)]
        for j in range(self.population_size):
            neighbor = self.solutions[j]
            for k in range(self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(self.problem.number_of_objectives):
                        new_solution.objectives[i] = float('inf')
                else:
                    new_solution.survive_time = 0
                self.leader_neighbors[j].append(new_solution)

    def visual_phase(self):
        for j in range(self.population_size):
            self.leader_neighbors[j] = self.sort_population.execute(self.leader_neighbors[j])
            acceptance_probability = self.compute_acceptance_probability(
                self.solutions[j].objectives[0], self.leader_neighbors[j][0].objectives[0], self.temperature
            )
            if self.comparator.compare(self.leader_neighbors[j][0], self.solutions[j]) == -1:
                self.solutions[j] = self.leader_neighbors[j][0]
            else:
                if acceptance_probability > random.random():
                    self.solutions[j] = self.leader_neighbors[j][0]
            self.leader_neighbors[j].clear()

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def scout_phase(self) -> None:
        for j in range(0, self.population_size):
            if self.solutions[j].survive_time >= self.scout_time:
                neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in range(0, self.neighbor_size)]
                neighbors = self.evaluate(neighbors)
                neighbors = self.sort_population.execute(neighbors)
                for solution in neighbors[0:]:
                    if self.identical_solutions_comparator.compare(solution, self.solutions[j]) != 0:
                        self.solutions[j] = solution
                        break
                self.offspring_population_size += self.neighbor_size

    def evolve(self):
        self.offspring_population_size = self.population_size * self.para_k
        for j in range(self.population_size):
            self.solutions[j].survive_time += 1
        self.olfactory_phase()
        self.visual_phase()
        self.scout_phase()
        self.temperature *= self.cooling_rate

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
