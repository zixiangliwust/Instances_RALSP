# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation, Selection
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic bee algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class BABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic bee algorithm
    References:
    [1] Çil, Zeynel Abidin, Zixiang Li, Suleyman Mete, and Eren Özceylan. 2020. "Mathematical model and bee algorithms for mixed-model assembly line balancing problem with physical human–robot collaboration."
    Applied Soft Computing 93:106394. doi: https://doi.org/10.1016/j.asoc.2020.106394.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(BABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Bee algorithm"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.num_of_elite_bees = 1
        self.num_of_selected_bees = 9
        self.neighbor_size_of_elite_bee = 5
        self.neighbor_size_of_selected_bee = 1
        self.offspring_population_size = self.num_of_elite_bees * self.neighbor_size_of_elite_bee + (
                self.num_of_selected_bees - self.num_of_elite_bees) * self.neighbor_size_of_selected_bee + \
                                         self.population_size - self.num_of_selected_bees

    def onlooker_phase(self) -> None:
        self.solutions = self.sort_population.execute(self.solutions)
        for j in range(0, self.num_of_elite_bees):
            neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in
                         range(0, self.neighbor_size_of_elite_bee)]
            neighbors = self.evaluate(neighbors)
            neighbors = self.sort_population.execute(neighbors)
            if self.comparator.compare(neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = neighbors[0]
        for j in range(self.num_of_elite_bees, self.num_of_selected_bees):
            neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in
                         range(0, self.neighbor_size_of_selected_bee)]
            neighbors = self.evaluate(neighbors)
            neighbors = self.sort_population.execute(neighbors)
            if self.comparator.compare(neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = neighbors[0]

    def scout_phase(self) -> None:
        for j in range(self.num_of_selected_bees, self.population_size):
            self.solutions[j] = self.problem.create_solution()
            self.solutions[j] = self.evaluate_solution(self.solutions[j])

    def evolve(self):
        self.onlooker_phase()
        self.scout_phase()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved bee algorithm 
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class IBA(SingleObjectiveSwarmRoot[S, R]):
    """
    Improved bee algorithm
    References:
    [1] Çil, Zeynel Abidin, Zixiang Li, Suleyman Mete, and Eren Özceylan. 2020. "Mathematical model and bee algorithms for mixed-model assembly line balancing problem with physical human–robot collaboration."
    Applied Soft Computing 93:106394. doi: https://doi.org/10.1016/j.asoc.2020.106394.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IBA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Improved bee algorithm"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.num_of_elite_bees = 1
        self.num_of_selected_bees = 9
        self.neighbor_size_of_elite_bee = 5
        self.neighbor_size_of_selected_bee = 1
        self.scout_time = 10
        self.neighbor_size = 10
        self.offspring_population_size = self.num_of_elite_bees * self.neighbor_size_of_elite_bee + (
                self.num_of_selected_bees - self.num_of_elite_bees) * self.neighbor_size_of_selected_bee

    def onlooker_phase(self) -> None:
        self.solutions = self.sort_population.execute(self.solutions)
        for j in range(0, self.num_of_elite_bees):
            for _ in range(0, self.neighbor_size_of_elite_bee):
                new_solution = self.mutation_operator.execute(self.solutions[j])
                new_solution = self.evaluate_solution(new_solution)
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) != 0:
                    new_solution.survive_time = 0
                else:
                    new_solution.survive_time = self.solutions[j].survive_time
                if self.comparator.compare(new_solution, self.solutions[j]) <= 0:
                    self.solutions[j] = new_solution
        for j in range(self.num_of_elite_bees, self.num_of_selected_bees):
            for _ in range(0, self.neighbor_size_of_selected_bee):
                new_solution = self.mutation_operator.execute(self.solutions[j])
                new_solution = self.evaluate_solution(new_solution)
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) != 0:
                    new_solution.survive_time = 0
                else:
                    new_solution.survive_time = self.solutions[j].survive_time
                if self.comparator.compare(new_solution, self.solutions[j]) <= 0:
                    self.solutions[j] = new_solution

    def scout_phase(self) -> None:
        for j in range(self.num_of_selected_bees, self.population_size):
            idx = random.randint(0, self.num_of_selected_bees - 1)
            neighbors = [self.mutation_operator.execute(self.solutions[idx]) for _ in range(0, self.neighbor_size)]
            neighbors = self.evaluate(neighbors)
            neighbors = self.sort_population.execute(neighbors)
            for solution in neighbors[0:]:
                if self.identical_solutions_comparator.compare(solution, self.solutions[idx]) != 0:
                    self.solutions[j] = solution
                    break
            self.offspring_population_size += self.neighbor_size
        for j in range(0, self.num_of_elite_bees):
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
        self.offspring_population_size = self.num_of_elite_bees * self.neighbor_size_of_elite_bee + (
                self.num_of_selected_bees - self.num_of_elite_bees) * self.neighbor_size_of_selected_bee
        for solution in self.solutions:
            solution.survive_time += 1
        self.onlooker_phase()
        self.scout_phase()

    def after_evolve(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        self.solutions = self.restart_operator.execute(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
