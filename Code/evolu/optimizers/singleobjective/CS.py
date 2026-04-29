# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.core.operator import Mutation
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.operator.replacement import JoinPopulationSelectionReplacement
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic cuckoo search algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class CSBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic cuckoo search algorithm
    References:
    [1] Li, Zixiang, Nilanjan Dey, Amira S. Ashour, and Qiuhua Tang. 2018. "Discrete cuckoo search algorithms for two-sided robotic assembly line balancing problem."
    Neural Computing & Applications 30 (9):2685-96. doi: 10.1007/s00521-017-2855-5.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(CSBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Cuckoo search algorithm"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.abandon_rate = 0.2

    def selection(self, population: List[S]) -> List[S]:
        selected_solutions = []
        for _ in range(0, self.population_size):
            idx = random.randint(0, self.population_size - 1)
            selected_solutions.append(population[idx])
        return selected_solutions

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        for j in range(0, self.population_size):
            new_solution = self.mutation_operator.execute(population[j])
            offsprings.append(new_solution)
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        result_list = population
        for j in range(0, self.population_size):
            idx = random.randint(0, self.population_size - 1)
            if self.comparator.compare(offsprings[j], result_list[idx]) == -1:
                result_list[idx] = offsprings[j]
        return result_list

    def abandon_phase(self) -> None:
        self.solutions = self.sort_population.execute(self.solutions)
        for j in range(self.population_size - int(self.abandon_rate * self.population_size + 1), self.population_size):
            self.solutions[j] = self.problem.create_solution()
            self.solutions[j] = self.evaluate_solution(self.solutions[j])

    def evolve(self) -> None:
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.abandon_phase()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Discrete cuckoo search algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class DCS(CSBase[S, R]):
    """
    Discrete cuckoo search algorithm
    References:
    [1] Li, Zixiang, Nilanjan Dey, Amira S. Ashour, and Qiuhua Tang. 2018. "Discrete cuckoo search algorithms for two-sided robotic assembly line balancing problem."
    Neural Computing & Applications 30 (9):2685-96. doi: 10.1007/s00521-017-2855-5.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(DCS, self).__init__(problem=problem, population_size=population_size, mutation=mutation,
                                  population_generator=population_generator, population_evaluator=population_evaluator,
                                  termination_criterion=termination_criterion)
        self.algorithm_name = "Discrete cuckoo search algorithm"
        self.abandon_rate = 0.2

    def abandon_phase(self) -> None:
        self.solutions = self.sort_population.execute(self.solutions)
        for j in range(self.population_size - int(self.abandon_rate * self.population_size + 1), self.population_size):
            idx = random.randint(0, self.population_size - int(self.abandon_rate * self.population_size + 1) - 1)
            self.solutions[j] = self.mutation_operator.execute(self.solutions[idx])
            self.solutions[j] = self.evaluate_solution(self.solutions[j])

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved cuckoo search algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class ICS(CSBase[S, R]):
    """
    Improved cuckoo search algorithm
    References:
    [1] Li, Zixiang, and Mukund Nilakantan Janardhanan. 2021. "Modelling and solving profit-oriented U-shaped partial disassembly line balancing problem."
    Expert Systems with Applications 183:115431. doi: https://doi.org/10.1016/j.eswa.2021.115431.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(ICS, self).__init__(problem=problem, population_size=population_size, mutation=mutation,
                                  population_generator=population_generator, population_evaluator=population_evaluator,
                                  termination_criterion=termination_criterion)
        self.algorithm_name = "Improved cuckoo search algorithm"
        self.abandon_rate = 0.2
        self.neighbor_size = 10
        self.local_search_probability = 0.1
        self.scout_time = 10
        self.replacement_operator = JoinPopulationSelectionReplacement(self.comparator)

    def selection(self, population: List[S]) -> List[S]:
        selected_solutions = []
        for _ in range(0, self.population_size):
            idx = random.randint(0, self.population_size - 1)
            selected_solutions.append(population[idx])
        return selected_solutions

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        for j in range(0, self.population_size):
            new_solution = self.mutation_operator.execute(population[j])
            if self.identical_solutions_comparator.compare(population[j], new_solution) != 0:
                new_solution.survive_time = 0
            else:
                new_solution.survive_time = population[j].survive_time
            offsprings.append(new_solution)
        return offsprings

    def abandon_phase(self) -> None:
        self.solutions = self.sort_population.execute(self.solutions)
        for j in range(self.population_size - int(self.abandon_rate * self.population_size + 1), self.population_size):
            idx = random.randint(0, self.population_size - int(self.abandon_rate * self.population_size + 1) - 1)
            neighbors = [self.mutation_operator.execute(self.solutions[idx]) for _ in range(0, self.neighbor_size)]
            neighbors = self.evaluate(neighbors)
            neighbors = self.sort_population.execute(neighbors)
            for solution in neighbors[0:]:
                if self.identical_solutions_comparator.compare(solution, self.solutions[idx]) != 0:
                    self.solutions[j] = solution
                    break
            self.offspring_population_size += self.neighbor_size
        for j in range(0, self.population_size - int(self.abandon_rate * self.population_size + 1)):
            if self.solutions[j].survive_time >= self.scout_time:
                neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in range(0, self.neighbor_size)]
                neighbors = self.evaluate(neighbors)
                neighbors = self.sort_population.execute(neighbors)
                for solution in neighbors[0:]:
                    if self.identical_solutions_comparator.compare(solution, self.solutions[j]) != 0:
                        self.solutions[j] = solution
                        break
                self.offspring_population_size += self.neighbor_size

    def local_search(self) -> None:
        for j in range(0, self.population_size):
            if random.random() < self.local_search_probability:
                for i in range(0, self.problem.number_of_variables):
                    new_solution = self.mutation_operator.execute(self.solutions[j])
                    if self.identical_solutions_comparator.compare(self.solutions[j], new_solution) != 0:
                        new_solution.survive_time = 0
                    else:
                        new_solution.survive_time = self.solutions[j].survive_time
                    new_solution = self.evaluate_solution(new_solution)
                    self.offspring_population_size += 1
                    if self.comparator.compare(new_solution, self.solutions[j]) == -1:
                        self.solutions[j] = copy.deepcopy(new_solution)

    def evolve(self) -> None:
        self.offspring_population_size = self.population_size
        for solution in self.solutions:
            solution.survive_time += 1
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.abandon_phase()
        self.local_search()

    def after_evolve(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        self.solutions = self.restart_operator.execute(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
