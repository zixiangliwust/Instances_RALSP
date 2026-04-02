# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Crossover, Mutation, Selection
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.operator.replacement import JoinPopulationSelectionReplacement
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic memetic algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic memetic algorithm
    References:
    [1] Moscato, Pablo. 1989. "On evolution, search, optimization, genetic algorithms and martial arts: Towards memetic algorithms."
    Caltech concurrent computation program, C3P Report 826:1989.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Memetic algorithm"
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.mating_pool_size = (
                self.offspring_population_size
                * self.crossover_operator.get_number_of_parents()
                // self.crossover_operator.get_number_of_children()
        )
        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()
        self.replacement_operator = JoinPopulationSelectionReplacement(self.comparator)
        self.neighbor_size = 10

    def selection(self, population: List[S]):
        selected_solutions = []
        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            selected_solutions.append(solution)
        return selected_solutions

    def reproduction(self, population: List[S]) -> List[S]:
        from evolu.core.exceptions import InvalidParentsException
        number_of_parents = self.crossover_operator.get_number_of_parents()
        if len(population) % number_of_parents != 0:
            raise InvalidParentsException(
                f"Wrong number of parents: {len(population)} is not divisible by {number_of_parents}"
            )
        offsprings = []
        for j in range(0, self.offspring_population_size, number_of_parents):
            parents = []
            for k in range(number_of_parents):
                parents.append(population[j + k])
            new_solutions = self.crossover_operator.execute(parents)
            for solution in new_solutions:
                new_solution = self.mutation_operator.execute(solution)
                offsprings.append(new_solution)
                if len(offsprings) >= self.offspring_population_size:
                    break
        return offsprings

    def local_search(self) -> None:
        for j in range(0, self.population_size):
            neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in range(0, self.neighbor_size)]
            neighbors = self.evaluate(neighbors)
            neighbors = self.sort_population.execute(neighbors)
            self.offspring_population_size = self.offspring_population_size + self.neighbor_size
            if self.comparator.compare(neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = neighbors[0]

    def evolve(self) -> None:
        self.offspring_population_size = self.population_size
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.local_search()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
