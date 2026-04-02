# -*- coding: utf-8 -*-
from typing import List, TypeVar
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.config import store
from evolu.util.evaluator import Evaluator, SequentialEvaluator
from evolu.util.generator import Generator, RandomGenerator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Evolution strategies
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class ESBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Evolution strategies
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy, Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Ben√≠tez-Hidalgo <antonio.b@uma.es>
    [2] https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [3] Schwefel, H. P. 1984. "Evolution strategies: A family of non-linear optimization techniques based on imitating some principles of organic evolution."
    Annals of Operations Research 1 (2):165-7. doi: 10.1007/BF01876146.
    [4] Beyer, Hans-Georg, and Hans-Paul Schwefel. 2002. "Evolution strategies ‚Ä?A comprehensive introduction."
    Natural Computing 1 (1):3-52. doi: 10.1023/A:1015059928466.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 elitist: bool,
                 mutation: Mutation,
                 population_generator: Generator = RandomGenerator(),
                 population_evaluator: Evaluator = SequentialEvaluator(),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(ESBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Elitist evolution Strategy"
        self.offspring_population_size = offspring_population_size
        self.elitist = elitist
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

    def reproduction(self, population: List[S]) -> List[S]:
        population = self.sort_population.execute(population)
        offsprings = []
        for _ in range(0, max(1, int(self.offspring_population_size / self.population_size))):
            for solution in population:
                new_solution = self.mutation_operator.execute(solution)
                offsprings.append(new_solution)
                if len(offsprings) >= self.offspring_population_size:
                    break
            if len(offsprings) >= self.offspring_population_size:
                break
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        population_pool = []
        if self.elitist:
            population_pool = population
            population_pool.extend(offsprings)
            population_pool = self.sort_population.execute(population_pool)
            new_solutions = []
            for i in range(self.population_size):
                new_solutions.append(population_pool[i])
            return new_solutions
        else:
            population_pool.extend(offsprings)
            population_pool = self.sort_population.execute(population_pool)
            new_solutions = []
            for j in range(0, len(population_pool)):
                new_solutions.append(population_pool[j])
                if len(new_solutions) >= self.population_size:
                    break
            if len(new_solutions) < self.population_size:
                population = self.sort_population.execute(population)
                for j in range(0, len(population)):
                    new_solutions.append(population[j])
                    if len(new_solutions) >= self.population_size:
                        break
            return new_solutions

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
