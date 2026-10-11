# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.operator import Selection, Crossover, Mutation
from evolu.core.problem import Problem
from evolu.util.generator import Generator
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.comparator import DominanceWithConstraintsComparator, EpsilonDominanceComparator
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.evaluator import Evaluator
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import (GreedyPopulationRankingAndDensityEstimatorReplacement,
                                          JoinPopulationRankingAndDensityEstimatorReplacement)
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Multi-objective artificial bee colony
Creator: Zixiang Li, zixiangliwust@gmail.com;
"""


class MOABC(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective artificial bee colony
    References:
    [1] Saif, Ullah, Zailin Guan, Weiqi Liu, Baoxi Wang, and Chaoyong Zhang. 2014.
    "Multi-objective artificial bee colony algorithm for simultaneous sequencing and balancing of mixed model assembly line."
    The International Journal of Advanced Manufacturing Technology 75 (9-12):1809-27.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MOABC, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective artificial bee colony"
        self.selection_operator = selection
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.scout_time = 200
        self.employed_bee_replacement_operator = GreedyPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(DominanceWithConstraintsComparator()), CrowdingDistance())
        self.onlooker_replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(DominanceWithConstraintsComparator()), CrowdingDistance())
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
    
    def employed_bee_selection(self, population: List[S]) -> List[S]:
        # For ABC algorithm, we can use the same selection mechanism
        selected_solutions = []
        for i in range(self.population_size):
            solution = self.selection_operator.execute(population)
            selected_solutions.append(solution)
        return selected_solutions

    def employed_bee_reproduction(self, population: List[S]) -> List[S]:
        # For ABC algorithm, reproduction is mutation-based
        offsprings = []
        for j in range(0, self.population_size):
            new_solution = self.mutation_operator.execute(population[j])
            if self.identical_solutions_comparator.compare(population[j], new_solution) != 0:
                new_solution.survive_time = 0
            else:
                new_solution.survive_time = population[j].survive_time
            offsprings.append(new_solution)
        return offsprings

    def employed_bee_replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.employed_bee_replacement_operator.replace(population, offsprings)

    def onlooker_selection(self, population: List[S]) -> List[S]:
        selected_solutions = []
        for j in range(0, self.population_size):
            solution = self.selection_operator.execute(population)
            selected_solutions.append(solution)
        return selected_solutions

    def onlooker_reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        for j in range(0, self.population_size):
            new_solution = self.mutation_operator.execute(population[j])
            if self.identical_solutions_comparator.compare(population[j], new_solution) != 0:
                new_solution.survive_time = 0
            else:
                new_solution.survive_time = population[j].survive_time
            offsprings.append(new_solution)
        return offsprings

    def onlooker_replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.onlooker_replacement_operator.replace(population, offsprings)

    def evolve(self):
        self.offspring_population_size = 2 * self.population_size
        for solution in self.solutions:
            solution.survive_time += 1
        selected_solutions = self.employed_bee_selection(self.solutions)
        offsprings = self.employed_bee_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.employed_bee_replacement(self.solutions, offsprings)
        selected_solutions = self.onlooker_selection(self.solutions)
        offsprings = self.onlooker_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.onlooker_replacement(self.solutions, offsprings)
        self.scout_phase()

    def scout_phase(self) -> None:
        worst_solution_index = 0
        for j in range(0, self.population_size):
            if self.solutions[worst_solution_index].survive_time < self.solutions[j].survive_time:
                worst_solution_index = j
        for j in range(0, self.population_size):
            if j == worst_solution_index and self.solutions[j].survive_time >= self.scout_time:
                self.solutions[j] = self.create_solution()
                self.solutions[j] = self.evaluate_solution(self.solutions[j])
                self.offspring_population_size += 1
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved multi-objective artificial bee colony
Creator: Zixiang Li, zixiangliwust@gmail.com;
"""


class IMOABC(MOABC[S, R]):
    """
    Improved multi-objective artificial bee colony
    References:
    [1] Li, Zixiang, Mukund Nilakantan Janardhanan, and S. G. Ponnambalam. 2021. "Cost-oriented robotic assembly line balancing problem with setup times: multi-objective algorithms."
    Journal of Intelligent Manufacturing 32 (4):989-1007. doi: 10.1007/s10845-020-01598-7.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IMOABC, self).__init__(problem=problem, 
                                     population_size=population_size,
                                     selection=selection,
                                     mutation=mutation,
                                     population_generator=population_generator,
                                     population_evaluator=population_evaluator,
                                     termination_criterion=termination_criterion)
        # Note: MOABC parent class handles operator assignments properly
        self.algorithm_name = "Improved multi-objective artificial bee colony"
        self.crossover_operator = crossover
        self.leaders_archive = NonDominatedSolutionsArchive()
        self.scout_time = 200

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            self.leaders_archive.add(solution)

    def employed_bee_reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        for j in range(0, self.population_size):
            parents = [population[j], self.selection_operator.execute(population)]
            new_solutions = self.crossover_operator.execute(parents)
            new_solution = new_solutions[0]
            if self.identical_solutions_comparator.compare(population[j], new_solution) != 0:
                new_solution.survive_time = 0
            else:
                new_solution.survive_time = population[j].survive_time
            offsprings.append(new_solution)
        return offsprings

    def scout_phase(self) -> None:
        worst_solution_index = 0
        for j in range(0, self.population_size):
            if self.solutions[worst_solution_index].survive_time < self.solutions[j].survive_time:
                worst_solution_index = j
        for j in range(0, self.population_size):
            if j == worst_solution_index and self.solutions[j].survive_time >= self.scout_time:
                self.offspring_population_size += 1
                idx = random.randint(0, len(self.leaders_archive.solution_list) - 1)
                self.solutions[j] = copy.deepcopy(self.leaders_archive.solution_list[idx])
                self.solutions[j].survive_time = 0

    def evolve(self):
        self.offspring_population_size = 2 * self.population_size
        for solution in self.solutions:
            solution.survive_time += 1
        selected_solutions = self.employed_bee_selection(self.solutions)
        offsprings = self.employed_bee_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        for solution in offsprings:
            self.leaders_archive.add(solution)
        self.solutions = self.employed_bee_replacement(self.solutions, offsprings)
        selected_solutions = self.onlooker_selection(self.solutions)
        offsprings = self.onlooker_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        for solution in offsprings:
            self.leaders_archive.add(solution)
        self.solutions = self.onlooker_replacement(self.solutions, offsprings)
        self.scout_phase()
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
