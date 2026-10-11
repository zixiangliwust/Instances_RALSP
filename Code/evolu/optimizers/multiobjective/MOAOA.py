# -*- coding: utf-8 -*-
import random
import copy
from typing import List, Optional, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.core.solution import FloatSolution
from evolu.util.archive import BoundedArchive, NonDominatedSolutionsArchive
from evolu.util.comparator import (DominanceWithConstraintsComparator, MultiComparator,
                                         SolutionAttributeComparator,
                                         EpsilonDominanceComparator)
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacement
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Multi-objective arithmetic optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOAOA(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective arithmetic optimization algorithm
    References:
    [1] Abualigah, Laith, Ali Diabat, Seyedali Mirjalili, Mohamed Abd Elaziz, and Amir H. Gandomi. 2021. "The Arithmetic Optimization Algorithm."
    Computer Methods in Applied Mechanics and Engineering 376:113609. doi: https://doi.org/10.1016/j.cma.2020.113609.
    """
    __EPS = 1.0e-14

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 leaders_archive: Optional[BoundedArchive],
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MOAOA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective arithmetic optimization algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.leaders_archive = leaders_archive
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.comparator = MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                           SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(self.dominance_comparator), CrowdingDistance())
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
        self.alpha = 5
        self.miu = 0.5
        self.moa_min = 0.2
        self.moa_max = 0.9

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            self.leaders_archive.add(solution)
        self.leaders_archive.compute_density_estimator()

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        moa = self.moa_min + (self.iterations + 1) * ((self.moa_max - self.moa_min) / self.max_iterations)  # Eq. 2
        mop = 1 - ((self.iterations + 1) ** (1.0 / self.alpha)) / (self.max_iterations ** (1.0 / self.alpha))  # Eq. 4
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            for i in range(0, self.problem.number_of_variables):
                r1, r2, r3 = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)
                if r1 > moa:  # Exploration phase
                    if r2 < 0.5:
                        offsprings[j].variables[i] = g_best.variables[i] / (mop + self.__EPS) * \
                                                     ((self.problem.upper_bound[i] - self.problem.lower_bound[
                                                         i]) * self.miu +
                                                      self.problem.lower_bound[i])
                    else:
                        offsprings[j].variables[i] = g_best.variables[i] * mop * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * self.miu +
                                self.problem.lower_bound[i])
                else:  # Exploitation phase
                    if r3 < 0.5:
                        offsprings[j].variables[i] = g_best.variables[i] - mop * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * self.miu +
                                self.problem.lower_bound[i])
                    else:
                        offsprings[j].variables[i] = g_best.variables[i] + mop * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * self.miu +
                                self.problem.lower_bound[i])
        return offsprings


    def select_global_best(self) -> FloatSolution:
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) > 2:
            solutions = random.sample(leaders_archive, 2)
            if self.leaders_archive.comparator.compare(solutions[0], solutions[1]) < 1:
                g_best = copy.deepcopy(solutions[0])
            else:
                g_best = copy.deepcopy(solutions[1])
        else:
            g_best = copy.deepcopy(self.leaders_archive.solution_list[0])
        return g_best

    def evolve(self):
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        for solution in offsprings:
            self.leaders_archive.add(solution)
        self.leaders_archive.compute_density_estimator()
        self.solutions = self.replacement(self.solutions, offsprings)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
