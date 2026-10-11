# -*- coding: utf-8 -*-
import random
import copy
import math
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
Module: Multi-objective whale optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOWOA(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective whale optimization algorithm
    References:
    [1] Mirjalili, Seyedali, and Andrew Lewis. 2016. "The Whale Optimization Algorithm."
    Advances in Engineering Software 95:51-67. doi: https://doi.org/10.1016/j.advengsoft.2016.01.008.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 leaders_archive: Optional[BoundedArchive],
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MOWOA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective whale optimization algorithm"
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
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            r1 = random.uniform(0, 1)  # r1 is a random number in [0,1]
            r2 = random.uniform(0, 1)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if random.uniform(0, 1) < 0.5:
                if abs(a) < 1:
                    for i in range(self.problem.number_of_variables):
                        d_leader_i = abs(c * g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = g_best.variables[i] - a * d_leader_i
                elif abs(a) >= 1:
                    idx = random.randint(0, self.population_size - 1)
                    for i in range(self.problem.number_of_variables):
                        d_idx_i = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx_i
            else:
                l = - 2 * random.uniform(0, 1) + 1  # parameters in Eq. (2.5)
                for i in range(self.problem.number_of_variables):
                    d_leader2_i = abs(g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2_i * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + g_best.variables[i]
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
