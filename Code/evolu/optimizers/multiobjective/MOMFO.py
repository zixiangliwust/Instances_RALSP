# -*- coding: utf-8 -*-
import random
import copy
import math
from typing import List, Optional, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.operator.selection import RankingAndDensityEstimatorSelection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.comparator import (DominanceWithConstraintsComparator, MultiComparator,
                                         SolutionAttributeComparator,
                                         EpsilonDominanceComparator)
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacement
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.sort_population import SortPopulation
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Multi-objective moth-flame optimization algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOMFO(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective moth-flame optimization algorithm
    References:
    [1] Mirjalili, Seyedali. 2015. "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm."
    Knowledge-Based Systems 89:228-49. doi: https://doi.org/10.1016/j.knosys.2015.07.006.
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
        super(MOMFO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective moth-flame optimization algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.comparator = MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                           SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
        self.sort_population = SortPopulation(comparator=self.comparator)
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(self.dominance_comparator), CrowdingDistance())
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))


    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = copy.deepcopy(self.solutions)
        self.local_best_solutions = RankingAndDensityEstimatorSelection(len(self.local_best_solutions)).execute(
            self.local_best_solutions)
        self.local_best_solutions = self.sort_population.execute(self.local_best_solutions)

    def reproduction(self, population: List[S]) -> List[S]:
        self.local_best_solutions = self.sort_population.execute(self.local_best_solutions)
        offsprings = copy.deepcopy(population)
        # Number of flames Eq.(3.14) in the paper (linearly decreased)
        num_flame = round(
            self.population_size - (self.iterations + 1) * ((self.population_size - 1) / self.max_iterations))
        # a linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + (self.iterations + 1) * ((-1) / self.max_iterations)
        for j in range(0, self.population_size):
            for i in range(self.problem.number_of_variables):
                #   D in Eq.(3.13)
                distance_to_flame = abs(self.local_best_solutions[j].variables[i] - self.solutions[j].variables[i])
                t = (a - 1) * random.uniform(0, 1) + 1
                b = 1
                if j <= num_flame:  # Update the variables of the moth with respect to its corresponding flame
                    # Eq.(3.12)
                    offsprings[j].variables[i] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + \
                                                 self.local_best_solutions[j].variables[i]
                else:  # Update the variables of the moth with respect to one flame
                    # Eq.(3.12).
                    offsprings[j].variables[i] = distance_to_flame * math.exp(b * t) * math.cos(t * 2 * math.pi) + \
                                                 self.local_best_solutions[num_flame].variables[i]
        return offsprings

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
