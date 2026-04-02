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
Module: Multi-objective sine cosine algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOSCA(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective sine cosine algorithm
    References:
    [1] Mirjalili, Seyedali. 2016. "SCA: A Sine Cosine Algorithm for solving optimization problems."
    Knowledge-Based Systems 96:120-33. doi: https://doi.org/10.1016/j.knosys.2015.12.022.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 leaders_archive: Optional[BoundedArchive],
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MOSCA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective sine cosine algorithm"
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
        a = 2.0
        r1 = a - (self.iterations + 1) * a / float(self.max_iterations)
        g_best = self.select_global_best()
        for j in range(0, self.population_size):
            # Eq 3.4, r1 decreases linearly from a to 0
            for i in range(self.problem.number_of_variables):
                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = 2 * math.pi * random.random()
                r3 = 2 * random.random()
                r4 = random.random()
                # Eq. 3.3, 3.1 and 3.2
                if r4 < 0.5:
                    offsprings[j].variables[i] = population[j].variables[i] + r1 * math.sin(r2) * abs(
                        r3 * g_best.variables[i] - population[j].variables[i])
                else:
                    offsprings[j].variables[i] = population[j].variables[i] + r1 * math.cos(r2) * abs(
                        r3 * g_best.variables[i] - population[j].variables[i])
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
