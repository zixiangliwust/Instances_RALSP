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
Module: Multi-objective salp swarm optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOSSO(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective salp swarm optimization
    References:
    [1] Mirjalili, Seyedali, Amir H. Gandomi, Seyedeh Zahra Mirjalili, Shahrzad Saremi, Hossam Faris, and Seyed Mohammad Mirjalili. 2017.
    "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems."  Advances in Engineering Software 114:163-91.
    doi: https://doi.org/10.1016/j.advengsoft.2017.07.002.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 leaders_archive: Optional[BoundedArchive],
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MOSSO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective salp swarm optimization"
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
        # Eq. (3.2) in the paper
        c1 = 2 * math.exp(-((4 * (self.iterations + 1) / self.max_iterations) ** 2))
        g_best = self.select_global_best()
        for j in range(0, self.population_size):
            if j < self.population_size / 2:
                for i in range(0, self.problem.number_of_variables):
                    c2 = random.random()
                    if random.random() < 0.5:
                        offsprings[j].variables[i] = g_best.variables[i] + c1 * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * c2 +
                                self.problem.lower_bound[i])
                    else:
                        offsprings[j].variables[i] = g_best.variables[i] - c1 * (
                                (self.problem.upper_bound[i] - self.problem.lower_bound[i]) * c2 +
                                self.problem.lower_bound[i])
            else:
                # Eq. (3.4) in the paper
                for i in range(0, self.problem.number_of_variables):
                    offsprings[j].variables[i] = (self.solutions[j].variables[i] + self.solutions[j - 1].variables[
                        i]) / 2
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
