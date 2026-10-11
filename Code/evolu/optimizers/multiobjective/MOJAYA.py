# -*- coding: utf-8 -*-
import random
import copy
from typing import List, Optional, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.core.solution import FloatSolution
from evolu.operator.selection import WorstSolutionSelection, RankingAndDensityEstimatorSelection
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
Module: Multi-objective Jaya algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOJAYA(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective Jaya algorithm
    References:
    [1] Venkata Rao, R. 2016. "Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems."
    International Journal of Industrial Engineering Computations 7:19-34. doi: 10.5267/j.ijiec.2015.8.004.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 leaders_archive: Optional[BoundedArchive],
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Input parameters:
        population_size (int): number of population size, default = 100; [2, 10000]
        """
        super(MOJAYA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective Jaya algorithm"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.leaders_archive = leaders_archive
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.comparator = MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                           SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
        self.worst_solution_selection = WorstSolutionSelection(comparator=self.comparator)
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
        self.solutions = RankingAndDensityEstimatorSelection(len(self.solutions)).execute(self.solutions)
        self.p_worst = copy.deepcopy(self.worst_solution_selection.execute(self.solutions)[0])

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + \
                                      random.uniform(0, 1) * (
                                              g_best.variables[i] - abs(population[j].variables[i])) - \
                                      random.uniform(0, 1) * (
                                              self.p_worst.variables[i] - abs(population[j].variables[i]))
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

    def after_initialization(self):
        self.solutions = RankingAndDensityEstimatorSelection(len(self.solutions)).execute(self.solutions)
        self.p_worst = copy.deepcopy(self.worst_solution_selection.execute(self.solutions))

    def after_evolve(self):
        self.solutions = RankingAndDensityEstimatorSelection(len(self.solutions)).execute(self.solutions)
        self.p_worst = copy.deepcopy(self.worst_solution_selection.execute(self.solutions))
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
