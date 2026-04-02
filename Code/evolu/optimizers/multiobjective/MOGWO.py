# -*- coding: utf-8 -*-
import random
import copy
from typing import List, Optional, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.operator.selection import RankingAndDensityEstimatorSelection
from evolu.util.archive import BoundedArchive, NonDominatedSolutionsArchive
from evolu.util.comparator import (DominanceWithConstraintsComparator, MultiComparator,
                                         SolutionAttributeComparator,
                                         EpsilonDominanceComparator)
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacement
from evolu.util.sort_population import SortPopulation
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Multi-objective grey wolf optimizer
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOGWO(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective grey wolf optimizer
    References:
    [1] Mirjalili, Seyedali, Seyed Mohammad Mirjalili, and Andrew Lewis. 2014. "Grey Wolf Optimizer."
    Advances in Engineering Software 69:46-61. doi: https://doi.org/10.1016/j.advengsoft.2013.12.007.
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
        super(MOGWO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective grey wolf optimizer"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.leaders_archive = leaders_archive
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
        self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
        self.solutions = self.sort_population.execute(self.solutions)
        self.alpha_solution, self.beta_solution, self.delta_solution = copy.deepcopy(self.solutions[0:3])

    def update_leaders(self):
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) >= 3:
            solutions = random.sample(leaders_archive, 3)
            self.alpha_solution = copy.deepcopy(solutions[0])
            self.beta_solution = copy.deepcopy(solutions[1])
            self.delta_solution = copy.deepcopy(solutions[2])
        elif len(leaders_archive) >= 2:
            self.alpha_solution = copy.deepcopy(leaders_archive[0])
            self.beta_solution = copy.deepcopy(leaders_archive[1])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            self.delta_solution = copy.deepcopy(self.solutions[0])
        else:
            self.alpha_solution = copy.deepcopy(leaders_archive[0])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            self.beta_solution = copy.deepcopy(self.solutions[0])
            self.delta_solution = copy.deepcopy(self.solutions[1])


    def reproduction(self, population: List[S]) -> List[S]:
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            a1 = a * (2 * random.uniform(0, 1) - 1)
            a2 = a * (2 * random.uniform(0, 1) - 1)
            a3 = a * (2 * random.uniform(0, 1) - 1)
            c1 = 2 * random.uniform(0, 1)
            c2 = 2 * random.uniform(0, 1)
            c3 = 2 * random.uniform(0, 1)
            for i in range(self.problem.number_of_variables):
                d_alpha_i = abs(c1 * self.alpha_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 1
                x1_i = self.alpha_solution.variables[i] - a1 * d_alpha_i  # Equation (3.6)-part 1
                d_beta_i = abs(c2 * self.beta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 2
                x2_i = self.beta_solution.variables[i] - a2 * d_beta_i  # Equation (3.6)-part 2
                d_delta_i = abs(c3 * self.delta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 3
                x3_i = self.delta_solution.variables[i] - a3 * d_delta_i  # Equation (3.5)-part 3
                offsprings[j].variables[i] = (x1_i + x2_i + x3_i) / 3  # Equation (3.7)
        # Calculate objective function for each search solution
        return offsprings

    def evolve(self):
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        for solution in offsprings:
            self.leaders_archive.add(solution)
        self.leaders_archive.compute_density_estimator()
        self.solutions = self.replacement(self.solutions, offsprings)
        self.update_leaders()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
