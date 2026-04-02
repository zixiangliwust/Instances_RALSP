# -*- coding: utf-8 -*-
import copy
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.util.generator import Generator
from evolu.util.archive import NonDominatedSolutionsArchive, ModifiedNonDominatedSolutionsArchive
from evolu.util.comparator import DominanceWithConstraintsComparator, EpsilonDominanceComparator
from evolu.util.density_estimator import CrowdingDistance, ModifiedCrowdingDistance
from evolu.util.evaluator import Evaluator
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacementRemoveDuplicatedSolution
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Multi-objective migrating bird optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOMBO(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective migrating bird optimization
    References:
    [1] Zixiang Li, Mukund Nilakantan Janardhanan, Qiuhua Tang. 2021.
    "Multi-objective migrating bird optimization algorithm for cost-oriented assembly line balancing problem with collaborative robots."
    Neural Computing and Applications, 33(14):8575-8596.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MOMBO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Multi-objective migrating birds optimization"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.comparator = DominanceWithConstraintsComparator()
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacementRemoveDuplicatedSolution(
            FastNonDominatedRanking(DominanceWithConstraintsComparator()), CrowdingDistance())
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
        self.leaders_archive = ModifiedNonDominatedSolutionsArchive()
        self.density_estimator = ModifiedCrowdingDistance()
        self.scout_time = 200
        self.offsprings = []
        self.para_k = 20  # Number of neighbors to generate
        self.para_x = 10  # Number of neighbors to generate for followers

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            self.leaders_archive.add(solution)

    def leader_improvement(self) -> None:
        neighbor = self.solutions[0]
        for k in range(0, self.para_k):
            new_solution = self.mutation_operator.execute(neighbor)
            new_solution = self.evaluate_solution(new_solution)
            if self.comparator.compare(new_solution, neighbor) <= 0:
                neighbor = new_solution
            if self.identical_solutions_comparator.compare(new_solution, self.solutions[0]) == 0:
                for i in range(0, self.problem.number_of_objectives):
                    new_solution.objectives[i] = float("inf")
            else:
                new_solution.survive_time = 0
            self.offsprings.append(new_solution)

    def population_improvement(self) -> None:
        for j in range(1, self.population_size, 2):
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                else:
                    new_solution.survive_time = 0
                self.offsprings.append(new_solution)
        for j in range(2, self.population_size, 2):
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                else:
                    new_solution.survive_time = 0
                self.offsprings.append(new_solution)

    def scout_phase(self) -> None:
        worst_solution_index = 0
        for j in range(0, self.population_size):
            if self.solutions[worst_solution_index].survive_time < self.solutions[j].survive_time:
                worst_solution_index = j
        for j in range(0, self.population_size):
            if j == worst_solution_index and self.solutions[j].survive_time >= self.scout_time:
                self.offspring_population_size += 1
                self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
                sorted_population = self.leaders_archive.solution_list
                self.density_estimator.sort(sorted_population)
                self.solutions[j] = copy.deepcopy(sorted_population[0])
                sorted_population[0].survive_time += 1
                self.solutions[j].survive_time = 0

    def evolve(self):
        self.offspring_population_size = 0
        self.offspring_population_size = self.para_k + (self.para_k - self.para_x) * (self.population_size - 1)
        for j in range(0, self.population_size):
            self.solutions[j].survive_time += 1
        self.leader_improvement()
        self.population_improvement()
        for solution in self.offsprings:
            self.leaders_archive.add(solution)
        self.solutions = self.replacement(self.solutions, self.offsprings)
        self.offsprings = []
        self.scout_phase()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
