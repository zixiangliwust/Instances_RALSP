import copy
from functools import cmp_to_key
from typing import List, TypeVar

from evolu.config import store
from evolu.core.operator import Crossover, Mutation, Selection
from evolu.core.problem import Problem
from evolu.operator.selection import BinaryTournamentSelection
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.util.archive import NonDominatedSolutionsArchive, BoundedArchive
from evolu.util.comparator import (Comparator, MultiComparator, SolutionAttributeComparator,
                                         EpsilonDominanceComparator)
from evolu.util.density_estimator import CrowdingDistance, DensityEstimator
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.neighborhood import Neighborhood
from evolu.util.ranking import FastNonDominatedRanking, Ranking
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
module:: MOCell
synopsis: MOCell (Multi-Objective Cellular evolutionary algorithm) implementation
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class MOCell(MultiObjectiveSwarmRoot[S, R]):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 neighborhood: Neighborhood,
                 archive: BoundedArchive,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                      SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
                 ),
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        MOCEll implementation as described in:
        param problem: The problem to solve.
        param population_size: Size of the population.
        param mutation: Mutation operator (see :py:mod:`evolu.operator.mutation`).
        param crossover: Crossover operator (see :py:mod:`evolu.operator.crossover`).
        param selection: Selection operator (see :py:mod:`evolu.operator.selection`).
        """
        super(MOCell, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "MOCell"
        self.offspring_population_size = 1
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.population_evaluator = population_evaluator
        self.population_generator = population_generator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.dominance_comparator = dominance_comparator
        self.neighborhood = neighborhood
        self.archive = archive
        self.current_individual = 0
        self.current_neighbors = []
        self.comparator = MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                           SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))

    def init_progress(self) -> None:
        super().init_progress()
        for solution in self.solutions:
            self.archive.add(copy.copy(solution))

    def update_progress(self) -> None:
        super().update_progress()
        self.current_individual = (self.current_individual + 1) % self.population_size

    def selection(self, population: List[S]):
        parents = []
        self.current_neighbors = self.neighborhood.get_neighbors(self.current_individual, population)
        self.current_neighbors.append(self.solutions[self.current_individual])
        parents.append(self.selection_operator.execute(self.current_neighbors))
        if len(self.archive.solution_list) > 0:
            parents.append(self.selection_operator.execute(self.archive.solution_list))
        else:
            parents.append(self.selection_operator.execute(self.current_neighbors))
        return parents

    def reproduction(self, population: List[S]) -> List[S]:
        from evolu.core.exceptions import InvalidParentsException
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()
        if len(population) % number_of_parents_to_combine != 0:
            raise InvalidParentsException(
                f"Wrong number of parents: {len(population)} is not divisible by {number_of_parents_to_combine}"
            )
        offsprings = self.crossover_operator.execute(population)
        offsprings[0] = self.mutation_operator.execute(offsprings[0])
        return [offsprings[0]]

    def replacement(self, population: List[S], offsprings: List[S]) -> List[List[S]]:
        result = self.dominance_comparator.compare(population[self.current_individual], offsprings[0])
        if result == 1:  # the offspring individual dominates the current one
            population[self.current_individual] = offsprings[0]
            self.archive.add(copy.deepcopy(offsprings[0]))
        elif result == 0:  # the offspring and current individuals are non-dominated
            new_individual = offsprings[0]
            self.current_neighbors.append(new_individual)
            ranking: Ranking = FastNonDominatedRanking()
            ranking.compute_ranking(self.current_neighbors)
            density_estimator: DensityEstimator = CrowdingDistance()
            for i in range(ranking.get_number_of_sub_fronts()):
                density_estimator.compute_density_estimator(ranking.get_sub_front(i))
            self.current_neighbors.sort(key=cmp_to_key(self.comparator.compare))
            worst_solution = self.current_neighbors[-1]
            self.archive.add(copy.deepcopy(new_individual))
            if worst_solution != new_individual:
                # if worst_solution.objectives != new_individual.objectives:
                population[self.current_individual] = new_individual
        return population

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
