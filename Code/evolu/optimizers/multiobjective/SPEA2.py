from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.operator import Crossover, Mutation
from evolu.core.problem import Problem
from evolu.operator.selection import BinaryTournamentSelection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.comparator import Comparator, MultiComparator, DominanceWithConstraintsComparator, \
    EpsilonDominanceComparator
from evolu.util.density_estimator import KNearestNeighborDensityEstimator
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.ranking import StrengthRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacement, RemovalPolicyType
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
module:: SPEA2
synopsis: SPEA2  implementation. Note that we do not follow the structure of the original SPEA2 code. We consider
SPEA2 as a genetic algorithm with binary tournament selection, with a comparator based on the strength fitness and 
the KNN distance, and a sequential replacement strategy based in iteratively (sequentially) 
removing the worst solution of the population + offspring population. The worst solutions is selected again 
considering the strength fitness and KNN distance. Note that the implementation is exactly the same of NSGA-II, 
but using the fast nondominated sorting and the crowding distance density estimator, and the replacement follows a 
one-shot scheme (once the solutions are ordered, the best ones are selected without recomputing the ranking and
density estimator).
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class SPEA2(MultiObjectiveSwarmRoot[S, R]):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param mutation: Mutation operator (see :py:mod:`evolu.operator.mutation`).
        param crossover: Crossover operator (see :py:mod:`evolu.operator.crossover`).
        """
        multi_comparator = MultiComparator(
            [StrengthRanking.get_comparator(), KNearestNeighborDensityEstimator.get_comparator()]
        )
        selection = BinaryTournamentSelection(comparator=multi_comparator)
        super(SPEA2, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "SPEA2"
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.termination_criterion = termination_criterion
        self.population_evaluator = population_evaluator
        self.population_generator = population_generator
        self.observable.register(termination_criterion)        
        self.dominance_comparator = dominance_comparator
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            StrengthRanking(self.dominance_comparator),
            KNearestNeighborDensityEstimator(),
            RemovalPolicyType.SEQUENTIAL)
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
        self.mating_pool_size = (
                self.offspring_population_size
                * self.crossover_operator.get_number_of_parents()
                // self.crossover_operator.get_number_of_children()
        )
        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def selection(self, population: List[S]):
        selected_solutions = []
        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            selected_solutions.append(solution)
        return selected_solutions

    def reproduction(self, population: List[S]) -> List[S]:
        from evolu.core.exceptions import InvalidParentsException
        number_of_parents = self.crossover_operator.get_number_of_parents()
        if len(population) % number_of_parents != 0:
            raise InvalidParentsException(
                f"Wrong number of parents: {len(population)} is not divisible by {number_of_parents}"
            )
        offsprings = []
        for j in range(0, self.offspring_population_size, number_of_parents):
            parents = []
            for k in range(number_of_parents):
                parents.append(population[j + k])
            new_solutions = self.crossover_operator.execute(parents)
            for solution in new_solutions:
                new_solution = self.mutation_operator.execute(solution)
                offsprings.append(new_solution)
                if len(offsprings) >= self.offspring_population_size:
                    break
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[List[S]]:
        """
        This method joins the current and offspring populations to produce the population of the next generation
        by applying the ranking and crowding distance selection.
        param population: Parent population.
        param offsprings: Offspring population.
        return: New population after ranking and crowding distance selection is applied.
        """
        solutions = self.replacement_operator.replace(population, offsprings)
        return solutions

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
