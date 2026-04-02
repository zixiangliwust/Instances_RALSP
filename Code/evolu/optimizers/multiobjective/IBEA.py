import time
import math
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.operator import Crossover, Mutation
from evolu.core.problem import Problem
from evolu.core.quality_indicator import EpsilonIndicator
from evolu.operator.selection import BinaryTournamentSelection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.comparator import SolutionAttributeComparator, EpsilonDominanceComparator
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")


class IBEA(MultiObjectiveSwarmRoot[S, R]):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 kappa: float,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Epsilon IBEA implementation as described in
        * Zitzler, Eckart, and Simon Künzli. "Indicator-based selection in multiobjective search."
        In International Conference on Parallel Problem Solving from Nature, pp. 832-842. Springer,
        Berlin, Heidelberg, 2004.
        https://link.springer.com/chapter/10.1007/978-3-540-30217-9_84
        IBEA is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs)
        family. The multi-objective search in IBEA is guided by a fitness associated to every solution,
        which is in turn controlled by a binary quality indicator. This implementation uses the so-called
        additive epsilon indicator, along with a binary tournament mating selector.
        param problem: The problem to solve.
        param population_size: Size of the population.
        param mutation: Mutation operator (see :py:mod:`evolu.operator.mutation`).
        param crossover: Crossover operator (see :py:mod:`evolu.operator.crossover`).
        param kappa: Weight in the fitness computation.
        """
        self.algorithm_name = "Epsilon-IBEA"
        selection = BinaryTournamentSelection(
            comparator=SolutionAttributeComparator(key="fitness", lowest_is_best=False)
        )
        self.kappa = kappa
        super(IBEA, self).__init__(problem=problem, population_size=population_size)
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.population_evaluator = population_evaluator
        self.population_generator = population_generator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
        self.mating_pool_size = (
                self.offspring_population_size
                * self.crossover_operator.get_number_of_parents()
                // self.crossover_operator.get_number_of_children()
        )
        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()

    def compute_fitness_values(self, population: List[S], kappa: float) -> List[S]:
        for i in range(len(population)):
            population[i].attributes["fitness"] = 0
            for j in range(len(population)):
                if j != i:
                    population[i].attributes["fitness"] += -math.exp(
                        -EpsilonIndicator([population[i].objectives]).compute([population[j].objectives]) / self.kappa
                    )
        return population

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

    def create_initial_solutions(self, population_size=None) -> List[S]:
        if population_size is None:
            population_size = self.population_size
        population = [self.population_generator.create_solution(self.problem) for _ in range(population_size)]
        population = self.compute_fitness_values(population, self.kappa)
        return population

    def replacement(self, population: List[S], offsprings: List[S]) -> List[List[S]]:
        join_population = population + offsprings
        join_population_size = len(join_population)
        join_population = self.compute_fitness_values(join_population, self.kappa)
        while join_population_size > self.population_size:
            current_fitnesses = [individual.attributes["fitness"] for individual in join_population]
            index_worst = current_fitnesses.index(min(current_fitnesses))
            for i in range(join_population_size):
                join_population[i].attributes["fitness"] += math.exp(
                    -EpsilonIndicator([join_population[i].objectives]).compute(
                        [join_population[index_worst].objectives]
                    )
                    / self.kappa
                )
            join_population.pop(index_worst)
            join_population_size = join_population_size - 1
        return join_population

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
