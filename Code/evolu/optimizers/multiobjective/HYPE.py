import time
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.operator import Crossover, Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.operator.selection import BinaryTournamentSelection, RankingAndFitnessSelection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.comparator import (Comparator, SolutionAttributeComparator, DominanceWithConstraintsComparator,
                                         EpsilonDominanceComparator)
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")


class HYPE(MultiObjectiveSwarmRoot[S, R]):
    def __init__(self,
                 problem: Problem,
                 reference_point: Solution,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 dominance_comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """This is an implementation of the Hypervolume Estimation Algorithm for Multi-objective Optimization
        proposed in:
        * J. Bader and E. Zitzler. HypE: An Algorithm for Fast Hypervolume-Based Many-Objective
        Optimization. TIK Report 286, Computer Engineering and Networks Laboratory (TIK), ETH
        Zurich, November 2008.
        It uses the Exact Hypervolume-based indicator formulation, which once computed, guides both
        the environmental selection and the binary tournament selection operator
        Please note that as per the publication above, the evaluator and replacement should not be changed
        anyhow. It also requires that Problem() has a reference_point with objective values defined, e.g.
        problem = ZDT1()
        reference_point = FloatSolution(problem.number_of_variables,problem.number_of_objectives, [0], [1])
        reference_point.objectives = [1., 1.]
        """
        self.algorithm_name = "HYPE"
        selection = BinaryTournamentSelection(
            comparator=SolutionAttributeComparator(key="fitness", lowest_is_best=False)
        )
        self.ranking_fitness = RankingAndFitnessSelection(
            population_size, dominance_comparator=dominance_comparator, reference_point=reference_point
        )
        self.reference_point = reference_point
        self.dominance_comparator = dominance_comparator
        super(HYPE, self).__init__(problem=problem, population_size=population_size)
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.termination_criterion = termination_criterion
        self.population_evaluator = population_evaluator
        self.population_generator = population_generator
        self.observable.register(termination_criterion)
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

    def evaluate(self, solution_list: List[S]):
        solution_list = self.population_evaluator.evaluate(solution_list, self.problem)
        solution_list = self.ranking_fitness.compute_hypervol_fitness_values(
            solution_list, self.reference_point, len(solution_list)
        )
        return solution_list

    def replacement(self, population: List[S], offsprings: List[S]) -> List[List[S]]:
        join_population = population + offsprings
        return self.ranking_fitness.execute(join_population)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
