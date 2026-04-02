# -*- coding: utf-8 -*-
import time
from typing import Generator, List, TypeVar

try:
    import dask
    from distributed import Client, as_completed
except ImportError:
    pass
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.algorithm import Algorithm, DynamicAlgorithm
from evolu.core.operator import Crossover, Mutation, Selection
from evolu.core.problem import DynamicProblem, Problem
from evolu.operator.selection import BinaryTournamentSelection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.comparator import (Comparator, DominanceWithConstraintsComparator, MultiComparator,
                                         SolutionAttributeComparator,
                                         EpsilonDominanceComparator)
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.evaluator import Evaluator
from evolu.util.ranking import FastNonDominatedRanking
from evolu.operator.replacement import JoinPopulationRankingAndDensityEstimatorReplacement
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: NSGA-II (Fast and elitist multiobjective genetic algorithm)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class NSGAII(MultiObjectiveSwarmRoot[S, R]):
    """
    NSGA-II (Fast and elitist multiobjective genetic algorithm)
    Note: NSGA-II is a genetic algorithm (GA), i.e. it belongs to the evolutionary algorithms (EAs) family.
    Note: A steady-state version of this algorithm can be run by setting the offspring size to 1.
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and TAMT Meyarivan. 2002. "A fast and elitist multiobjective genetic algorithm: NSGA-II."
    IEEE transactions on evolutionary computation 6 (2):182-97.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 dominance_comparator: Comparator = store.default_comparator,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(NSGAII, self).__init__(problem=problem, population_size=population_size)
        self.offspring_population_size = offspring_population_size
        self.selection_operator = selection
        self.crossover_operator = crossover
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.algorithm_name = "NSGAII"
        self.dominance_comparator = dominance_comparator
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(self.dominance_comparator), CrowdingDistance())
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

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name



class DynamicNSGAII(NSGAII[S, R], DynamicAlgorithm):
    def __init__(self,
                 problem: DynamicProblem[S],
                 population_size: int,
                 offspring_population_size: int,
                 selection: Selection,
                 crossover: Crossover,
                 mutation: Mutation,
                 dominance_comparator: DominanceWithConstraintsComparator = DominanceWithConstraintsComparator(),
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(DynamicNSGAII, self).__init__(
            problem=problem,
            population_size=population_size,
            offspring_population_size=offspring_population_size,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            dominance_comparator=dominance_comparator,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
        )
        # Note: NSGAII parent class handles operator assignments properly
        self.completed_iterations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0

    def restart(self):
        self.solutions = self.evaluate(self.solutions)

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.evaluations += self.offspring_population_size

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            observable_data["TERMINATION_CRITERIA_IS_MET"] = True
            self.observable.notify_all(**observable_data)
            self.restart()
            self.init_progress()
            self.completed_iterations += 1
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class DistributedNSGAII(Algorithm[S, R]):
    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 number_of_cores: int,
                 client,
                 selection: Selection = BinaryTournamentSelection(
                     MultiComparator([SolutionAttributeComparator("dominance_ranking"),
                                      SolutionAttributeComparator("crowding_distance", lowest_is_best=False)])
                 ),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 dominance_comparator: DominanceWithConstraintsComparator = DominanceWithConstraintsComparator(),
                 ):
        super(DistributedNSGAII, self).__init__()
        self.problem = problem
        self.population_size = population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.dominance_comparator = dominance_comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.number_of_cores = number_of_cores
        self.client = client
        self.replacement_operator = JoinPopulationRankingAndDensityEstimatorReplacement(
            FastNonDominatedRanking(self.dominance_comparator), CrowdingDistance())

    def create_initial_solutions(self) -> List[S]:
        return [self.problem.create_solution() for _ in range(self.number_of_cores)]

    def evaluate(self, solutions: List[S]) -> List[S]:
        return self.client.map(self.problem.evaluate_solution, solutions)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_observable_data(self) -> dict:
        total_computing_time = time.time() - self.start_computing_time
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "TOTAL_TIME": total_computing_time,
        }

    def init_progress(self) -> None:
        self.evaluations = self.number_of_cores
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def evolve(self) -> None:
        pass

    def update_progress(self):
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def run(self):
        """Execute the algorithm."""
        self.start_computing_time = time.time()
        create_solution = dask.delayed(self.problem.create_solution)
        evaluate_solution = dask.delayed(self.problem.evaluate_solution)
        task_pool = as_completed([], with_results=True)
        for _ in range(self.number_of_cores):
            new_solution = create_solution()
            new_evaluated_solution = evaluate_solution(new_solution)
            future = self.client.compute(new_evaluated_solution)
            task_pool.add(future)
        batches = task_pool.batches()
        auxiliar_population = []
        while len(auxiliar_population) < self.population_size:
            batch = next(batches)
            for _, received_solution in batch:
                auxiliar_population.append(received_solution)
                if len(auxiliar_population) < self.population_size:
                    break
            # submit as many new tasks as we collected
            for _ in batch:
                new_solution = create_solution()
                new_evaluated_solution = evaluate_solution(new_solution)
                future = self.client.compute(new_evaluated_solution)
                task_pool.add(future)
        self.init_progress()
        # perform an algorithm step to create a new solution to be evaluated
        while not self.stopping_condition_is_met():
            batch = next(batches)
            for _, received_solution in batch:
                offsprings = [received_solution]
                # replacement
                auxiliar_population = self.replacement_operator.replace(auxiliar_population, offsprings)
                # selection
                selected_solutions = []
                for _ in range(2):
                    solution = self.selection_operator.execute(auxiliar_population)
                    selected_solutions.append(solution)
                # Reproduction and evaluation
                new_task = self.client.submit(
                    reproduction, selected_solutions, self.problem, self.crossover_operator, self.mutation_operator
                )
                task_pool.add(new_task)
                # update progress
                self.evaluations += 1
                self.solutions = auxiliar_population
                self.update_progress()
                if self.stopping_condition_is_met():
                    break
        self.total_computing_time = time.time() - self.start_computing_time
        # at this point, computation is done
        for future, _ in task_pool:
            future.cancel()

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return "dNSGA-II"
        
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


def reproduction(population: List[S], problem, crossover_operator, mutation_operator) -> S:
    offspring_pool = []
    for parents in zip(*[iter(population)] * 2):
        offspring_pool.append(crossover_operator.execute(parents))
    offsprings = []
    for pair in offspring_pool:
        for solution in pair:
            mutated_solution = mutation_operator.execute(solution)
            offsprings.append(mutated_solution)
    return problem.evaluate(offsprings[0])
