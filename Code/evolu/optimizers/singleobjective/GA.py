# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Crossover, Mutation, Selection
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.operator.replacement import JoinPopulationSelectionReplacement
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic genetic algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""
"""
Module: Steady-State genetic algorithm(offspring_population_size=1)
Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
Please contact me (zixiangliwust@gmail.com) freely if you find some mistakes or verify that this module is correct;
Modified or confirmed by the researchers listed as follows[Hoping for 10 researchers to confirm the codes]:
[1] Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
"""


class GABase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic genetic algorithm
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy, Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
    [2] https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [3] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [4] Whitley, Darrell. 1994. "A genetic algorithm tutorial."  Statistics and Computing 4 (2):65-85. doi: 10.1007/BF00175354.
    """
    """
    Steady-State genetic algorithm(offspring_population_size=1)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy, Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
    [2] Durillo, Juan J., Antonio J. Nebro, Francisco Luna, and Enrique Alba. 2009. On the Effect of the Steady-State Selection Scheme in Multi-Objective Genetic Algorithms.
    Paper presented at the Evolutionary Multi-Criterion Optimization, Berlin, Heidelberg, 2009.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(GABase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Genetic algorithm"
        self.offspring_population_size = offspring_population_size
        self.mutation_operator = mutation
        self.crossover_operator = crossover
        self.selection_operator = selection
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.mating_pool_size = (
                self.offspring_population_size
                * self.crossover_operator.get_number_of_parents()
                // self.crossover_operator.get_number_of_children()
        )
        if self.mating_pool_size < self.crossover_operator.get_number_of_children():
            self.mating_pool_size = self.crossover_operator.get_number_of_children()
        self.replacement_operator = JoinPopulationSelectionReplacement(self.comparator)

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


"""
Module: Elite genetic algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class GAElite(GABase[S, R]):
    """
    Elite genetic algorithm
    References:
    [1] https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [3] Whitley, Darrell. 1994. "A genetic algorithm tutorial."  Statistics and Computing 4 (2):65-85. doi: 10.1007/BF00175354.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(GAElite, self).__init__(problem=problem,
                                      population_size=population_size,
                                      offspring_population_size=offspring_population_size,
                                      mutation=mutation,
                                      crossover=crossover,
                                      selection=selection,
                                      population_generator=population_generator,
                                      population_evaluator=population_evaluator,
                                      termination_criterion=termination_criterion,
                                      )
        self.algorithm_name = "Elite genetic algorithm"

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        population = self.sort_population.execute(population)
        population[-len(offsprings):] = copy.deepcopy(offsprings)
        return population

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved genetic algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class IGA(GABase[S, R]):
    """
    Improved genetic algorithm
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 offspring_population_size: int,
                 mutation: Mutation,
                 crossover: Crossover,
                 selection: Selection,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IGA, self).__init__(problem=problem,
                                  population_size=population_size,
                                  offspring_population_size=offspring_population_size,
                                  mutation=mutation,
                                  crossover=crossover,
                                  selection=selection,
                                  population_generator=population_generator,
                                  population_evaluator=population_evaluator,
                                  termination_criterion=termination_criterion,
                                  )
        self.algorithm_name = "Improved genetic algorithm"
        self.scout_time = 10
        self.neighbor_size = 10
        self.local_search_probability = 0.1

    def scout_phase(self) -> None:
        for j in range(0, self.population_size):
            if self.solutions[j].survive_time >= self.scout_time:
                neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in range(0, self.neighbor_size)]
                neighbors = self.evaluate(neighbors)
                neighbors = self.sort_population.execute(neighbors)
                for solution in neighbors[0:]:
                    if self.identical_solutions_comparator.compare(solution, self.solutions[j]) != 0:
                        self.solutions[j] = solution
                        break
                self.offspring_population_size += self.neighbor_size

    def local_search(self) -> None:
        for j in range(0, self.population_size):
            if random.random() < self.local_search_probability:
                for i in range(0, self.problem.number_of_variables):
                    new_solution = self.mutation_operator.execute(self.solutions[j])
                    if self.identical_solutions_comparator.compare(self.solutions[j], new_solution) != 0:
                        new_solution.survive_time = 0
                    else:
                        new_solution.survive_time = self.solutions[j].survive_time
                    new_solution = self.evaluate_solution(new_solution)
                    self.offspring_population_size += 1
                    if self.comparator.compare(new_solution, self.solutions[j]) == -1:
                        self.solutions[j] = copy.deepcopy(new_solution)

    def evolve(self) -> None:
        self.offspring_population_size = self.population_size
        for solution in self.solutions:
            solution.survive_time += 1
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)
        self.scout_phase()
        self.local_search()

    def after_evolve(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        self.solutions = self.restart_operator.execute(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
