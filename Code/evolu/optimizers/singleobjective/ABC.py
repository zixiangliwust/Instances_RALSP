# -*- coding: utf-8 -*-
import copy
import random
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation, Selection
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.operator.replacement import JoinPopulationSelectionReplacement
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic artificial bee colony
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class ABCBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic artificial bee colony
    References:
    [1] Karaboga, Dervis. 2005. "An idea based on honey bee swarm for numerical optimization."
    In.: Technical report-tr06, Erciyes university, engineering faculty, computer engineering department.
    [2] Karaboga, D., and B. Basturk. 2008. "On the performance of artificial bee colony (ABC) algorithm."  
    Applied Soft Computing 8 (1):687-97. doi: https://doi.org/10.1016/j.asoc.2007.05.007.    
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(ABCBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Artificial bee colony"
        self.selection_operator = selection
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.scout_time = 120
        self.offspring_population_size = 2 * self.population_size

    def employed_bee_selection(self, population: List[S]) -> List[S]:
        return population

    def employed_bee_reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            offsprings[j] = self.mutation_operator.execute(offsprings[j])
            if self.identical_solutions_comparator.compare(population[j], offsprings[j]) != 0:
                offsprings[j].survive_time = 0
            else:
                offsprings[j].survive_time = offsprings[j].survive_time
        return offsprings

    def employed_bee_replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.replacement_operator.replace(population, offsprings)

    def onlooker_reproduction(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for _ in range(0, self.population_size):
            idx = self.selection_operator.return_index(offsprings)
            new_solution = self.mutation_operator.execute(offsprings[idx])
            if self.identical_solutions_comparator.compare(offsprings[idx], new_solution) != 0:
                new_solution.survive_time = 0
            else:
                new_solution.survive_time = offsprings[idx].survive_time
            new_solution = self.evaluate_solution(new_solution)
            if self.comparator.compare(new_solution, offsprings[idx]) == -1:
                offsprings[idx] = copy.deepcopy(new_solution)
        return offsprings

    def scout_phase(self) -> None:
        worst_solution_index = 0
        for j in range(0, self.population_size):
            if self.solutions[worst_solution_index].survive_time < self.solutions[j].survive_time:
                worst_solution_index = j
        for j in range(0, self.population_size):
            if j == worst_solution_index and self.solutions[j].survive_time >= self.scout_time:
                self.solutions[j] = self.create_solution()
                self.solutions[j] = self.evaluate_solution(self.solutions[j])
                self.offspring_population_size += 1

    def evolve(self):
        self.offspring_population_size = 2 * self.population_size
        for solution in self.solutions:
            solution.survive_time += 1
        selected_solutions = self.employed_bee_selection(self.solutions)
        offsprings = self.employed_bee_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.employed_bee_replacement(self.solutions, offsprings)
        self.solutions = self.onlooker_reproduction(self.solutions)
        self.scout_phase()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Artificial bee colony 1 [see Section 3.3. Swarm intelligence algorithms in Li et al. 2019]
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class ABC1(ABCBase[S, R]):
    """
    Artificial bee colony 1 [see Section 3.3. Swarm intelligence algorithms in Li et al. 2019]
    References:
    [1] Li, Zixiang, Mukund Nilakantan Janardhanan, Qiuhua Tang, and S. G. Ponnambalam. 2019.
    "Model and metaheuristics for robotic two-sided assembly line balancing problems with setup times."
    Swarm and Evolutionary Computation 50:100567. doi: 10.1016/j.swevo.2019.100567.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(ABC1, self).__init__(problem=problem, population_size=population_size, selection=selection,
                                   mutation=mutation,
                                   population_generator=population_generator, population_evaluator=population_evaluator,
                                   termination_criterion=termination_criterion)
        self.algorithm_name = "Artificial bee colony1"

    def scout_phase(self) -> None:
        worst_solution_index = 0
        for idx in range(1, len(self.solutions)):
            if self.comparator.compare(self.solutions[idx], self.solutions[worst_solution_index]) > 0:
                worst_solution_index = idx
            elif self.comparator.compare(self.solutions[idx], self.solutions[worst_solution_index]) == 0:
                if self.solutions[worst_solution_index].survive_time < self.solutions[idx].survive_time:
                    worst_solution_index = idx
        j = random.randint(0, self.population_size - 1)
        self.solutions[worst_solution_index] = self.mutation_operator.execute(self.solutions[j])
        self.solutions[worst_solution_index] = self.evaluate_solution(self.solutions[worst_solution_index])
        self.offspring_population_size += 1

    def after_evolve(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        self.solutions = self.restart_operator.execute(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Artificial bee colony 2 [see Section 3.3. Swarm intelligence algorithms in Li et al. 2019]
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class ABC2(ABCBase[S, R]):
    """
    Artificial bee colony 2 [see Section 3.3. Swarm intelligence algorithms in Li et al. 2019]
    References:
    [1] Li, Zixiang, Mukund Nilakantan Janardhanan, Qiuhua Tang, and S. G. Ponnambalam. 2019.
    "Model and metaheuristics for robotic two-sided assembly line balancing problems with setup times."
    Swarm and Evolutionary Computation 50:100567. doi: 10.1016/j.swevo.2019.100567.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(ABC2, self).__init__(problem=problem, population_size=population_size, selection=selection,
                                   mutation=mutation,
                                   population_generator=population_generator, population_evaluator=population_evaluator,
                                   termination_criterion=termination_criterion)
        self.algorithm_name = "Artificial bee colony2"

    def scout_phase(self) -> None:
        worst_solution_index = 0
        for idx in range(1, len(self.solutions)):
            if self.comparator.compare(self.solutions[idx], self.solutions[worst_solution_index]) > 0:
                worst_solution_index = idx
            elif self.comparator.compare(self.solutions[idx], self.solutions[worst_solution_index]) == 0:
                if self.solutions[worst_solution_index].survive_time < self.solutions[idx].survive_time:
                    worst_solution_index = idx
        j = random.randint(0, self.population_size - 1)
        self.solutions[worst_solution_index] = self.mutation_operator.execute(self.solutions[j])
        self.solutions[worst_solution_index] = self.evaluate_solution(self.solutions[worst_solution_index])
        self.offspring_population_size += 1

    def after_evolve(self) -> None:
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        self.solutions = self.restart_operator.execute(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved artificial bee colony algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class IABC(ABCBase[S, R]):
    """
    Improved artificial bee colony algorithm
    References:
    [1] Çil, Zeynel Abidin, Zixiang Li, Suleyman Mete, and Eren Özceylan. 2020.
    "Mathematical model and bee algorithms for mixed-model assembly line balancing problem with physical human–robot collaboration."
    Applied Soft Computing 93:106394. doi: https://doi.org/10.1016/j.asoc.2020.106394.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 selection: Selection,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IABC, self).__init__(problem=problem, population_size=population_size, selection=selection,
                                   mutation=mutation,
                                   population_generator=population_generator, population_evaluator=population_evaluator,
                                   termination_criterion=termination_criterion)
        self.algorithm_name = "Improved artificial bee colony algorithm"
        self.neighbor_size = 10
        self.local_search_probability = 0.1
        self.scout_time = 10
        self.join_population_selection_replacement = JoinPopulationSelectionReplacement(self.comparator)

    def onlooker_reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        for _ in range(0, self.population_size):
            idx = self.selection_operator.return_index(population)
            new_solution = self.mutation_operator.execute(population[idx])
            if self.identical_solutions_comparator.compare(population[idx], new_solution) != 0:
                new_solution.survive_time = 0
            else:
                new_solution.survive_time = population[idx].survive_time
            offsprings.append(new_solution)
        return offsprings

    def onlooker_replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        return self.join_population_selection_replacement.replace(population, offsprings)

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

    def evolve(self):
        self.offspring_population_size = 2 * self.population_size
        for solution in self.solutions:
            solution.survive_time += 1
        selected_solutions = self.employed_bee_selection(self.solutions)
        offsprings = self.employed_bee_reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.employed_bee_replacement(self.solutions, offsprings)
        offsprings = self.onlooker_reproduction(self.solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.onlooker_replacement(self.solutions, offsprings)
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
