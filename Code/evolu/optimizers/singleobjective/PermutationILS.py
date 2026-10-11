# -*- coding: utf-8 -*-
import copy
import random
import time
import math
import numpy as np
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.generator import Generator
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Iterated greedy algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PermutationIG(SingleObjectiveSwarmRoot[S, R]):
    """
    Iterated greedy algorithm
    References:
    [1] Li, Zixiang, Qiuhua Tang, and LiPing Zhang. 2017. "Two-sided assembly line balancing problem of type I: Improvements, a simple algorithm and a comprehensive study."
    Computers & Operations Research 79:78-93. doi: 10.1016/j.cor.2016.10.006.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PermutationIG, self).__init__(problem=problem)
        self.algorithm_name = "Iterated greedy algorithm"
        self.number_of_variables = problem.number_of_variables
        self.mutation_operator = mutation
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.perturbation_num = 2
        self.temperature = 0.001
        self.perturbation_num = min(self.perturbation_num, self.number_of_variables - 1)
        self.solution = None

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        if self.problem.problem_name == "SimpleALBP1":
            self.solution = self.problem.create_evaluate_ranked_positional_weight_heuristic_solution()
        else:
            self.solution = self.create_solution()
            self.solution = self.evaluate_solution(self.solution)
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best
        self.offspring_population_size = 0
        self.solution = self.strong_local_search(self.solution)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1 + self.offspring_population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def destruction_construction(self, parent: Solution) -> Solution:
        new_solution = copy.deepcopy(parent)
        permutation = copy.deepcopy(new_solution.variables)
        removed_task = []
        for i in range(0, self.perturbation_num):
            idx = np.random.randint(0, len(permutation))
            removed_task.append(permutation[idx])
            permutation.remove(permutation[idx])
        permutation = permutation + removed_task
        new_solution.variables = permutation
        new_solution = self.evaluate_solution(new_solution)
        for i in range(0, self.perturbation_num):
            for insert_index in range(0, self.number_of_variables):
                neighbor = copy.deepcopy(new_solution)
                permutation = np.array(neighbor.variables)
                selected_index = np.where(permutation == removed_task[i])
                selected_index = selected_index[0][0]
                neighbor.variables.insert(insert_index, neighbor.variables[selected_index])
                if selected_index < insert_index:
                    neighbor.variables.remove(neighbor.variables[selected_index])
                else:
                    neighbor.variables.remove(neighbor.variables[selected_index + 1])
                neighbor = self.evaluate_solution(neighbor)
                if self.comparator.compare(neighbor, new_solution) == -1:
                    new_solution = neighbor
                self.offspring_population_size += 1
        return new_solution

    def strong_local_search(self, parent) -> Solution:
        new_solution = copy.deepcopy(parent)
        best_permutation = self.g_best.variables
        non_improve_time = 0
        para_a, para_b = 20, 4
        task_index = 0
        while non_improve_time < self.number_of_variables * para_a:
            last_swap_index = -1
            last_insert_index = -1
            for j in range(0, para_b):
                neighbor = copy.deepcopy(new_solution)
                if np.random.random() < 0.5:
                    permutation = np.array(neighbor.variables)
                    selected_index = np.where(permutation == best_permutation[task_index])
                    selected_index = selected_index[0][0]
                    swap_index = np.random.randint(0, self.number_of_variables)
                    while (selected_index == swap_index) or (swap_index == last_swap_index):
                        swap_index = np.random.randint(0, self.number_of_variables)
                    neighbor.variables[selected_index], neighbor.variables[swap_index] = \
                        neighbor.variables[swap_index], neighbor.variables[selected_index]
                    last_swap_index = swap_index
                else:
                    permutation = np.array(neighbor.variables)
                    selected_index = np.where(permutation == best_permutation[task_index])
                    selected_index = selected_index[0][0]
                    insert_index = np.random.randint(0, self.number_of_variables)
                    while (selected_index == insert_index) or (insert_index == last_insert_index):
                        insert_index = np.random.randint(0, self.number_of_variables)
                    neighbor.variables.insert(insert_index, neighbor.variables[selected_index])
                    if selected_index < insert_index:
                        neighbor.variables.remove(neighbor.variables[selected_index])
                    else:
                        neighbor.variables.remove(neighbor.variables[selected_index + 1])
                    last_insert_index = insert_index
                neighbor = self.evaluate_solution(neighbor)
                if self.comparator.compare(neighbor, new_solution) == -1:
                    new_solution = neighbor
                    non_improve_time = 0
                elif self.comparator.compare(neighbor, new_solution) == 0:
                    new_solution = neighbor
                non_improve_time = non_improve_time + 1
                self.offspring_population_size += 1
            task_index = task_index + 1
            task_index = task_index % self.number_of_variables
        return new_solution

    def evolve(self) -> None:
        self.offspring_population_size = 0
        new_solution = self.destruction_construction(self.g_best)
        acceptance_probability = self.compute_acceptance_probability(
            self.solution.objectives[0], new_solution.objectives[0], self.temperature
        )
        if self.comparator.compare(new_solution, self.solution) == -1:
            self.solution = new_solution
        elif acceptance_probability > random.random():
            self.solution = new_solution
        self.solution = self.strong_local_search(self.solution)

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return np.exp(-1.0 * value)

    def after_initialization(self) -> None:
        if self.comparator.compare(self.solution, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.solution)

    def after_evolve(self) -> None:
        if self.comparator.compare(self.solution, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.solution)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name



"""
Module: Iterated local search
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PermutationILS(SingleObjectiveSwarmRoot[S, R]):
    """
    Iterated local search
    References:
    [1] Li, Zixiang, Ibrahim Kucukkoc, and Zikai Zhang. 2019. "Iterated local search method and mathematical model for sequence-dependent U-shaped disassembly line balancing problem."
    Computers & Industrial Engineering 137:106056. doi: 10.1016/j.cie.2019.106056.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PermutationILS, self).__init__(problem=problem)
        self.algorithm_name = "Iterated local search"
        self.number_of_variables = problem.number_of_variables
        self.mutation_operator = mutation
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.perturbation_num = 4
        self.neighbor_size = 50
        self.solution = None

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        if self.problem.problem_name == "SimpleALBP1":
            self.solution = self.problem.create_evaluate_ranked_positional_weight_heuristic_solution()
        else:
            self.solution = self.create_solution()
            self.solution = self.evaluate_solution(self.solution)
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best
        self.offspring_population_size = 0
        self.solution = self.strong_local_search(self.solution)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1 + self.offspring_population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def strong_local_search(self, parent: Solution) -> Solution:
        new_solution = copy.deepcopy(parent)
        best_permutation = self.g_best.variables
        non_improve_time = 0
        para_a, para_b = 20, 4
        task_index = 0
        while non_improve_time < self.number_of_variables * para_a:
            last_swap_index = -1
            last_insert_index = -1
            for j in range(0, para_b):
                neighbor = copy.deepcopy(new_solution)
                if np.random.random() < 0.5:
                    permutation = np.array(neighbor.variables)
                    selected_index = np.where(permutation == best_permutation[task_index])
                    selected_index = selected_index[0][0]
                    swap_index = np.random.randint(0, self.number_of_variables)
                    while (selected_index == swap_index) or (swap_index == last_swap_index):
                        swap_index = np.random.randint(0, self.number_of_variables)
                    neighbor.variables[selected_index], neighbor.variables[swap_index] = \
                        neighbor.variables[swap_index], neighbor.variables[selected_index]
                    last_swap_index = swap_index
                else:
                    permutation = np.array(neighbor.variables)
                    selected_index = np.where(permutation == best_permutation[task_index])
                    selected_index = selected_index[0][0]
                    insert_index = np.random.randint(0, self.number_of_variables)
                    while (selected_index == insert_index) or (insert_index == last_insert_index):
                        insert_index = np.random.randint(0, self.number_of_variables)
                    neighbor.variables.insert(insert_index, neighbor.variables[selected_index])
                    if selected_index < insert_index:
                        neighbor.variables.remove(neighbor.variables[selected_index])
                    else:
                        neighbor.variables.remove(neighbor.variables[selected_index + 1])
                    last_insert_index = insert_index
                neighbor = self.evaluate_solution(neighbor)
                if self.comparator.compare(neighbor, new_solution) == -1:
                    new_solution = neighbor
                    non_improve_time = 0
                elif self.comparator.compare(neighbor, new_solution) == 0:
                    new_solution = neighbor
                non_improve_time = non_improve_time + 1
                self.offspring_population_size += 1
            task_index = task_index + 1
            task_index = task_index % self.number_of_variables
        return new_solution

    def perturbation_phase(self, parent: Solution) -> Solution:
        new_solution = copy.deepcopy(parent)
        neighbors = [copy.deepcopy(parent) for _ in range(0, self.neighbor_size)]
        for h in range(0, self.perturbation_num):
            for i in range(0, self.neighbor_size):
                neighbors[i] = self.mutation_operator.execute(neighbors[i])
        neighbors = self.evaluate(neighbors)
        neighbors = self.sort_population.execute(neighbors)
        for solution in neighbors[0:]:
            if self.identical_solutions_comparator.compare(solution, parent) != 0:
                new_solution = solution
                break
        self.offspring_population_size += self.neighbor_size
        return new_solution

    def evolve(self) -> None:
        self.offspring_population_size = 0
        self.solution = self.perturbation_phase(self.g_best)
        self.solution = self.strong_local_search(self.solution)

    def after_initialization(self) -> None:
        if self.comparator.compare(self.solution, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.solution)

    def after_evolve(self) -> None:
        if self.comparator.compare(self.solution, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.solution)
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
