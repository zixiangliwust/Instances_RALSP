# -*- coding: utf-8 -*-
import copy
import random
import time
import math
from typing import List, TypeVar
import numpy as np
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.operator.selection import RouletteWheelSelection
from evolu.util.generator import Generator
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.util.evaluator import Evaluator, SequentialEvaluator
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic tabu search algorithm
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PermutationTS(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic tabu search algorithm
    References:
    [1] Özcan, Uğur, and Bilal Toklu. 2009. "A tabu search algorithm for two-sided assembly line balancing."
    The International Journal of Advanced Manufacturing Technology 43 (7):822-9. doi: 10.1007/s00170-008-1753-5.
    """

    def __init__(self,
                 problem: Problem[S],
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PermutationTS, self).__init__(problem=problem)
        self.algorithm_name = "Tabu search algorithm"
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.tabu_length = 20
        self.neighbor_size = 20
        self.swap_tabu_list = np.zeros((self.problem.number_of_variables, self.problem.number_of_variables))
        self.insert_tabu_list = np.zeros((self.problem.number_of_variables, self.problem.number_of_variables))
        self.solution = None

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def evolve(self) -> None:
        best_swap_neighbor, best_insert_neighbor = copy.deepcopy(self.solution), copy.deepcopy(self.solution)
        best_swap_pair, best_insert_pair = [], []
        best_swap_neighbor.objectives = [pow(10, 8) for _ in range(0, len(best_swap_neighbor.objectives))]
        best_insert_neighbor.objectives = [pow(10, 8) for _ in range(0, len(best_insert_neighbor.objectives))]
        for j in range(0, self.neighbor_size):
            neighbor = copy.deepcopy(self.solution)
            if np.random.random() < 0.5:
                selected_task_a, selected_task_b = np.random.choice(range(0, self.problem.number_of_variables), 2,
                                                                    replace=False)
                task_list = np.array(neighbor.variables)
                idx1 = np.where(task_list == selected_task_a)
                idx1 = idx1[0][0]
                task_list = np.array(neighbor.variables)
                idx2 = np.where(task_list == selected_task_b)
                idx2 = idx2[0][0]
                swap_pair = [selected_task_a, selected_task_b]
                neighbor.variables[idx1], neighbor.variables[idx2] = neighbor.variables[idx2], neighbor.variables[idx1]
                neighbor = self.evaluate_solution(neighbor)
                if self.comparator.compare(neighbor, self.g_best) == -1:
                    self.g_best = copy.deepcopy(neighbor)
                if self.swap_tabu_list[swap_pair[0]][swap_pair[1]] == 0 or \
                        self.comparator.compare(neighbor, self.solution) == -1:
                    if self.comparator.compare(neighbor, best_swap_neighbor) == -1:
                        best_swap_neighbor = copy.deepcopy(neighbor)
                        best_swap_pair = copy.deepcopy(swap_pair)
            else:
                selected_task_a = random.randint(0, self.problem.number_of_variables - 1)
                selected_position = random.randint(0, self.problem.number_of_variables - 1)
                task_list = np.array(neighbor.variables)
                idx1 = np.where(task_list == selected_task_a)
                idx1 = idx1[0][0]
                while idx1 == selected_position:
                    selected_position = random.randint(0, self.problem.number_of_variables - 1)
                insert_pair = [selected_task_a, selected_position]
                neighbor.variables.insert(selected_position, neighbor.variables[idx1])
                if idx1 < selected_position:
                    neighbor.variables.remove(neighbor.variables[idx1])
                else:
                    neighbor.variables.remove(neighbor.variables[idx1 + 1])
                neighbor = self.evaluate_solution(neighbor)
                if self.comparator.compare(neighbor, self.g_best) == -1:
                    self.g_best = copy.deepcopy(neighbor)
                if self.insert_tabu_list[insert_pair[0]][insert_pair[1]] == 0 or \
                        self.comparator.compare(neighbor, self.solution) == -1:
                    if self.comparator.compare(neighbor, best_insert_neighbor) == -1:
                        best_insert_neighbor = copy.deepcopy(neighbor)
                        best_insert_pair = copy.deepcopy(insert_pair)
        if best_swap_neighbor.objectives[0] < pow(10, 8) or best_insert_neighbor.objectives[0] < pow(10, 8):
            if self.comparator.compare(best_swap_neighbor, best_insert_neighbor) <= 0:
                self.swap_tabu_list[best_swap_pair[0]][best_swap_pair[1]] = self.tabu_length
                self.swap_tabu_list[best_swap_pair[1]][best_swap_pair[0]] = self.tabu_length
                if self.comparator.compare(best_swap_neighbor, self.solution) == -1:
                    self.solution = best_swap_neighbor
            else:
                self.insert_tabu_list[best_insert_pair[0]][best_insert_pair[1]] = self.tabu_length
                if self.comparator.compare(best_insert_neighbor, self.solution) == -1:
                    self.solution = best_insert_neighbor
        for i in range(0, self.problem.number_of_variables):
            for j in range(0, self.problem.number_of_variables):
                if self.swap_tabu_list[i][j] >= 1:
                    self.swap_tabu_list[i][j] = self.swap_tabu_list[i][j] - 1
                if self.insert_tabu_list[i][j] >= 1:
                    self.insert_tabu_list[i][j] = self.insert_tabu_list[i][j] - 1

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        pass

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name

