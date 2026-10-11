# -*- coding: utf-8 -*-
import copy
import random
import time
import math
from typing import List, TypeVar, Optional, Any
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
Module: Tabu search utilizing station-oriented encoding [see 2.6. Simple-Tabu in Pape 2015]
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP2ntergerTS(SingleObjectiveSwarmRoot[S, R]):
    """
    Tabu search utilizing station-oriented encoding [see 2.6. Simple-Tabu in Pape 2015]
    References:
    [1] Pape, Tom. 2015. "Heuristics and lower bounds for the simple assembly line balancing problem type 1: Overview, computational tests and improvements."
    European Journal of Operational Research 240 (1):32-42. doi: http://dx.doi.org/10.1016/j.ejor.2014.06.023.
    """

    def __init__(self,
                 problem: Problem[S],
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ) -> None:
        """Initialize SALBP2 integer TS algorithm."""
        super(SALBP2ntergerTS, self).__init__(problem=problem)
        self.algorithm_name: str = "Tabu search utilizing station-oriented encoding"
        self.comparator: Comparator = comparator
        self.termination_criterion: TerminationCriterion = termination_criterion
        self.observable.register(termination_criterion)
        self.roulette_wheel_selection: RouletteWheelSelection = RouletteWheelSelection()
        # self.tabu_length = 10
        self.tabu_length: int = int(max(10, self.problem.n_tasks * 0.03))
        self.n_tasks: int = self.problem.n_tasks
        self.ct: int = self.problem.ct
        self.t: List[int] = self.problem.t
        self.imm_precedence: List[tuple] = self.problem.imm_precedence
        self.imm_precedence_matrix: Any = self.problem.imm_precedence_matrix
        self.all_precedence_matrix: Any = self.problem.all_precedence_matrix
        self.predecessors: List[List[int]] = self.problem.predecessors
        self.successors: List[List[int]] = self.problem.successors
        self.current_station_time: List[float] = []
        self.new_station_time: List[float] = []
        self.current_task_station: List[int] = [0 for _ in range(0, self.n_tasks)]
        self.new_task_station: List[int] = [0 for _ in range(0, self.n_tasks)]
        self.earliest_task_station: List[int] = [0 for _ in range(0, self.n_tasks)]
        self.latest_task_station: List[int] = [0 for _ in range(0, self.n_tasks)]
        self.tabu_list: np.ndarray = np.zeros((self.n_tasks, self.n_tasks))
        self.n_stations: Optional[int] = None
        self.current_ct: Optional[float] = None
        self.current_score: Optional[float] = None
        self.solution: Optional[Solution] = None

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.problem.create_evaluate_ranked_positional_weight_heuristic_integer_solution()
        self.n_stations = int(self.solution.objectives[0])
        self.current_station_time = [0 for _ in range(0, self.n_stations)]
        self.new_station_time = [0 for _ in range(0, self.n_stations)]
        for i in range(0, self.n_tasks):
            self.current_station_time[self.solution.variables[i]] = \
                self.current_station_time[self.solution.variables[i]] + self.t[i]
        self.current_task_station = copy.deepcopy(self.solution.variables)
        self.current_ct = max(self.current_station_time)
        self.current_score = 0
        for j in range(0, self.n_stations):
            if self.current_station_time[j] > self.ct:
                self.current_score = self.current_score + (self.current_station_time[j] - self.ct) * (
                        self.current_station_time[j] - self.ct)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def reduce_station(self) -> None:
        """Reduce number of stations."""
        self.check_assignment(self.current_task_station, self.n_stations, self.current_ct)
        selected_station = 0
        for j in range(1, self.n_stations):
            if self.current_station_time[selected_station] > self.current_station_time[j]:
                selected_station = j
        for task1 in range(0, self.n_tasks):
            if self.current_task_station[task1] == selected_station:
                earliest_station = 0
                for k in range(0, len(self.predecessors[task1])):
                    j = self.predecessors[task1][k]
                    if self.current_task_station[j] > earliest_station:
                        earliest_station = self.current_task_station[j]
                latest_station = self.n_stations - 1
                for k in range(0, len(self.successors[task1])):
                    j = self.successors[task1][k]
                    if self.current_task_station[j] < latest_station:
                        latest_station = self.current_task_station[j]
                if earliest_station != latest_station:
                    station1 = self.current_task_station[task1]
                    station2 = earliest_station + random.randint(0, latest_station - earliest_station)
                    if station2 == station1:
                        station2 = earliest_station + random.randint(0, latest_station - earliest_station)
                    if station2 == station1:
                        station2 = earliest_station + random.randint(0, latest_station - earliest_station)
                    if station2 == station1:
                        station2 = earliest_station + random.randint(0, latest_station - earliest_station)
                    self.current_station_time[station1] = self.current_station_time[station1] - self.t[task1]
                    self.current_station_time[station2] = self.current_station_time[station2] + self.t[task1]
                    self.current_task_station[task1] = station2
        if selected_station < self.n_stations - 1:
            for i in range(0, self.n_tasks):
                if self.current_task_station[i] > selected_station:
                    self.current_task_station[i] = self.current_task_station[i] - 1
        else:
            for i in range(0, self.n_tasks):
                if self.current_task_station[i] == selected_station:
                    self.current_task_station[i] = self.current_task_station[i] - 1
        self.n_stations = self.n_stations - 1
        self.current_station_time = [0 for _ in range(0, self.n_stations)]
        for i in range(0, self.n_tasks):
            self.current_station_time[self.current_task_station[i]] = \
                self.current_station_time[self.current_task_station[i]] + self.t[i]
        self.current_ct = max(self.current_station_time)
        current_score = 0
        for j in range(0, self.n_stations):
            if self.current_station_time[j] > self.ct:
                current_score = current_score + (self.current_station_time[j] - self.ct) * \
                                (self.current_station_time[j] - self.ct)
        self.tabu_list = np.zeros((self.n_tasks, self.n_tasks))

    def check_assignment(self, task_station: List[int], n_stations: int, max_cycle_time: float) -> int:
        """Check if task assignment is valid."""
        self.new_station_time = [0 for _ in range(0, self.n_stations)]
        for i in range(0, self.n_tasks):
            j = task_station[i]
            self.new_station_time[j] = self.new_station_time[j] + self.t[i]
            for k in range(0, len(self.successors[i])):
                j = self.successors[i][k]
                if task_station[j] < task_station[i]:
                    print("Successor of %d assigned to an earlier station" % i)
                    return 0
        for j in range(0, n_stations):
            if self.new_station_time[j] > max_cycle_time:
                print("cycle time is violated")
                return 0
            if self.current_station_time[j] != self.new_station_time[j]:
                print("current_station_time are incorrect")
                return 0
        return 1

    def evolve(self) -> None:
        if self.current_ct <= self.ct:
            self.reduce_station()
        station_select_set = []
        station_select_probability = []
        if random.uniform(0.0, 1.0) < 0.9:
            for j in range(0, self.n_stations):
                if self.current_station_time[j] > self.ct:
                    station_select_probability.append(
                        (self.current_station_time[j] - self.ct) * (
                                self.current_station_time[j] - self.ct))
                    station_select_set.append(j)
            if len(station_select_probability):
                selected_station = self.roulette_wheel_selection.return_element_from_probabilities(
                    station_select_probability, station_select_set)
            else:
                selected_station = random.randint(0, self.n_stations - 1)
            station_select_probability.clear()
            station_select_set.clear()
        else:
            selected_station = random.randint(0, self.n_stations - 1)
        best_insert_ct = 1.0e30
        best_insert_score = 1.0e30
        best_swap_ct = 1.0e30
        best_swap_score = 1.0e30
        best_insert = []
        best_swap = []
        self.new_task_station = copy.deepcopy(self.current_task_station)
        self.new_station_time = copy.deepcopy(self.current_station_time)
        for task1 in range(0, self.n_tasks):
            if self.current_task_station[task1] == selected_station:
                self.earliest_task_station[task1] = 0
                for k in range(0, len(self.predecessors[task1])):
                    j = self.predecessors[task1][k]
                    if self.current_task_station[j] > self.earliest_task_station[task1]:
                        self.earliest_task_station[task1] = self.current_task_station[j]
                self.latest_task_station[task1] = self.n_stations - 1
                for k in range(0, len(self.successors[task1])):
                    j = self.successors[task1][k]
                    if self.current_task_station[j] < self.latest_task_station[task1]:
                        self.latest_task_station[task1] = self.current_task_station[j]
                if self.earliest_task_station[task1] != self.latest_task_station[task1]:
                    for station2 in range(self.earliest_task_station[task1], self.latest_task_station[task1] + 1):
                        station1 = self.new_task_station[task1]
                        if station2 == station1:
                            continue
                        self.new_station_time[station1] = self.new_station_time[station1] - self.t[task1]
                        self.new_station_time[station2] = self.new_station_time[station2] + self.t[task1]
                        self.new_task_station[task1] = station2
                        new_ct = max(self.new_station_time)
                        new_score = 0
                        for j in range(0, self.n_stations):
                            if self.new_station_time[j] > self.ct:
                                new_score = new_score + (self.new_station_time[j] - self.ct) * (
                                        self.new_station_time[j] - self.ct)
                        if self.tabu_list[task1][station2] == 0 or new_ct <= self.ct:
                            if new_score < best_insert_score:
                                best_insert = [task1, station1, station2]
                                best_insert_ct = new_ct
                                best_insert_score = new_score
                        self.new_station_time[station1] = self.new_station_time[station1] + self.t[task1]
                        self.new_station_time[station2] = self.new_station_time[station2] - self.t[task1]
                        self.new_task_station[task1] = station1
                for task2 in range(0, self.n_tasks):
                    if self.all_precedence_matrix[task2][task1] == 0 and \
                            self.all_precedence_matrix[task1][task2] == 0 \
                            and task2 != task1:
                        if self.earliest_task_station[task1] <= self.current_task_station[task2] <= \
                                self.latest_task_station[task1] and \
                                self.current_task_station[task2] != self.current_task_station[task1]:
                            self.earliest_task_station[task2] = 0
                            for k in range(0, len(self.predecessors[task2])):
                                j = self.predecessors[task2][k]
                                if self.current_task_station[j] > self.earliest_task_station[task2]:
                                    self.earliest_task_station[task2] = self.current_task_station[j]
                            self.latest_task_station[task2] = self.n_stations - 1
                            for k in range(0, len(self.successors[task2])):
                                j = self.successors[task2][k]
                                if self.current_task_station[j] < self.latest_task_station[task2]:
                                    self.latest_task_station[task2] = self.current_task_station[j]
                            if self.earliest_task_station[task2] <= self.current_task_station[task1] <= \
                                    self.latest_task_station[task2]:
                                station1 = self.new_task_station[task1]
                                station2 = self.new_task_station[task2]
                                if station2 == station1:
                                    continue
                                self.new_station_time[station1] = self.new_station_time[station1] + self.t[
                                    task2] - self.t[task1]
                                self.new_station_time[station2] = self.new_station_time[station2] + self.t[
                                    task1] - self.t[task2]
                                self.new_task_station[task1] = station2
                                self.new_task_station[task2] = station1
                                new_ct = max(self.new_station_time)
                                new_score = 0
                                for j in range(0, self.n_stations):
                                    if self.new_station_time[j] > self.ct:
                                        new_score = new_score + (self.new_station_time[j] - self.ct) * (
                                                self.new_station_time[j] - self.ct)
                                if (self.tabu_list[task1][station2] == 0 and self.tabu_list[task2][station1] == 0) or \
                                        new_ct <= self.ct:
                                    if new_score < best_swap_score:
                                        best_swap = [task1, station1, task2, station2]
                                        best_swap_ct = new_ct
                                        best_swap_score = new_score
                                self.new_station_time[station1] = self.new_station_time[station1] + self.t[
                                    task1] - self.t[task2]
                                self.new_station_time[station2] = self.new_station_time[station2] + self.t[
                                    task2] - self.t[task1]
                                self.new_task_station[task1] = station1
                                self.new_task_station[task2] = station2
        if best_insert_score < 1.0e30 or best_swap_score < 1.0e30:
            if best_insert_score <= best_swap_score:
                task1 = best_insert[0]
                station1 = best_insert[1]
                station2 = best_insert[2]
                self.current_ct = best_insert_ct
                self.current_score = best_insert_score
                self.current_station_time[station1] = self.current_station_time[station1] - self.t[task1]
                self.current_station_time[station2] = self.current_station_time[station2] + self.t[task1]
                self.current_task_station[task1] = station2
                self.tabu_list[task1][station1] = self.tabu_length
            else:
                task1 = best_swap[0]
                station1 = best_swap[1]
                task2 = best_swap[2]
                station2 = best_swap[3]
                self.current_ct = best_swap_ct
                self.current_score = best_swap_score
                self.current_station_time[station1] = self.current_station_time[station1] + self.t[task2] - self.t[
                    task1]
                self.current_station_time[station2] = self.current_station_time[station2] + self.t[task1] - self.t[
                    task2]
                self.current_task_station[task1] = station2
                self.current_task_station[task2] = station1
                self.tabu_list[task1][station1] = self.tabu_length
                self.tabu_list[task2][station2] = self.tabu_length
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.tabu_list[i][j] >= 1:
                    self.tabu_list[i][j] = self.tabu_list[i][j] - 1
        if self.current_ct <= self.ct and self.solution.objectives[0] > self.n_stations:
            self.solution.objectives[0] = self.n_stations
            self.solution.variables = copy.deepcopy(self.current_task_station)
            if self.comparator.compare(self.solution, self.g_best) == -1:
                self.g_best = self.solution

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        pass

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
