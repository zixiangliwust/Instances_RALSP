# -*- coding: utf-8 -*-
import copy
import math
import random
import time
from typing import TypeVar, List, Optional

import numpy

from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.logger import get_logger
from evolu.util.comparator import Comparator
from evolu.util.termination_criterion import TerminationCriterion

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")

"""
Module: Simulated annealing utilizing station-oriented encoding
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP2ntergerSA(SingleObjectiveSwarmRoot[S, R]):
    """
    Simulated annealing utilizing station-oriented encoding
    References:
    [1] Otto, Alena, and Armin Scholl. 2011. "Incorporating ergonomic risks into assembly line balancing."
    European Journal of Operational Research 212 (2):277-86. doi: http://dx.doi.org/10.1016/j.ejor.2011.01.056.
    """

    def __init__(self,
                 problem: Problem[S],
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ) -> None:
        """Initialize SALBP2 integer SA algorithm."""
        super(SALBP2ntergerSA, self).__init__(problem=problem)
        self.algorithm_name: str = "Simulated annealing utilizing station-oriented encoding"
        self.comparator: Comparator = comparator
        self.termination_criterion: TerminationCriterion = termination_criterion
        self.observable.register(termination_criterion)
        self.temperature_start: float = 1.0
        self.temperature: float = self.temperature_start
        self.sub_max_iteration: int = 5
        self.minimum_temperature: float = 0.000001
        self.cooling_rate: float = 0.95
        self.offspring_population_size: int = self.sub_max_iteration
        self.solution: Optional[Solution] = None
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
        self.n_stations: Optional[int] = None
        self.current_ct: Optional[float] = None
        self.current_score: Optional[float] = None

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

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def reduce_station(self):
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
        for _ in range(0, self.sub_max_iteration):
            if self.current_ct < self.ct:
                self.reduce_station()
            self.new_task_station = copy.deepcopy(self.current_task_station)
            self.new_station_time = copy.deepcopy(self.current_station_time)
            if random.random() < 0.5:
                task_list = []
                for i in range(0, self.n_tasks):
                    if self.current_station_time[self.current_task_station[i]] == self.current_ct:
                        task_list.append(i)
                task1 = task_list[random.randint(0, len(task_list) - 1)]
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
                    station1 = self.new_task_station[task1]
                    station2 = earliest_station + random.randint(0, latest_station - earliest_station)
                    self.new_station_time[station1] = self.new_station_time[station1] - self.t[task1]
                    self.new_station_time[station2] = self.new_station_time[station2] + self.t[task1]
                    self.new_task_station[task1] = station2
                    new_ct = max(self.new_station_time)
                    if new_ct <= self.current_ct or random.uniform(0.0, 1.0) < math.exp(
                            (self.current_ct - new_ct) / (self.temperature * self.current_ct)):
                        self.current_ct = new_ct
                        self.current_station_time[station1] = self.current_station_time[station1] - self.t[task1]
                        self.current_station_time[station2] = self.current_station_time[station2] + self.t[task1]
                        self.current_task_station[task1] = station2
            else:
                task_list = []
                for i in range(0, self.n_tasks):
                    if self.current_station_time[self.current_task_station[i]] == self.current_ct:
                        task_list.append(i)
                task1 = task_list[random.randint(0, len(task_list) - 1)]
                for i in range(0, self.n_tasks):
                    self.earliest_task_station[i] = 0
                    for k in range(0, len(self.predecessors[i])):
                        j = self.predecessors[i][k]
                        if self.current_task_station[j] > self.earliest_task_station[i]:
                            self.earliest_task_station[i] = self.current_task_station[j]
                    self.latest_task_station[i] = self.n_stations - 1
                    for k in range(0, len(self.successors[i])):
                        j = self.successors[i][k]
                        if self.current_task_station[j] < self.latest_task_station[i]:
                            self.latest_task_station[i] = self.current_task_station[j]
                task_list = []
                for i in range(0, self.n_tasks):
                    if self.all_precedence_matrix[i][task1] == 0 and \
                            self.all_precedence_matrix[task1][i] == 0 and i != task1:
                        if self.earliest_task_station[task1] <= self.current_task_station[i] <= \
                                self.latest_task_station[task1]:
                            if self.earliest_task_station[i] <= self.current_task_station[task1] <= \
                                    self.latest_task_station[i]:
                                task_list.append(i)
                if len(task_list) > 0:
                    task2 = task_list[random.randint(0, len(task_list) - 1)]
                    station1 = self.new_task_station[task1]
                    station2 = self.new_task_station[task2]
                    self.new_station_time[station1] = self.new_station_time[station1] + self.t[task2] - self.t[task1]
                    self.new_station_time[station2] = self.new_station_time[station2] + self.t[task1] - self.t[task2]
                    self.new_task_station[task1] = station2
                    self.new_task_station[task2] = station1
                    new_ct = max(self.new_station_time)
                    if new_ct <= self.current_ct or random.uniform(0.0, 1.0) < math.exp(
                            (self.current_ct - new_ct) / self.temperature):
                        self.current_ct = new_ct
                        self.current_station_time[station1] = self.current_station_time[station1] + self.t[
                            task2] - self.t[task1]
                        self.current_station_time[station2] = self.current_station_time[station2] + self.t[
                            task1] - self.t[task2]
                        self.current_task_station[task1] = station2
                        self.current_task_station[task2] = station1
            if self.current_ct <= self.ct and self.solution.objectives[0] > self.n_stations:
                self.solution.objectives[0] = self.n_stations
                self.solution.variables = copy.deepcopy(self.current_task_station)
                if self.comparator.compare(self.solution, self.g_best) == -1:
                    self.g_best = self.solution
        self.temperature *= self.cooling_rate

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def after_initialization(self):
        self.g_best = copy.deepcopy(self.solution)
        self.problem.g_best = self.g_best

    def after_evolve(self):
        pass

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
