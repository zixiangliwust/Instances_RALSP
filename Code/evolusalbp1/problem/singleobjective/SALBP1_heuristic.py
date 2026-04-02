import copy
import random
from typing import List
import numpy as np
from evolu.core.problem import Problem
from evolu.core.solution import Solution, FloatSolution, PermutationSolution, IntegerSolution
from evolu.operator.selection import RouletteWheelSelection


class SALBP1SimpleHeuristic:
    def __init__(self, problem: Problem) -> None:
        """Initialize SALBP1 simple heuristic solver."""
        self.problem: Problem = problem
        self.n_tasks: int = self.problem.n_tasks
        self.ct: int = self.problem.ct
        self.t: List[int] = self.problem.t
        self.imm_precedence_matrix: Any = self.problem.imm_precedence_matrix
        self.all_precedence_matrix: Any = self.problem.all_precedence_matrix
        self.predecessors: List[List[int]] = self.problem.predecessors
        self.successors: List[List[int]] = self.problem.successors
        self.n_successors: np.ndarray = self.problem.n_successors
        self.n_predecessors: np.ndarray = self.problem.n_predecessors
        self.positional_weight: List[float] = self.problem.positional_weight
        self.rev_positional_weight: List[float] = self.problem.rev_positional_weight
        self.hash_values: List[int] = self.problem.hash_values
        self.root_degrees: List[int] = self.problem.root_degrees
        self.rev_root_degrees: List[int] = self.problem.rev_root_degrees
        self.lb2_values: np.ndarray = self.problem.lb2_values
        self.lb3_values: np.ndarray = self.problem.lb3_values
        self.task_probability: List[float] = self.problem.task_probability
        self.rev_task_probability: List[float] = self.problem.rev_task_probability
        self.normalized_task_probability: List[float] = self.problem.normalized_task_probability
        self.rev_normalized_task_probability: List[float] = self.problem.rev_normalized_task_probability
        self.Hash_Size: int = self.problem.Hash_Size
        self.roulette_wheel_selection: RouletteWheelSelection = RouletteWheelSelection()
        self.alpha: float = 1.00
        self.beta: float = 2.00
        self.gamma: float = 0.00
        self.n_stations: int = 0
        self.degrees: List[int] = []
        self.rev_degrees: List[int] = []
        self.task_permutation: List[int] = []
        self.task_start_time: List[float] = []
        self.task_end_time: List[float] = []
        self.task_station: List[int] = []
        self.station_time: List[float] = []
        self.tasks_to_stations: List[List[int]] = []
        self.objectives: List[float] = []
        self.task_select_set: List[int] = []
        self.task_select_probability: List[float] = []
        self.task_probability: List[float] = [0.0 for _ in range(0, self.n_tasks)]
        self.rev_task_probability: List[float] = [0.0 for _ in range(0, self.n_tasks)]
        self.normalized_task_probability: List[float] = [0.0 for _ in range(0, self.n_tasks)]
        self.rev_normalized_task_probability: List[float] = [0.0 for _ in range(0, self.n_tasks)]
        max_successor_num = max(self.n_successors)
        max_predecessor_num = max(self.n_predecessors)
        for i in range(0, self.n_tasks):
            self.task_probability[i] = self.t[i] / (self.ct * 1.0) + self.n_successors[i] / (max_successor_num * 1.0)
        for i in range(0, self.n_tasks):
            self.rev_task_probability[i] = self.t[i] / (self.ct * 1.0) + self.n_predecessors[i] / (
                    max_predecessor_num * 1.0)
        min_probability = min(self.task_probability)
        max_probability = max(self.task_probability)
        for i in range(0, self.n_tasks):
            self.normalized_task_probability[i] = (self.task_probability[i] - min_probability + 1) / max_probability
        min_probability = min(self.rev_task_probability)
        max_probability = max(self.rev_task_probability)
        for i in range(0, self.n_tasks):
            self.rev_normalized_task_probability[i] = (self.rev_task_probability[
                                                           i] - min_probability + 1) / max_probability

    def initialize_decoding(self) -> None:
        """Initialize decoding variables."""
        self.degrees = copy.deepcopy(self.root_degrees)
        self.rev_degrees = copy.deepcopy(self.rev_root_degrees)
        self.task_station = [0 for _ in range(0, self.n_tasks)]
        self.task_permutation = []
        self.task_start_time = []
        self.task_end_time = []
        self.station_time = []
        self.tasks_to_stations = []
        self.objectives = []

    def obtain_scores(self) -> None:
        """Calculate and store objective scores."""
        self.objectives.append(self.n_stations + 1)
        self.objectives.append(self.n_stations + 1 + (self.station_time[self.n_stations - 1] +
                                                      self.station_time[self.n_stations]) / (2.0 * self.ct))

    def random_search_heuristic1(self):
        self.initialize_decoding()
        sub_hash_value = 0
        self.task_permutation = list(range(0, self.n_tasks))
        random.shuffle(self.task_permutation)
        unassigned_task_set = copy.deepcopy(self.task_permutation)
        assigned_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                selected_task = -1
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        selected_task = i
                        break
                if selected_task == -1:
                    break
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def random_search_heuristic2(self) -> PermutationSolution:
        """Random search heuristic method 2."""
        self.initialize_decoding()
        sub_hash_value = 0
        self.task_permutation = list(range(0, self.n_tasks))
        unassigned_task_set = copy.deepcopy(self.task_permutation)
        assigned_task_set = []
        eligible_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    for i in eligible_task_set:
                        self.task_select_set.append(i)
                        self.task_select_probability.append(1.0)
                    selected_task = self.roulette_wheel_selection.return_element_from_probabilities(
                        self.task_select_probability, self.task_select_set)
                    self.task_select_set.clear()
                    self.task_select_probability.clear()
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def random_task_priority_heuristic(self):
        self.initialize_decoding()
        sub_hash_value = 0
        if self.n_tasks < 1000:
            self.alpha = 25
        else:
            self.alpha = 45
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    min_priority, max_priority = pow(10, 8), 0.0
                    switcher = np.random.randint(0, 4)
                    if switcher == 0:
                        for i in eligible_task_set:
                            min_priority = min(min_priority, self.t[i])
                            max_priority = max(max_priority, self.t[i])
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            if min_priority != max_priority:
                                self.task_select_probability.append(
                                    pow(1 + (self.t[i] - min_priority) / (max_priority - min_priority), self.alpha))
                            else:
                                self.task_select_probability.append(1.0)
                    elif switcher == 1:
                        for i in eligible_task_set:
                            min_priority = min(min_priority, self.n_successors[i])
                            max_priority = max(max_priority, self.n_successors[i])
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            if min_priority != max_priority:
                                self.task_select_probability.append(
                                    pow(1 + (self.n_successors[i] - min_priority) / (max_priority - min_priority),
                                        self.alpha))
                            else:
                                self.task_select_probability.append(1.0)
                    elif switcher == 2:
                        for i in eligible_task_set:
                            min_priority = min(min_priority, len(self.successors[i]))
                            max_priority = max(max_priority, len(self.successors[i]))
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            if min_priority != max_priority:
                                self.task_select_probability.append(
                                    pow(1 + (len(self.successors[i]) - min_priority) / (max_priority - min_priority),
                                        self.alpha))
                            else:
                                self.task_select_probability.append(1.0)
                    elif switcher == 3:
                        for i in eligible_task_set:
                            min_priority = min(min_priority, self.positional_weight[i])
                            max_priority = max(max_priority, self.positional_weight[i])
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            if min_priority != max_priority:
                                self.task_select_probability.append(
                                    pow(1 + (self.positional_weight[i] - min_priority) / (max_priority - min_priority),
                                        self.alpha))
                            else:
                                self.task_select_probability.append(1.0)
                    selected_task = self.roulette_wheel_selection.return_element_from_probabilities(
                        self.task_select_probability, self.task_select_set)
                    self.task_select_set.clear()
                    self.task_select_probability.clear()
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def longest_operation_time_heuristic(self) -> PermutationSolution:
        """Longest operation time heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if self.t[selected_task] < self.t[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def shortest_operation_time_heuristic(self) -> PermutationSolution:
        """Shortest operation time heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if self.t[selected_task] > self.t[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def maximum_total_number_of_successors_heuristic(self) -> PermutationSolution:
        """Maximum total number of successors heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if self.n_successors[selected_task] < self.n_successors[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def minimum_total_number_of_successors_heuristic(self) -> PermutationSolution:
        """Minimum total number of successors heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if self.n_successors[selected_task] > self.n_successors[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def maximum_total_operation_time_of_successors_heuristic(self) -> PermutationSolution:
        """Maximum total operation time of successors heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        total_operation_time_of_successors = [0.0 for _ in range(0, self.n_tasks)]
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j] == 1:
                    total_operation_time_of_successors[i] += self.t[j]
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if total_operation_time_of_successors[selected_task] < total_operation_time_of_successors[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def minimum_total_operation_time_of_successors_heuristic(self) -> PermutationSolution:
        """Minimum total operation time of successors heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        total_operation_time_of_successors = [0.0 for _ in range(0, self.n_tasks)]
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j] == 1:
                    total_operation_time_of_successors[i] += self.t[j]
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if total_operation_time_of_successors[selected_task] > total_operation_time_of_successors[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def maximum_ranked_positional_weight_heuristic(self) -> PermutationSolution:
        """Maximum ranked positional weight heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if self.positional_weight[selected_task] < self.positional_weight[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def maximum_average_ranked_positional_weight_heuristic(self) -> PermutationSolution:
        """Maximum average ranked positional weight heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        average_ranked_positional_weight = [0.0 for _ in range(0, self.n_tasks)]
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j] == 1:
                    average_ranked_positional_weight[i] += self.positional_weight[j]
            average_ranked_positional_weight[i] = average_ranked_positional_weight[i] / (
                    (self.n_successors[i] + 1) * 1.0)
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if average_ranked_positional_weight[selected_task] < average_ranked_positional_weight[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def maximum_weighted_operation_time_and_successors_heuristic(self) -> PermutationSolution:
        """Maximum weighted operation time and successors heuristic."""
        self.initialize_decoding()
        sub_hash_value = 0
        unassigned_task_set = list(range(0, self.n_tasks))
        assigned_task_set = []
        eligible_task_set = []
        elite_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    selected_task = eligible_task_set[0]
                    for i in eligible_task_set[1:]:
                        if self.task_probability[selected_task] < self.task_probability[i]:
                            selected_task = i
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.task_permutation.append(selected_task)
                self.degrees[selected_task] = -1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()
        self.obtain_scores()
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.problem.number_of_objectives
        )
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution
