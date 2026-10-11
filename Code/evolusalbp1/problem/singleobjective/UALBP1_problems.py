import copy
import random
from typing import List, Optional
import numpy as np
from evolu.operator.selection import RouletteWheelSelection
from evolu.core.solution import FloatSolution, PermutationSolution, IntegerSolution
from evolusalbp1.problem.singleobjective.SALBP1_problems import ALBP1Base
from evolusalbp1.problem.singleobjective.SALBP1_lower_bounding import SALBP1LowerBounding


class UALBProblemI(ALBP1Base):
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize UALBP1 problem instance."""
        super(UALBProblemI, self).__init__()
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.seed = 3.1567
        self.search_direction = 1
        self.reverse_direction = 0
        self.Hash_Size = 200000033
        print("Input problem and precess the problem data")
        self.__read_instance_from_file(file_path, file_name)
        self.feasible_task_permutation: List[int] = []
        self.real_task_permutation: List[int] = [-1 for _ in range(0, self.n_tasks)]
        self.task_side: List[int] = [-1 for _ in range(0, self.n_tasks)]
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
        self.alpha: float = 1.00
        self.beta: float = 2.00
        self.gamma: float = 0.00
        self.n_stations: int = 0
        self.degrees: List[int] = []
        self.rev_degrees: List[int] = []
        self.task_permutation: List[int] = []
        self.task_station: List[int] = []
        self.station_time: List[float] = []
        self.tasks_to_stations: List[List[int]] = []
        self.objectives: List[float] = []
        self.task_select_set: List[int] = []
        self.task_select_probability: List[float] = []
        self.roulette_wheel_selection: RouletteWheelSelection = RouletteWheelSelection()

    def obtain_low_bound(self) -> None:
        """Calculate and set the lower bound for number of stations."""
        lower_bounding = SALBP1LowerBounding(self)
        list_of_items = range(0, self.n_tasks)
        self.root_ns_lb = lower_bounding.calculate_lb1(list_of_items)
        lb2 = lower_bounding.calculate_lb2(list_of_items)
        lb3 = lower_bounding.calculate_lb3(list_of_items)
        self.root_ns_lb = max(self.root_ns_lb, lb2, lb3)
        print("root_ns_lb= %d" % self.root_ns_lb)

    def __read_instance_from_file(self, file_path: str, file_name: str) -> None:
        """Read problem instance from file."""
        self.file_path = file_path
        self.file_name = file_name
        file_path_and_name = file_path + file_name
        print("The selected problem is: " + str(file_name))
        file = open(file_path_and_name, "r")
        lines = file.readlines()
        file.close()
        for line in lines:  # delete empty lines
            if line == '\n':
                lines.remove(line)
        for line in lines:  # delete empty lines
            if line == '\n':
                lines.remove(line)
        for line in lines:  # delete comment line
            if line[0] == '<':
                lines.remove(line)
        line_index = 0
        self.n_tasks = int(lines[line_index])
        self.number_of_variables = self.n_tasks
        print("task number= %d" % self.n_tasks)
        line_index = line_index + 1
        self.ct = int(lines[line_index])
        print("cycle time= %d" % self.ct)
        line_index = line_index + 1
        order_strength = None
        order_strength = float(lines[line_index].replace(',', '.'))
        line_index = line_index + 1
        self.t = []
        for i in range(0, self.n_tasks):
            line = lines[line_index].split(' ')
            line = [int(j) for j in line]
            self.t.append(line[1])
            line_index = line_index + 1
        self.imm_precedence = []
        self.imm_precedence_matrix = np.zeros((self.n_tasks, self.n_tasks))
        for i in range(line_index, len(lines)):
            line = lines[line_index].split(',')
            line = [int(j) for j in line]
            self.imm_precedence.append((line[0] - 1, line[1] - 1))
            self.imm_precedence_matrix[line[0] - 1][line[1] - 1] = 1
            line_index = line_index + 1
        self.seed = 3.1567
        self.hash_values = []
        for i in range(0, self.n_tasks):
            self.hash_values.append(self.randomize(self.Hash_Size) - 1)
        # self.input_true_low_bound()
        self.close_precedence()
        self.find_successors()
        self.compute_positional_weights()
        self.compute_lb2_values()
        self.compute_lb3_values()
        self.obtain_low_bound()

    def input_true_low_bound(self):
        self.root_ns_lb_true = 0
        file = open("E:/Benchmarks/ALBPInstances/UALBP1_lb.txt", "r")
        lines = file.readlines()
        file.close()
        for line in lines:  # delete empty lines
            if line == '\n':
                lines.remove(line)
        for line in lines:  # delete comment line
            if line[0] == '<':
                lines.remove(line)
        for line_index in range(0, len(lines)):
            line = lines[line_index].split(' ')
            line = [int(j) for j in line]
            if line[0] == self.n_tasks and line[1] == self.ct and \
                    line[2:-1] == self.t[:min(10, self.n_tasks)]:
                self.root_ns_lb_true = line[-1]
                print("root_ns_lb_true= %d" % self.root_ns_lb_true)
                break

    def initialize_decoding(self) -> None:
        """Initialize decoding variables."""
        self.degrees = copy.deepcopy(self.root_degrees)
        self.rev_degrees = copy.deepcopy(self.rev_root_degrees)
        self.task_station = [0 for _ in range(0, self.n_tasks)]
        self.task_permutation = []
        self.station_time = []
        self.tasks_to_stations = []
        self.objectives = []
        self.feasible_task_permutation = []
        self.real_task_permutation = [-1 for _ in range(0, self.n_tasks)]
        self.task_side = [-1 for _ in range(0, self.n_tasks)]

    def decoding1(self) -> None:
        """Decoding method 1 for UALBP1."""
        sub_hash_value = 0
        unassigned_task_set = copy.deepcopy(self.task_permutation)
        selected_task = -1
        while len(unassigned_task_set) > 0:
            for i in unassigned_task_set:
                if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                        self.degrees[i] == 0 or self.rev_degrees[i] == 0):
                    selected_task = i
                    break
            unassigned_task_set.remove(selected_task)
            self.feasible_task_permutation.append(selected_task)
            if self.degrees[selected_task] == 0:
                self.task_side[selected_task] = 1
            else:
                self.task_side[selected_task] = 2
            if self.task_side[selected_task] == 1:
                for i in range(0, self.n_tasks):
                    if self.real_task_permutation[i] == -1:
                        self.real_task_permutation[i] = selected_task
                        break
            else:
                for i in reversed(range(0, self.n_tasks)):
                    if self.real_task_permutation[i] == -1:
                        self.real_task_permutation[i] = selected_task
                        break
            if self.task_side[selected_task] == 1:
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] -= 1
            else:
                self.rev_degrees[selected_task] -= 1
                for j in self.predecessors[selected_task]:
                    self.rev_degrees[j] -= 1
        unassigned_task_set = copy.deepcopy(self.feasible_task_permutation)
        assigned_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                if self.t[unassigned_task_set[0]] + self.station_time[self.n_stations] <= self.ct:
                    selected_task = unassigned_task_set[0]
                    self.station_time[self.n_stations] += self.t[selected_task]
                    unassigned_task_set.remove(selected_task)
                    assigned_task_set.append(selected_task)
                    sub_hash_value += self.hash_values[selected_task]
                    if sub_hash_value > self.Hash_Size:
                        sub_hash_value = sub_hash_value % self.Hash_Size
                    self.task_station[selected_task] = self.n_stations
                    assigned_tasks.append(selected_task)
                else:
                    break
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()

    def decoding2(self) -> None:
        """Decoding method 2 for UALBP1."""
        sub_hash_value = 0
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
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
                        selected_task = i
                        break
                if selected_task == -1:
                    break
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()

    def decoding3(self):
        sub_hash_value = 0
        unassigned_task_set = copy.deepcopy(self.task_permutation)
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
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
                        eligible_task_set.append(i)
                for i in eligible_task_set:
                    if self.t[i] + self.station_time[self.n_stations] == self.ct:
                        elite_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                elif len(elite_task_set) == 0:
                    selected_task = eligible_task_set[0]
                else:
                    selected_task = elite_task_set[0]
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
                eligible_task_set.clear()
                elite_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()
        elite_task_set.clear()

    def random_decoding1(self) -> None:
        """Random decoding method 1."""
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
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
                        selected_task = i
                        break
                if selected_task == -1:
                    break
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[
                    selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()

    def random_decoding2(self) -> None:
        """Random decoding method 2."""
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
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
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
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def heuristic_decoding(self) -> None:
        """Heuristic decoding method."""
        sub_hash_value = 0
        if self.n_tasks < 1000:
            self.alpha = 25
        else:
            self.alpha = 45
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
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
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
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def obtain_scores(self) -> None:
        """Calculate and store objective scores."""
        self.objectives.append(self.n_stations + 1)
        self.objectives.append(self.n_stations + 1 + (self.station_time[self.n_stations - 1] +
                                                      self.station_time[self.n_stations]) / (2.0 * self.ct))

    def create_evaluate_random_solution(self) -> PermutationSolution:
        """Create and evaluate a random solution."""
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives
        )
        self.initialize_decoding()
        self.random_decoding2()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def create_evaluate_heuristic_solution(self) -> PermutationSolution:
        """Create and evaluate a heuristic solution."""
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives
        )
        self.initialize_decoding()
        self.heuristic_decoding()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def create_evaluate_random_integer_solution(self) -> IntegerSolution:
        """Create and evaluate a random integer solution."""
        new_solution = IntegerSolution(lower_bound=[0 for _ in range(self.n_tasks)],
                                       upper_bound=[self.n_tasks for _ in range(self.n_tasks)],
                                       number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives,
                                       number_of_constraints=self.number_of_constraints
                                       )
        self.initialize_decoding()
        self.random_decoding2()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = copy.deepcopy(self.task_station)
        return new_solution

    def create_evaluate_heuristic_integer_solution(self) -> IntegerSolution:
        """Create and evaluate a heuristic integer solution."""
        new_solution = IntegerSolution(lower_bound=[0 for _ in range(self.n_tasks)],
                                       upper_bound=[self.n_tasks for _ in range(self.n_tasks)],
                                       number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives,
                                       number_of_constraints=self.number_of_constraints
                                       )
        self.initialize_decoding()
        self.heuristic_decoding()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = copy.deepcopy(self.task_station)
        return new_solution

    def get_name(self) -> str:
        """Get problem name."""
        return "Simple assembly line balancing"


class FloatUALBP1(UALBProblemI):
    def __init__(self, file_path, file_name):
        super(FloatUALBP1, self).__init__(file_path=file_path, file_name=file_name)
        self.lower_bound = [0.0 for _ in range(self.n_tasks)]
        self.upper_bound = [1.0 for _ in range(self.n_tasks)]
        self.solution_type = "Float"

    def create_variables(self, lower_bound=None, upper_bound=None):
        if lower_bound is None or upper_bound is None:
            variables = [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0)
                         for i in range(self.n_tasks)]
        else:
            variables = [random.uniform(lower_bound[i] * 1.0, upper_bound[i] * 1.0)
                         for i in range(self.n_tasks)]
        return variables

    def create_solution(self) -> FloatSolution:
        """Create a new float solution."""
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, self.n_tasks, self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        """Evaluate a float solution."""
        self.initialize_decoding()
        solution = self.remedy_solution(solution)
        task_variables = solution.variables
        self.task_permutation = list(range(0, self.n_tasks))
        for i in range(0, self.n_tasks - 1):
            for j in range(i + 1, self.n_tasks):
                if task_variables[i] < task_variables[j]:
                    task_variables[i], task_variables[j] = task_variables[j], task_variables[i]
                    self.task_permutation[i], self.task_permutation[j] = self.task_permutation[j], \
                        self.task_permutation[i]
        self.decoding2()
        self.obtain_scores()
        solution.objectives = copy.deepcopy(self.objectives)
        solution.fitness = self.objectives[0]
        return solution

    def remedy_solution(self, solution: FloatSolution) -> FloatSolution:
        """Remedy a float solution to satisfy bounds."""
        variables = copy.deepcopy(solution.variables)
        for i in range(0, len(variables)):
            if variables[i] < self.lower_bound[i]:
                variables[i] = self.lower_bound[i]
            if variables[i] > self.upper_bound[i]:
                variables[i] = self.upper_bound[i]
        solution.variables = variables
        return solution


class PermutationUALBP1(UALBProblemI):
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize Permutation UALBP1 problem."""
        super(PermutationUALBP1, self).__init__(file_path=file_path, file_name=file_name)
        self.solution_type: str = "Permutation"

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives
        )
        new_solution.variables = random.sample(range(self.n_tasks), k=self.n_tasks)
        return new_solution

    def evaluate_solution(self, solution: PermutationSolution) -> PermutationSolution:
        self.initialize_decoding()
        self.task_permutation = solution.variables
        self.decoding2()
        self.obtain_scores()
        solution.fitness = self.objectives[0]
        solution.objectives = copy.deepcopy(self.objectives)
        return solution


class AntUALBP1(UALBProblemI):
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize Ant UALBP1 problem for ACO algorithm."""
        super(AntUALBP1, self).__init__(file_path=file_path, file_name=file_name)
        self.first_task_prob: List[float] = []
        self.task_task_prob: List[List[float]] = []
        self.task_station_prob: List[List[float]] = []

    def probability_decoding1(self) -> None:
        """Perform probability decoding method 1."""
        sub_hash_value = 0
        self.task_permutation = list(range(0, self.n_tasks))
        unassigned_task_set = copy.deepcopy(self.task_permutation)
        eligible_task_set = []
        assigned_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
                        eligible_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    if len(assigned_task_set) == 0:
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            if self.degrees[i] == 0:
                                self.task_select_probability.append(pow(self.first_task_prob[i], self.alpha) *
                                                                    pow(self.task_probability[i], self.beta))
                            else:
                                self.task_select_probability.append(pow(self.first_task_prob[i], self.alpha) *
                                                                    pow(self.rev_task_probability[i], self.beta))
                    else:
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            if self.degrees[i] == 0:
                                self.task_select_probability.append(
                                    pow(self.task_task_prob[assigned_task_set[-1]][i], self.alpha) * pow(
                                        self.task_probability[i], self.beta))
                            else:
                                self.task_select_probability.append(
                                    pow(self.task_task_prob[assigned_task_set[-1]][i], self.alpha) * pow(
                                        self.rev_task_probability[i], self.beta))
                    uniform_num = np.random.uniform(0.0, 1.0)
                    if uniform_num < 0.6:
                        selected_task = self.task_select_set[0]
                        best_task_select_probability = self.task_select_probability[0]
                        for i in range(1, len(self.task_select_probability)):
                            if best_task_select_probability < self.task_select_probability[i]:
                                best_task_select_probability = self.task_select_probability[i]
                                selected_task = self.task_select_set[i]
                    elif uniform_num < 0.9:
                        selected_task = self.roulette_wheel_selection.return_element_from_probabilities(
                            self.task_select_probability, self.task_select_set)
                    else:
                        self.task_select_probability = [1.0 for _ in range(0, len(self.task_select_probability))]
                        selected_task = self.roulette_wheel_selection.return_element_from_probabilities(
                            self.task_select_probability, self.task_select_set)
                    self.task_select_set.clear()
                    self.task_select_probability.clear()
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        self.task_permutation = copy.deepcopy(assigned_task_set)
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def probability_decoding2(self) -> None:
        """Perform probability decoding method 2."""
        sub_hash_value = 0
        self.task_permutation = list(range(0, self.n_tasks))
        unassigned_task_set = copy.deepcopy(self.task_permutation)
        eligible_task_set = []
        assigned_task_set = []
        assigned_tasks = []
        self.n_stations = -1
        while len(unassigned_task_set) > 0:
            self.station_time.append(0)
            self.n_stations = self.n_stations + 1
            while len(unassigned_task_set) > 0:
                for i in unassigned_task_set:
                    if (self.degrees[i] >= 0 and self.rev_degrees[i] >= 0) and (
                            self.degrees[i] == 0 or self.rev_degrees[i] == 0) and self.station_time[self.n_stations] + \
                            self.t[i] <= self.ct:
                        eligible_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    for i in eligible_task_set:
                        sum_task_station_prob = 0.0
                        for j in range(0, self.n_stations + 1):
                            sum_task_station_prob = sum_task_station_prob + self.task_station_prob[i][j]
                        self.task_select_set.append(i)
                        self.task_select_probability.append(pow(sum_task_station_prob, self.alpha) *
                                                            pow(self.task_probability[i], self.beta))
                    uniform_num = np.random.uniform(0.0, 1.0)
                    if uniform_num < 0.6:
                        selected_task = self.task_select_set[0]
                        best_task_select_probability = self.task_select_probability[0]
                        for i in range(1, len(self.task_select_probability)):
                            if best_task_select_probability < self.task_select_probability[i]:
                                best_task_select_probability = self.task_select_probability[i]
                                selected_task = self.task_select_set[i]
                    elif uniform_num < 0.9:
                        selected_task = self.roulette_wheel_selection.return_element_from_probabilities(
                            self.task_select_probability, self.task_select_set)
                    else:
                        self.task_select_probability = [1.0 for _ in range(0, len(self.task_select_probability))]
                        selected_task = self.roulette_wheel_selection.return_element_from_probabilities(
                            self.task_select_probability, self.task_select_set)
                    self.task_select_set.clear()
                    self.task_select_probability.clear()
                self.station_time[self.n_stations] = self.station_time[self.n_stations] + self.t[selected_task]
                unassigned_task_set.remove(selected_task)
                assigned_task_set.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                if self.degrees[selected_task] == 0:
                    self.task_side[selected_task] = 1
                else:
                    self.task_side[selected_task] = 2
                if self.task_side[selected_task] == 1:
                    for i in range(0, self.n_tasks):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                else:
                    for i in reversed(range(0, self.n_tasks)):
                        if self.real_task_permutation[i] == -1:
                            self.real_task_permutation[i] = selected_task
                            break
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                if self.task_side[selected_task] == 1:
                    self.degrees[selected_task] -= 1
                    for j in self.successors[selected_task]:
                        self.degrees[j] = self.degrees[j] - 1
                else:
                    self.rev_degrees[selected_task] -= 1
                    for j in self.predecessors[selected_task]:
                        self.rev_degrees[j] = self.rev_degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        self.task_permutation = copy.deepcopy(assigned_task_set)
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def obtain_scores(self) -> None:
        """Calculate and store objective scores."""
        self.objectives.append(self.n_stations + 1)
        self.objectives.append(self.n_stations + 1 + (self.station_time[self.n_stations - 1] +
                                                      self.station_time[self.n_stations]) / (2.0 * self.ct))

    def create_solution1(self, first_task_prob_copy: List[float], task_task_prob_copy: List[List[float]]) -> PermutationSolution:
        """Create solution using probability decoding method 1."""
        self.first_task_prob = first_task_prob_copy
        self.task_task_prob = task_task_prob_copy
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )
        self.initialize_decoding()
        self.probability_decoding1()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = copy.deepcopy(self.task_permutation)
        return new_solution

    def create_solution2(self, task_station_prob_copy: List[List[float]]) -> IntegerSolution:
        """Create solution using probability decoding method 2."""
        self.task_station_prob = task_station_prob_copy
        new_solution = IntegerSolution(lower_bound=[0 for _ in range(self.n_tasks)],
                                       upper_bound=[self.n_tasks for _ in range(self.n_tasks)],
                                       number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives,
                                       number_of_constraints=self.number_of_constraints
                                       )
        self.initialize_decoding()
        self.probability_decoding2()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = copy.deepcopy(self.task_station)
        return new_solution

    def create_solution(self) -> PermutationSolution:
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives
        )
        new_solution.variables = random.sample(range(self.n_tasks), k=self.n_tasks)
        return new_solution

    def evaluate_solution(self, solution: PermutationSolution) -> PermutationSolution:
        self.initialize_decoding()
        self.task_permutation = solution.variables
        self.decoding2()
        self.obtain_scores()
        solution.fitness = self.objectives[0]
        solution.objectives = copy.deepcopy(self.objectives)
        return solution
