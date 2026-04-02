"""Simple Assembly Line Balancing Problem Type 1 (SALBP1) implementations.

This module provides the base class and concrete implementations for the
Simple Assembly Line Balancing Problem Type 1 (SALBP1). SALBP1 is a classical
optimization problem in manufacturing where tasks must be assigned to workstations
on an assembly line to minimize the number of workstations, given a fixed cycle time.

The problem considers:
- Task processing times
- Precedence constraints between tasks
- Cycle time constraints (maximum workload per workstation)
- Multiple solution representations (float, permutation, integer)

Classes:
    ALBP1Base: Base class for Assembly Line Balancing Problem Type 1
    SALBProblemI: Concrete SALBP1 problem implementation
    FloatSALBP1: SALBP1 with float solution representation
    PermutationSALBP1: SALBP1 with permutation solution representation
    AntSALBP1: SALBP1 with ant colony optimization solution representation
"""
import copy
import math
import random
from typing import List, Optional, Any
import numpy as np

from evolu.core.problem import Problem
from evolu.core.solution import FloatSolution, PermutationSolution, IntegerSolution
from evolu.operator.selection import RouletteWheelSelection
from evolusalbp1.problem.singleobjective.SALBP1_lower_bounding import SALBP1LowerBounding


class ALBP1Base(Problem):
    """Base class for Assembly Line Balancing Problem Type 1.
    
    This class provides the foundation for assembly line balancing problems,
    handling precedence constraints, positional weights, lower bounding,
    and solution decoding for various representations.
    
    Attributes:
        n_tasks (Optional[int]): Number of tasks in the problem.
        ct (Optional[int]): Cycle time (maximum workload per workstation).
        t (List[int]): Task processing times.
        imm_precedence (List[tuple]): Immediate precedence relations (pairs).
        imm_precedence_matrix (Optional[np.ndarray]): Immediate precedence matrix.
        all_precedence_matrix (Optional[np.ndarray]): Complete precedence matrix
            including transitive relations.
        predecessors (Optional[List[List[int]]]): Predecessor lists for each task.
        successors (Optional[List[List[int]]]): Successor lists for each task.
        positional_weight (List[float]): Forward positional weights.
        rev_positional_weight (List[float]): Reverse positional weights.
        root_ns_lb (Optional[int]): Lower bound on number of stations.
        seed (float): Random number generator seed.
        search_direction (int): Search direction (1=forward, 0=reverse).
    """
    
    def __init__(self) -> None:
        """Initialize ALBP1 base problem.
        
        Sets up the base structure for ALBP1 problems, initializing all
        necessary attributes for precedence handling, lower bounding, and
        solution evaluation.
        """
        super(ALBP1Base, self).__init__()
        self.problem_name: str = "ALBP1Base"
        self.n_tasks: Optional[int] = None
        self.ct: Optional[int] = None
        self.t: List[int] = []
        self.imm_precedence: List[tuple] = []
        self.imm_precedence_matrix: Optional[np.ndarray] = None
        self.all_precedence_matrix: Optional[np.ndarray] = None
        self.predecessors: Optional[List[List[int]]] = None
        self.successors: Optional[List[List[int]]] = None
        self.n_successors: np.ndarray = np.array([])
        self.n_predecessors: np.ndarray = np.array([])
        self.positional_weight: List[float] = []
        self.rev_positional_weight: List[float] = []
        self.hash_values: List[int] = []
        self.root_degrees: List[int] = []
        self.rev_root_degrees: List[int] = []
        self.lb2_values: np.ndarray = np.array([])
        self.lb3_values: np.ndarray = np.array([])
        self.seed: float = 3.1567
        self.search_direction: int = 1
        self.reverse_direction: int = 0
        self.root_ns_lb: Optional[int] = None
        self.root_ns_lb_true: Optional[int] = None
        self.Hash_Size: int = 200000033

    def ggubfs(self) -> float:
        """Generate random number using linear congruential generator.
        
        Implements the Park-Miller minimal standard random number generator
        (linear congruential generator) for generating pseudo-random numbers.
        This is used for randomized task selection in heuristics.
        
        Returns:
            float: A random number in the range [0, 1).
        
        Note:
            Uses the multiplier 16807 and modulus 2147483647 (2^31 - 1),
            which is a well-known good choice for LCG generators.
        """
        product = 16807.0 * self.seed
        div = int(product / 2147483647.0)
        self.seed = product - (div * 2147483647.0)
        return self.seed / 2147483648.0

    def randomize(self, n: int) -> int:
        """Generate random integer in range [1, n].
        
        Args:
            n (int): Upper bound for the random integer (inclusive).
        
        Returns:
            int: A random integer in the range [1, n].
        """
        return int(n * self.ggubfs()) + 1

    def close_precedence(self) -> None:
        """Calculate transitive closure of precedence matrix.
        
        Computes the complete precedence matrix including both direct and
        indirect (transitive) precedence relations. This is necessary for
        correctly handling precedence constraints during solution decoding.
        
        Note:
            This method modifies self.all_precedence_matrix in place by
            iteratively adding transitive relationships until no new
            precedence relations are found.
        """
        self.all_precedence_matrix = copy.deepcopy(self.imm_precedence_matrix)
        while 1:
            new_precedence = 0
            for i in range(0, self.n_tasks):
                for j in range(0, self.n_tasks):
                    for k in range(0, self.n_tasks):
                        if self.all_precedence_matrix[i][j] == 1 and self.all_precedence_matrix[j][k] == 1 \
                                and self.all_precedence_matrix[i][k] < 1:
                            self.all_precedence_matrix[i][k] = 1
                            new_precedence = 1
            if new_precedence == 0:
                break

    def select_search_direction(self) -> None:
        """Determine optimal search direction (forward or reverse).
        
        Analyzes the problem structure to decide whether solving in forward
        direction (task order) or reverse direction (reverse task order) would
        be more efficient. The decision is based on the distribution of earliest
        and latest possible station assignments for tasks.
        
        If reverse direction is selected, the precedence matrix is reversed
        and the transitive closure is recomputed.
        
        Note:
            This method may modify self.reverse_direction, call reverse_precedence(),
            and recompute all_precedence_matrix if reverse direction is chosen.
        """
        earliest_station, latest_station = [], []
        for j in range(0, self.n_tasks):
            f_time = self.t[j]
            r_time = self.t[j]
            for i in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j]:  # if task i precedes task j
                    f_time += self.t[i]
                if self.all_precedence_matrix[j][i]:  # if task j precedes task i
                    r_time += self.t[i]
            earliest_station.append(int(math.ceil(f_time / self.ct - 1.0)))
            latest_station.append(int(math.ceil(r_time / self.ct - 1.0)))
        f, r = 1.0, 1.0
        for m in range(0, 5):
            f_count, r_count = 0, 0
            for j in range(0, self.n_tasks):
                if earliest_station[j] <= m:
                    f_count = f_count + 1
                if latest_station[j] <= m:
                    r_count = r_count + 1
            f *= f_count
            r *= r_count
        if r < f:
            self.reverse_direction = 1
            print("running in reverse_direction: %.0f %.0f" % (f, r))
            self.reverse_precedence()
            self.close_precedence()
        else:
            self.reverse_direction = 0
            print("running forward: %.0f %.0f" % (f, r))

    # Reverse the immediate precedence relations and operation times;
    def reverse_precedence(self) -> None:
        """Reverse the immediate precedence relations and operation times."""
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks - i):
                self.imm_precedence_matrix[i][j], self.imm_precedence_matrix[self.n_tasks - 1 - j][
                    self.n_tasks - 1 - i] = \
                    self.imm_precedence_matrix[self.n_tasks - 1 - j][self.n_tasks - 1 - i], \
                        self.imm_precedence_matrix[i][j]
        imm_precedence_temp = copy.deepcopy(self.imm_precedence)
        self.imm_precedence.clear()
        for h, i in imm_precedence_temp:
            self.imm_precedence.append(
                (self.n_tasks - 1 - i, self.n_tasks - 1 - h))
        imm_precedence_temp.clear()
        i, j = 0, self.n_tasks - 1
        while i < j:
            self.t[i], self.t[j] = self.t[j], self.t[i]
            i = i + 1
            j = j - 1

    # Obtain the successors and predecessors of tasks;
    def find_successors(self) -> None:
        """Obtain the successors and predecessors of tasks."""
        self.predecessors = [[] for _ in range(0, self.n_tasks)]
        self.successors = [[] for _ in range(0, self.n_tasks)]
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.imm_precedence_matrix[j][i] == 1:
                    self.predecessors[i].append(j)
                if self.imm_precedence_matrix[i][j] == 1:
                    self.successors[i].append(j)
        self.root_degrees, self.rev_root_degrees = [], []
        for i in range(0, self.n_tasks):
            count = 0
            for j in range(0, self.n_tasks):
                if self.imm_precedence_matrix[j][i] == 1:
                    count = count + 1
            self.root_degrees.append(count)
        for i in range(0, self.n_tasks):
            count = 0
            for j in range(0, self.n_tasks):
                if self.imm_precedence_matrix[i][j] == 1:
                    count = count + 1
            self.rev_root_degrees.append(count)

    # Obtain the positional weights(the sum of operation times of the task and all its successors);
    def compute_positional_weights(self) -> None:
        """Compute positional weights (sum of operation times of task and all its successors)."""
        self.n_predecessors = np.zeros(self.n_tasks)
        self.n_successors = np.zeros(self.n_tasks)
        self.positional_weight = copy.deepcopy(self.t)
        self.rev_positional_weight = copy.deepcopy(self.t)
        for i in range(0, self.n_tasks):
            count = 0
            t_sum = 0
            for j in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j] == 1:
                    count = count + 1
                    t_sum = t_sum + self.t[j]
            self.n_successors[i] = count
            self.positional_weight[i] += t_sum
        for j in reversed(range(0, self.n_tasks)):
            count = 0
            t_sum = 0
            for i in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j] == 1:
                    count = count + 1
                    t_sum = t_sum + self.t[i]
            self.n_predecessors[j] = count
            self.rev_positional_weight[j] += t_sum

    def compute_lb2_values(self) -> None:
        """Compute LB2 values for all tasks."""
        self.lb2_values = np.zeros(self.n_tasks)
        for i in range(0, self.n_tasks):
            if self.t[i] > self.ct / 2.0:
                self.lb2_values[i] = 1
                continue
            if self.t[i] == self.ct / 2.0:
                self.lb2_values[i] = 0.5
                continue

    def compute_lb3_values(self) -> None:
        """Compute LB3 values for all tasks."""
        self.lb3_values = np.zeros(self.n_tasks)
        for i in range(0, self.n_tasks):
            if self.t[i] > 2.0 * self.ct / 3.0:
                self.lb3_values[i] = 1
                continue
            if self.t[i] == 2.0 * self.ct / 3.0:
                self.lb3_values[i] = 2 / 3.0
                continue
            if self.ct / 3.0 < self.t[i] < 2.0 * self.ct / 3.0:
                self.lb3_values[i] = 0.5
                continue
            if self.t[i] == self.ct / 3.0:
                self.lb3_values[i] = 1.0 / 3.0
                continue


class SALBProblemI(ALBP1Base):
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize SALBP1 problem instance."""
        super(SALBProblemI, self).__init__()
        self.problem_name = "SimpleALBP1"
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.seed = 3.1567
        self.search_direction = 1
        self.reverse_direction = 0
        self.Hash_Size = 200000033
        print("Input problem and precess the problem data")
        self.__read_instance_from_file(file_path, file_name)
        self.feasible_task_permutation = []
        self.alpha = 1.00
        self.beta = 2.00
        self.gamma = 0.00
        self.task_probability = [0.0 for _ in range(0, self.n_tasks)]
        self.rev_task_probability = [0.0 for _ in range(0, self.n_tasks)]
        self.normalized_task_probability = [0.0 for _ in range(0, self.n_tasks)]
        self.rev_normalized_task_probability = [0.0 for _ in range(0, self.n_tasks)]
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
        self.n_stations = 0
        self.degrees = []
        self.rev_degrees = []
        self.task_permutation = []
        self.task_station = []
        self.station_time = []
        self.tasks_to_stations = []
        self.objectives = []
        self.task_select_set = []
        self.task_select_probability = []
        self.roulette_wheel_selection = RouletteWheelSelection()

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
        # if self.search_direction == 1:
        #     self.select_search_direction()
        self.find_successors()
        self.compute_positional_weights()
        self.compute_lb2_values()
        self.compute_lb3_values()
        self.obtain_low_bound()

    def input_true_low_bound(self):
        self.root_ns_lb_true = 0
        file = open("E:/Benchmarks/ALBPInstances/SALBP1_lb.txt", "r")
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

    def initialize_decoding(self):
        self.degrees = copy.deepcopy(self.root_degrees)
        self.rev_degrees = copy.deepcopy(self.rev_root_degrees)
        self.task_station = [0 for _ in range(0, self.n_tasks)]
        self.task_permutation = []
        self.station_time = []
        self.tasks_to_stations = []
        self.objectives = []
        self.feasible_task_permutation = []

    def decoding1(self):
        sub_hash_value = 0
        unassigned_task_set = copy.deepcopy(self.task_permutation)
        selected_task = -1
        while len(unassigned_task_set) > 0:
            for i in unassigned_task_set:
                if self.degrees[i] == 0:
                    selected_task = i
                    break
            unassigned_task_set.remove(selected_task)
            self.feasible_task_permutation.append(selected_task)
            self.degrees[selected_task] -= 1
            for j in self.successors[selected_task]:
                self.degrees[j] = self.degrees[j] - 1
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

    def decoding2(self):
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
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
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
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
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
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
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

    def random_decoding1(self):
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
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()

    def random_decoding2(self):
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
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def heuristic_decoding(self):
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
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def ranked_positional_weight_heuristic_decoding(self):
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
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
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

    def obtain_scores(self):
        self.objectives.append(self.n_stations + 1)
        self.objectives.append(self.n_stations + 1 + (self.station_time[self.n_stations - 1] +
                                                      self.station_time[self.n_stations]) / (2.0 * self.ct))

    def create_evaluate_random_solution(self):
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

    def create_evaluate_heuristic_solution(self):
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

    def create_evaluate_ranked_positional_weight_heuristic_solution(self):
        new_solution = PermutationSolution(
            number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives
        )
        self.initialize_decoding()
        self.ranked_positional_weight_heuristic_decoding()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = self.task_permutation
        return new_solution

    def create_evaluate_random_integer_solution(self):
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

    def create_evaluate_heuristic_integer_solution(self):
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

    def create_evaluate_ranked_positional_weight_heuristic_integer_solution(self):
        new_solution = IntegerSolution(lower_bound=[0 for _ in range(self.n_tasks)],
                                       upper_bound=[self.n_tasks for _ in range(self.n_tasks)],
                                       number_of_variables=self.n_tasks, number_of_objectives=self.number_of_objectives,
                                       number_of_constraints=self.number_of_constraints
                                       )
        self.initialize_decoding()
        self.ranked_positional_weight_heuristic_decoding()
        self.obtain_scores()
        new_solution.survive_time = 0
        new_solution.objectives = copy.deepcopy(self.objectives)
        new_solution.variables = copy.deepcopy(self.task_station)
        return new_solution

    def get_name(self):
        return "Simple assembly line balancing"


class FloatSALBP1(SALBProblemI):
    def __init__(self, file_path, file_name):
        super(FloatSALBP1, self).__init__(file_path=file_path, file_name=file_name)
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

    def create_solution(self):
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, self.n_tasks, self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def evaluate_solution(self, solution):
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

    def remedy_solution(self, solution):
        variables = copy.deepcopy(solution.variables)
        for i in range(0, len(variables)):
            if variables[i] < self.lower_bound[i]:
                variables[i] = self.lower_bound[i]
            if variables[i] > self.upper_bound[i]:
                variables[i] = self.upper_bound[i]
        solution.variables = variables
        return solution


class PermutationSALBP1(SALBProblemI):
    def __init__(self, file_path, file_name):
        super(PermutationSALBP1, self).__init__(file_path=file_path, file_name=file_name)
        self.solution_type = "Permutation"

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


class AntSALBP1(SALBProblemI):
    def __init__(self, file_path, file_name):
        super(AntSALBP1, self).__init__(file_path=file_path, file_name=file_name)
        self.first_task_prob = []
        self.task_task_prob = []
        self.task_station_prob = []

    def probability_decoding1(self):
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
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
                        eligible_task_set.append(i)
                if len(eligible_task_set) == 0:
                    break
                else:
                    if len(assigned_task_set) == 0:
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            self.task_select_probability.append(pow(self.first_task_prob[i], self.alpha) *
                                                                pow(self.task_probability[i], self.beta))
                    else:
                        for i in eligible_task_set:
                            self.task_select_set.append(i)
                            self.task_select_probability.append(
                                pow(self.task_task_prob[assigned_task_set[-1]][i], self.alpha) * pow(
                                    self.task_probability[i], self.beta))
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
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        self.task_permutation = copy.deepcopy(assigned_task_set)
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def probability_decoding2(self):
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
                    if self.degrees[i] == 0 and self.t[i] + self.station_time[self.n_stations] <= self.ct:
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
                sub_hash_value = sub_hash_value + self.hash_values[selected_task]
                if sub_hash_value > self.Hash_Size:
                    sub_hash_value = sub_hash_value % self.Hash_Size
                self.task_station[selected_task] = self.n_stations
                assigned_tasks.append(selected_task)
                self.feasible_task_permutation.append(selected_task)
                self.degrees[selected_task] -= 1
                for j in self.successors[selected_task]:
                    self.degrees[j] = self.degrees[j] - 1
                eligible_task_set.clear()
            self.tasks_to_stations.append(copy.deepcopy(assigned_tasks))
            assigned_tasks.clear()
        self.task_permutation = copy.deepcopy(assigned_task_set)
        unassigned_task_set.clear()
        assigned_task_set.clear()
        eligible_task_set.clear()

    def obtain_scores(self):
        self.objectives.append(self.n_stations + 1)
        self.objectives.append(self.n_stations + 1 + (self.station_time[self.n_stations - 1] +
                                                      self.station_time[self.n_stations]) / (2.0 * self.ct))

    def create_solution1(self, first_task_prob_copy, task_task_prob_copy):
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

    def create_solution2(self, task_station_prob_copy):
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
