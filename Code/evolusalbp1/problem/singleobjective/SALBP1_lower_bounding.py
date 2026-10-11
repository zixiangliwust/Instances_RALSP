import math
from copy import deepcopy
from typing import List, Any
import numpy as np

from evolu.core.problem import Problem


class SALBP1LowerBounding:
    __EPS: float = 1.0e-14

    def __init__(self, problem: Problem) -> None:
        """Initialize SALBP1 lower bounding calculator."""
        self.n_tasks: int = problem.n_tasks
        self.ct: int = problem.ct
        self.t: List[int] = problem.t
        self.lb2_values: np.ndarray = np.array([])
        self.lb3_values: np.ndarray = np.array([])
        self.compute_lb2_values()
        self.compute_lb3_values()

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

    def calculate_lb1(self, list_of_items: List[int]) -> int:
        """Calculate LB1 bound."""
        if len(list_of_items) == 0:
            return 0
        t_sum = 0
        for i in list_of_items:
            t_sum = t_sum + self.t[i]
        return int(math.ceil(t_sum / float(self.ct) - self.__EPS))

    def calculate_lb2(self, list_of_items: List[int]) -> int:
        """Calculate LB2 bound."""
        if len(list_of_items) == 0:
            return 0
        lb2_sum = 0
        for i in list_of_items:
            lb2_sum = lb2_sum + self.lb2_values[i]
        return int(math.ceil(lb2_sum - self.__EPS))

    def calculate_lb3(self, list_of_items: List[int]) -> int:
        """Calculate LB3 bound."""
        if len(list_of_items) == 0:
            return 0
        lb3_sum = 0
        for i in list_of_items:
            lb3_sum = lb3_sum + self.lb3_values[i]
        return int(math.ceil(lb3_sum - self.__EPS))

    def calculate_lb1_lb2_lb3(self, list_of_items: List[int]) -> int:
        """Calculate maximum of LB1, LB2, and LB3 bounds."""
        if len(list_of_items) == 0:
            return 0
        lb1 = self.calculate_lb1(list_of_items)
        lb2 = self.calculate_lb2(list_of_items)
        lb3 = self.calculate_lb3(list_of_items)
        return max(lb1, lb2, lb3)

    # Bound LB6 [see bound LB6 in Scholl and Klein 1997]
    # Scholl, Armin, and Robert Klein. 1997. "SALOME: A bidirectional branch-and-bound procedure for assembly
    # line balancing."  INFORMS Journal on Computing 9 (4):319-34. doi: 10.1287/ijoc.9.4.319.
    def calculate_lb6(self, list_of_items: List[int]) -> int:
        """Calculate LB6 bound (see bound LB6 in Scholl and Klein 1997)."""
        if len(list_of_items) == 0:
            return 0
        items = deepcopy(list_of_items)
        for i in range(0, len(items) - 1):
            for j in range(i + 1, len(items)):
                if self.t[items[i]] < self.t[items[j]]:
                    items[i], items[j] = items[j], items[i]
        d1, d2, d3 = 0, 0, 0
        complete_states = [self.ct for _ in range(0, len(list_of_items))]
        for i in range(0, len(items)):
            if self.t[items[i]] > self.ct / 2.0:
                complete_states[d1] = complete_states[d1] - self.t[items[i]]
                d1 = d1 + 1
            elif self.ct / 2.0 >= self.t[items[i]] > self.ct / 3.0:
                d2 = d2 + 1
            elif self.t[items[i]] <= self.ct / 3.0:
                break
        for i in range(d1, len(items)):
            if self.ct / 2.0 >= self.t[items[i]] > self.ct / 3.0:
                for j in range(0, d1):
                    if self.t[items[i]] <= complete_states[j]:
                        complete_states[j] = complete_states[j] - self.t[items[i]]
                        d2 = d2 - 1
                        break
            else:
                break
        d2 = d2 // 2 + d2 % 2
        time_interval = 0
        for j in range(0, len(items)):
            if j >= 1:
                if j < len(items) and self.t[items[j - 1]] == self.t[items[j]]:
                    continue
                elif time_interval < self.ct - self.t[items[j - 1]]:
                    time_interval = self.ct - self.t[items[j - 1]]
            if time_interval > self.ct / 3.0:
                break
            t_sum_temp = 0
            if j == 0:
                for i in range(0, len(items)):
                    t_sum_temp = t_sum_temp + self.t[items[i]]
            else:
                for i in range(j, len(items)):
                    if time_interval < self.t[items[i]] < self.ct - time_interval:
                        t_sum_temp = t_sum_temp + self.t[items[i]]
            d3_temp = int(math.ceil(t_sum_temp / (self.ct * 1.0))) + j - d1 - d2
            d3 = max(d3, d3_temp)
        return d1 + d2 + d3

    # Simplified LB6 to strengthen lb2 and lb3 with fast speed than the original LB6
    # Scholl, Armin, and Robert Klein. 1997. "SALOME: A bidirectional branch-and-bound procedure for assembly
    # line balancing."  INFORMS Journal on Computing 9 (4):319-34. doi: 10.1287/ijoc.9.4.319.
    def calculate_lb6_simplified(self, list_of_items: List[int]) -> int:
        """Calculate simplified LB6 bound to strengthen lb2 and lb3 with fast speed."""
        if len(list_of_items) == 0:
            return 0
        items = deepcopy(list_of_items)
        for i in range(0, len(items) - 1):
            for j in range(i + 1, len(items)):
                if self.t[items[i]] < self.t[items[j]]:
                    items[i], items[j] = items[j], items[i]
        d1, d2, d3 = 0, 0, 0
        complete_states = [self.ct for _ in range(0, len(list_of_items))]
        for i in range(0, len(items)):
            if self.t[items[i]] > self.ct / 2.0:
                complete_states[d1] = complete_states[d1] - self.t[items[i]]
                d1 = d1 + 1
            elif self.ct / 2.0 >= self.t[items[i]] > self.ct / 3.0:
                d2 = d2 + 1
            elif self.t[items[i]] <= self.ct / 3.0:
                break
        for i in range(d1, len(items)):
            if self.ct / 2.0 >= self.t[items[i]] > self.ct / 3.0:
                for j in range(0, d1):
                    if self.t[items[i]] <= complete_states[j]:
                        complete_states[j] = complete_states[j] - self.t[items[i]]
                        d2 = d2 - 1
                        break
            else:
                break
        d2 = d2 // 2 + d2 % 2
        return d1 + d2 + d3

    # Bound LB7 [see bound LB7 in Scholl and Klein 1997]
    # Scholl, Armin, and Robert Klein. 1997. "SALOME: A bidirectional branch-and-bound procedure for assembly
    # line balancing."  INFORMS Journal on Computing 9 (4):319-34. doi: 10.1287/ijoc.9.4.319.
    def calculate_lb7(self, list_of_items: List[int]) -> int:
        """Calculate LB7 bound (see bound LB7 in Scholl and Klein 1997)."""
        lb = self.calculate_lb1(list_of_items)
        items = deepcopy(list_of_items)
        for i in range(0, len(items) - 1):
            for j in range(i + 1, len(items)):
                if self.t[items[i]] < self.t[items[j]]:
                    items[i], items[j] = items[j], items[i]
        while True:
            sub_ct = 0
            for k in range(1, math.floor((len(items) - 1) / (lb * 1.0)) + 1):
                sum_t = 0
                for i in range(0, k + 1):
                    sum_t = sum_t + self.t[items[lb * k - i]]
                sub_ct = max(sub_ct, sum_t)
            if sub_ct > self.ct:
                lb = lb + 1
            else:
                break
        return lb

    # Dual feasible functions with k as 3 in bin packing lower bounds
    # Pereira, Jordi. 2015. "Empirical evaluation of lower bounding methods for the simple assembly line
    # balancing problem."  International Journal of Production Research 53 (11):3327-40.
    # doi: 10.1080/00207543.2014.980014.
    def dual_feasible_function_3(self, list_of_items: List[int]) -> int:
        """Dual feasible functions with k as 3 in bin packing lower bounds."""
        if len(list_of_items) == 0:
            return 0
        items = deepcopy(list_of_items)
        for i in range(0, len(items) - 1):
            for j in range(i + 1, len(items)):
                if self.t[items[i]] < self.t[items[j]]:
                    items[i], items[j] = items[j], items[i]
        lb, time_interval = 0, 0
        for k in range(0, 3):
            time_interval = 0
            for j in range(0, len(items)):
                if j >= 1:
                    if time_interval < self.ct - self.t[items[j - 1]]:
                        time_interval = self.ct - self.t[items[j - 1]]
                    else:
                        continue
                if time_interval >= self.ct / 2.0:
                    break
                t_sum_temp = 0.0
                for i in range(0, len(items)):
                    task_weight = self.t[items[i]] / (self.ct * 1.0)
                    if task_weight >= (self.ct - time_interval) / (self.ct * 1.0):
                        task_weight = 1
                    elif task_weight <= time_interval / (self.ct * 1.0):
                        task_weight = 0
                    if (k >= 1) and (task_weight > time_interval / (self.ct * 1.0)) and (
                            task_weight < (self.ct - time_interval) / (self.ct * 1.0)):
                        if task_weight * (k + 1) - int(task_weight * (k + 1)) > 0.0:
                            task_weight = math.floor(task_weight * (k + 1)) / (k * 1.0)
                    t_sum_temp = t_sum_temp + task_weight
                sub_lb = int(math.ceil(t_sum_temp - self.__EPS))
                lb = max(lb, sub_lb)
        return lb

    # Dual feasible functions with k as 100 in bin packing lower bounds
    # Pereira, Jordi. 2015. "Empirical evaluation of lower bounding methods for the simple assembly line
    # balancing problem."  International Journal of Production Research 53 (11):3327-40.
    # doi: 10.1080/00207543.2014.980014.
    def dual_feasible_function_100(self, list_of_items: List[int]) -> int:
        """Dual feasible functions with k as 100 in bin packing lower bounds."""
        if len(list_of_items) == 0:
            return 0
        items = deepcopy(list_of_items)
        for i in range(0, len(items) - 1):
            for j in range(i + 1, len(items)):
                if self.t[items[i]] < self.t[items[j]]:
                    items[i], items[j] = items[j], items[i]
        lb, time_interval = 0, 0
        for k in range(0, 100):
            time_interval = 0
            for j in range(0, len(items)):
                if j >= 1:
                    if time_interval < self.ct - self.t[items[j - 1]]:
                        time_interval = self.ct - self.t[items[j - 1]]
                    else:
                        continue
                if time_interval >= self.ct / 2.0:
                    break
                t_sum_temp = 0.0
                for i in range(0, len(items)):
                    task_weight = self.t[items[i]] / (self.ct * 1.0)
                    if task_weight >= (self.ct - time_interval) / (self.ct * 1.0):
                        task_weight = 1
                    elif task_weight <= time_interval / (self.ct * 1.0):
                        task_weight = 0
                    if (k >= 1) and (task_weight > time_interval / (self.ct * 1.0)) and (
                            task_weight < (self.ct - time_interval) / (self.ct * 1.0)):
                        if task_weight * (k + 1) - int(task_weight * (k + 1)) > 0.0:
                            task_weight = math.floor(task_weight * (k + 1)) / (k * 1.0)
                    t_sum_temp = t_sum_temp + task_weight
                sub_lb = int(math.ceil(t_sum_temp - self.__EPS))
                lb = max(lb, sub_lb)
        return lb
