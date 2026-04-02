import time
import math
from typing import Any, Optional, Tuple, List
import numpy as np
from docplex.mp.model import Model
from docplex.cp.model import CpoModel
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.problem import Problem
from evolu.core.solution import Solution

"""
Module: Mixed-integer programming model
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP1MPModel(SingleObjectiveSwarmRoot):
    """
    Mixed-integer programming model
    References:
    [1] Bukchin, Yossi, and Tal Raviv. 2018. "Constraint programming for solving various assembly line balancing problems."
    Omega 78:57-68. doi: 10.1016/j.omega.2017.06.008.
    """

    def __init__(self, problem: Problem[Solution], max_seconds: int = 10) -> None:
        """Initialize SALBP1 mixed-integer programming model."""
        super(SALBP1MPModel, self).__init__(problem=problem)
        self.problem: Problem[Solution] = problem
        self.max_seconds: int = max_seconds
        self.n_tasks: int = problem.n_tasks
        self.t: List[int] = problem.t
        self.ct: int = problem.ct
        self.imm_precedence: List[tuple] = problem.imm_precedence
        self.optimality_verified: Optional[int] = None
        self.g_best: Solution = self.problem.create_solution()
        self.algorithm_name: str = "SALBP1MPModel"

    def solve_model(self) -> Tuple[Any, Any]:
        """Solve the mixed-integer programming model."""
        max_n_stations = self.n_tasks
        # Create my model
        model = Model()
        n_stations = model.integer_var(1, max_n_stations, name='n_stations')
        x = model.binary_var_matrix(range(0, self.n_tasks), range(0, max_n_stations), name='x')
        y = model.binary_var_list(max_n_stations, name='y')
        # task assignment constraint
        for i in range(0, self.n_tasks):
            model.add(model.sum(x[i, j] for j in range(0, max_n_stations)) == 1)
        # precedence constraint
        for h, i in self.imm_precedence:
            model.add(model.sum(j * x[h, j] for j in range(0, max_n_stations)) <=
                      model.sum(k * x[i, k] for k in range(0, max_n_stations)))
        # cycle time constraint
        for j in range(0, max_n_stations):
            model.add(model.sum(self.t[i] * x[i, j] for i in range(0, self.n_tasks)) <= self.ct)
        # calculate the number of stations
        for i in range(0, self.n_tasks):
            for j in range(0, max_n_stations):
                model.add(x[i, j] <= y[j])
        for j in range(0, max_n_stations):
            if j < len(range(0, max_n_stations)) - 1:
                model.add(y[j] >= y[j + 1])
        model.add(model.sum(y[j] for j in range(0, max_n_stations)) <= n_stations)
        # solve model
        model.minimize(n_stations)
        model.set_time_limit(self.max_seconds)
        model_result = model.solve()
        return model.solve_details, model_result

    def run(self):
        self.start_computing_time = time.time()
        model_solve_details, solution = self.solve_model()
        self.total_computing_time = time.time() - self.start_computing_time
        if model_solve_details.status != 'time limit exceeded, no integer solution':
            self.g_best.objectives[0] = solution.objective_value
        else:
            self.g_best.objectives[0] = math.pow(10, 8)
        if model_solve_details.status == 'integer optimal solution':
            self.optimality_verified = 1
        else:
            self.optimality_verified = 0
        self.g_best.objectives[1] = self.optimality_verified
        print(model_solve_details.status)
        print(model_solve_details.time)
        # print(model_solve_details.gap)
        print(model_solve_details)


"""
Module: Basic constraint programming model
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP1CPModel(SingleObjectiveSwarmRoot):
    """
    Basic constraint programming model
    References:
    [1] Bukchin, Yossi, and Tal Raviv. 2018. "Constraint programming for solving various assembly line balancing problems."
    Omega 78:57-68. doi: 10.1016/j.omega.2017.06.008.
    """

    def __init__(self, problem: Problem[Solution], max_seconds: int = 10) -> None:
        """Initialize SALBP1 constraint programming model."""
        super(SALBP1CPModel, self).__init__(problem=problem)
        self.problem: Problem[Solution] = problem
        self.max_seconds: int = max_seconds
        self.n_tasks: int = problem.n_tasks
        self.t: List[int] = problem.t
        self.ct: int = problem.ct
        self.imm_precedence: List[tuple] = problem.imm_precedence
        self.optimality_verified: Optional[int] = None
        self.g_best: Solution = self.problem.create_solution()
        self.algorithm_name: str = "SALBP1CPModel"

    def solve_model(self) -> Any:
        """Solve the constraint programming model."""
        max_n_stations = self.n_tasks
        # Create CP model
        model = CpoModel()
        n_stations = model.integer_var(1, self.n_tasks, name='n_stations')
        x = model.integer_var_list(self.n_tasks, 0, max_n_stations - 1, name='x')
        # precedence constraint
        for h, i in self.imm_precedence:
            model.add(x[h] <= x[i])
        # cycle time constraint
        for j in range(0, max_n_stations):
            model.add(sum(self.t[i] * (x[i] == j) for i in range(0, self.n_tasks)) <= self.ct)
        # calculate the number of stations
        for i in range(0, self.n_tasks):
            model.add(x[i] + 1 <= n_stations)
        # solve model
        model.add(model.minimize(n_stations))
        model_result = model.solve(FailLimit=100000, TimeLimit=self.max_seconds)
        # model_result = model.solve(FailLimit=100000, TimeLimit=10)
        return model_result

    def run(self):
        self.start_computing_time = time.time()
        model_result = self.solve_model()
        self.total_computing_time = time.time() - self.start_computing_time
        if model_result.is_solution():
            self.g_best.objectives[0] = model_result.get_objective_values()[0]
        else:
            self.g_best.objectives[0] = math.pow(10, 8)
        if model_result.get_solve_status() == 'Optimal':
            self.optimality_verified = 1
        else:
            self.optimality_verified = 0
        self.g_best.objectives[1] = self.optimality_verified
        print(model_result.get_solve_status())
        print(model_result.get_solve_time())
        print(model_result.get_objective_values()[0])


"""
Module: Improved constraint programming model in Bukchin and Raviv 2018
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SALBP1CPModelNew(SingleObjectiveSwarmRoot):
    """
    Improved constraint programming model in Bukchin and Raviv 2018
    References:
    [1] Bukchin, Yossi, and Tal Raviv. 2018. "Constraint programming for solving various assembly line balancing problems."
    Omega 78:57-68. doi: 10.1016/j.omega.2017.06.008.
    """

    def __init__(self, problem: Problem[Solution], max_seconds: int = 10) -> None:
        """Initialize improved SALBP1 constraint programming model."""
        super(SALBP1CPModelNew, self).__init__(problem=problem)
        self.problem: Problem[Solution] = problem
        self.max_seconds: int = max_seconds
        self.n_tasks: int = problem.n_tasks
        self.t: List[int] = problem.t
        self.ct: int = problem.ct
        self.imm_precedence: List[tuple] = problem.imm_precedence
        self.all_precedence_matrix: Any = problem.all_precedence_matrix
        self.positional_weight: List[float] = problem.positional_weight
        self.rev_positional_weight: List[float] = problem.rev_positional_weight
        self.algorithm_name: str = "SALBP1CPModelNew"
        self.optimality_verified: Optional[int] = None
        self.earliest_task_station: List[int] = [0 for _ in range(0, self.n_tasks)]
        self.latest_task_station: List[int] = [0 for _ in range(0, self.n_tasks)]
        self.task_distance: np.ndarray = np.zeros((self.n_tasks, self.n_tasks))
        for i in range(0, self.n_tasks):
            self.earliest_task_station[i] = int(math.ceil(self.rev_positional_weight[i] / (self.ct * 1.0))) - 1
            self.latest_task_station[i] = int(math.ceil(self.positional_weight[i] / (self.ct * 1.0))) - 1
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.all_precedence_matrix[i][j]:
                    self.task_distance[i][j] = self.t[i] + self.t[j]
                    for h in range(0, self.n_tasks):
                        if self.all_precedence_matrix[i][h] and self.all_precedence_matrix[h][j]:
                            self.task_distance[i][j] = self.task_distance[i][j] + self.t[h]
                    self.task_distance[i][j] = int(math.ceil(self.task_distance[i][j] / (self.ct * 1.0))) - 1
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.task_distance[i][j]:
                    for h in range(0, self.n_tasks):
                        if self.all_precedence_matrix[i][h] and self.all_precedence_matrix[h][j] and \
                                self.task_distance[i][j] <= self.task_distance[i][h] + self.task_distance[h][j]:
                            self.task_distance[i][j] = 0
                            break
        self.g_best = self.problem.create_evaluate_random_solution()
        for i in range(0, 99):
            new_solution = self.problem.create_evaluate_random_solution()
            if new_solution.objectives[0] < self.g_best.objectives[0]:
                self.g_best = new_solution
        print("The input upper bound is: %f" % self.g_best.objectives[0])

    def solve_model(self) -> Any:
        """Solve the improved constraint programming model."""
        max_n_stations: int = int(self.g_best.objectives[0])
        # Create CP model
        model = CpoModel()
        n_stations = model.integer_var(1, max_n_stations, name='n_stations')
        x = model.integer_var_list(self.n_tasks, 0, max_n_stations - 1, name='x')
        # precedence constraint
        for h, i in self.imm_precedence:
            model.add(x[h] <= x[i])
        # cycle time constraint
        for j in range(0, max_n_stations):
            model.add(sum(self.t[i] * (x[i] == j) for i in range(0, self.n_tasks)) <= self.ct)
        # calculate the number of stations
        for i in range(0, self.n_tasks):
            model.add(x[i] + 1 <= n_stations)
        for i in range(0, self.n_tasks):
            model.add(x[i] >= self.earliest_task_station[i])
            model.add(x[i] <= n_stations - 1 - self.latest_task_station[i])
        for i in range(0, self.n_tasks):
            for j in range(0, self.n_tasks):
                if self.task_distance[i][j] > 0:
                    model.add(x[i] + self.task_distance[i][j] <= x[j])
        # solve model
        model.add(model.minimize(n_stations))
        model_result = model.solve(FailLimit=100000, TimeLimit=self.max_seconds)
        return model_result

    def run(self):
        self.start_computing_time = time.time()
        model_result = self.solve_model()
        self.total_computing_time = time.time() - self.start_computing_time
        if model_result.is_solution():
            self.g_best.objectives[0] = model_result.get_objective_values()[0]
        if model_result.get_solve_status() == 'Optimal':
            self.optimality_verified = 1
        else:
            self.optimality_verified = 0
        self.g_best.objectives[1] = self.optimality_verified
        print(model_result.get_solve_status())
        print(model_result.get_solve_time())
        print(model_result.get_objective_values()[0])
