import time
import math
import numpy as np
from docplex.mp.model import Model
from docplex.cp.model import CpoModel
from evolu.core.algorithm import SingleObjectiveSwarmRoot

"""
Module: Mixed-integer programming model
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class UALBP1MPModel(SingleObjectiveSwarmRoot):
    """
    Mixed-integer programming model
    References:
    [1] Bukchin, Yossi, and Tal Raviv. 2018. "Constraint programming for solving various assembly line balancing problems."
    Omega 78:57-68. doi: 10.1016/j.omega.2017.06.008.
    """

    def __init__(self, problem: Problem[Solution], max_seconds: int = 10) -> None:
        """Initialize UALBP1 mixed-integer programming model."""
        super(UALBP1MPModel, self).__init__(problem=problem)
        self.problem: Problem[Solution] = problem
        self.max_seconds: int = max_seconds
        self.n_tasks: int = problem.n_tasks
        self.t: List[int] = problem.t
        self.ct: int = problem.ct
        self.imm_precedence: List[tuple] = problem.imm_precedence
        self.optimality_verified: Optional[int] = None
        self.g_best: Solution = self.problem.create_solution()
        self.algorithm_name: str = "UALBP1MPModel"

    def solve_model(self) -> Tuple[Any, Any]:
        """Solve the mixed-integer programming model."""
        max_n_stations = self.n_tasks
        # Create my model
        model = Model()
        n_stations = model.integer_var(1, max_n_stations, name='n_stations')
        x = model.binary_var_matrix(range(0, self.n_tasks), range(0, 2 * max_n_stations), name='x')
        y = model.binary_var_list(max_n_stations, name='y')
        # task assignment constraint
        for i in range(0, self.n_tasks):
            model.add(model.sum(x[i, j] for j in range(0, 2 * max_n_stations)) == 1)
        # precedence constraint
        for h, i in self.imm_precedence:
            model.add(model.sum(j * x[h, j] for j in range(0, 2 * max_n_stations)) <=
                      model.sum(k * x[i, k] for k in range(0, 2 * max_n_stations)))
        # cycle time constraint
        for j in range(0, max_n_stations):
            model.add(model.sum(self.t[i] * x[i, j] + self.t[i] * x[i, 2 * max_n_stations - 1 - j] for i in
                                range(0, self.n_tasks)) <= self.ct)
        # calculate the number of stations
        for i in range(0, self.n_tasks):
            for j in range(0, max_n_stations):
                model.add(x[i, j] + x[i, 2 * max_n_stations - 1 - j] <= y[j])
        for j in range(0, max_n_stations):
            if j < max_n_stations - 1:
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
        print(solution.objective_value)
        print(model_solve_details.status)
        print(model_solve_details.time)
        # print(model_solve_details.gap)
        print(model_solve_details)


"""
Module: Basic constraint programming model
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class UALBP1CPModel:
    """
    Basic constraint programming model
    References:
    [1] Bukchin, Yossi, and Tal Raviv. 2018. "Constraint programming for solving various assembly line balancing problems."
    Omega 78:57-68. doi: 10.1016/j.omega.2017.06.008.
    """

    def __init__(self, problem: Problem[Solution]) -> None:
        """Initialize UALBP1 constraint programming model."""
        self.n_tasks: int = problem.n_tasks
        self.t: List[int] = problem.t
        self.ct: int = problem.ct
        self.imm_precedence: List[tuple] = problem.imm_precedence
        self.algorithm_name: str = "UALBP1CPModel"
        self.optimality_verified: Optional[int] = None

    def solve_model(self) -> Any:
        """Solve the constraint programming model."""
        max_n_stations = self.n_tasks
        # Create CP model
        model = CpoModel()
        n_stations = model.integer_var(1, max_n_stations, name='n_stations')
        x = model.integer_var_list(self.n_tasks, 0, 2 * max_n_stations - 1, name='x')
        # precedence constraint
        for h, i in self.imm_precedence:
            model.add(x[h] <= x[i])
        # cycle time constraint
        for j in range(0, max_n_stations):
            model.add(sum(self.t[i] * (x[i] == j) + self.t[i] * (x[i] == 2 * n_stations - 1 - j) for i in
                          range(0, self.n_tasks)) <= self.ct)
        # calculate the number of stations
        for i in range(0, self.n_tasks):
            model.add(x[i] + 1 <= 2 * n_stations)
        # solve model
        model.add(model.minimize(n_stations))
        model_result = model.solve(FailLimit=100000, TimeLimit=100)
        # model_result = model.solve(FailLimit=100000, TimeLimit=10)
        return model_result

    def run(self):
        start_computing_time = time.time()
        model_result = self.solve_model()
        end_computing_time = time.time()
        print(model_result.get_solve_status())
        print(model_result.get_solve_time())
        print(model_result.get_objective_values()[0])
        if model_result.get_solve_status() == 'Optimal':
            self.optimality_verified = 1
        else:
            self.optimality_verified = 0
        if model_result.is_solution():
            print(model_result.get_objective_values()[0])
        else:
            print(math.pow(10, 8))
