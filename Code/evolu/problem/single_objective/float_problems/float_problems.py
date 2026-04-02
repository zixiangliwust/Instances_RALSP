import math
from evolu.core.problem import FloatProblem
from evolu.core.solution import FloatSolution

"""
module:: unconstrained
synopsis: Unconstrained test problems for single-objective optimization
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Sphere(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Sphere, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        solution = self.remedy_solution(solution)
        total = 0.0
        for x in solution.variables:
            total += x * x
        solution.objectives[0] = total
        solution.fitness = solution.objectives[0]
        return solution

    def get_name(self) -> str:
        return "Sphere"


class Rastrigin(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        solution = self.remedy_solution(solution)
        a = 10.0
        result = a * solution.number_of_variables
        x = solution.variables
        for i in range(solution.number_of_variables):
            result += x[i] * x[i] - a * math.cos(2 * math.pi * x[i])
        solution.objectives[0] = result
        solution.fitness = solution.objectives[0]
        return solution

    def get_name(self) -> str:
        return "Rastrigin"
