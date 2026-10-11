# -*- coding: utf-8 -*-
import copy
import random
import numpy
from typing import List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")
"""
Module: Basic migrating birds optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MBOBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Basic migrating birds optimization
    References:
    [1] Duman, Ekrem, Mitat Uysal, and Ali Fuat Alkaya. 2012. "Migrating Birds Optimization: A new metaheuristic approach and its performance on quadratic assignment problem."
    Information Sciences 217:65-77. doi: http://dx.doi.org/10.1016/j.ins.2012.06.032.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MBOBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Migrating birds optimization"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.para_k = 3
        self.para_x = 1
        self.para_m = 20
        self.leader_neighbors = []
        self.previous_neighbors = []
        self.individual_neighbors = []

    def leader_improvement(self) -> None:
        for k in range(0, self.para_k):
            new_solution = self.mutation_operator.execute(self.solutions[0])
            new_solution = self.evaluate_solution(new_solution)
            self.leader_neighbors.append(new_solution)
        self.leader_neighbors = self.sort_population.execute(self.leader_neighbors)
        if self.comparator.compare(self.leader_neighbors[0], self.solutions[0]) == -1:
            self.solutions[0] = self.leader_neighbors[0]

    def population_improvement(self) -> None:
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 1])
        for j in range(1, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(self.solutions[j])
                new_solution = self.evaluate_solution(new_solution)
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 2])
        for j in range(2, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(self.solutions[j])
                new_solution = self.evaluate_solution(new_solution)
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        self.leader_neighbors.clear()
        self.individual_neighbors.clear()

    def move_leader(self) -> None:
        old_solutions = copy.deepcopy(self.solutions)
        if random.random() < 0.5:
            self.solutions[0] = old_solutions[1]
            for j in range(3, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 2] = old_solutions[0]
        else:
            self.solutions[0] = old_solutions[2]
            for j in range(4, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 1] = old_solutions[0]

    def evolve(self) -> None:
        self.offspring_population_size = 0
        for _ in range(0, self.para_m):
            self.offspring_population_size += self.para_k + (self.para_k - self.para_x) * (self.population_size - 1)
            self.leader_improvement()
            self.population_improvement()
        self.move_leader()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Improved migrating birds optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class IMBO(SingleObjectiveSwarmRoot[S, R]):
    """
    Improved migrating birds optimization
    References:
    [1] Janardhanan, Mukund Nilakantan, Zixiang Li, Grzegorz Bocewicz, Zbigniew Banaszak, and Peter Nielsen. 2019.
    "Metaheuristic algorithms for balancing robotic assembly lines with sequence-dependent robot setup times."
    Applied Mathematical Modelling 65:256-70. doi: 10.1016/j.apm.2018.08.016.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IMBO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Improved migrating birds optimization"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.para_k = 11
        self.para_x = 5
        self.para_m = 20
        self.scout_time = 100
        self.neighbor_size = 10
        self.leader_neighbors = []
        self.previous_neighbors = []
        self.individual_neighbors = []

    def leader_improvement(self) -> None:
        neighbor = self.solutions[0]
        for k in range(0, self.para_k):
            new_solution = self.mutation_operator.execute(neighbor)
            new_solution = self.evaluate_solution(new_solution)
            if self.comparator.compare(new_solution, neighbor) <= 0:
                neighbor = new_solution
            if self.identical_solutions_comparator.compare(new_solution, self.solutions[0]) == 0:
                for i in range(0, self.problem.number_of_objectives):
                    new_solution.objectives[i] = float("inf")
            else:
                new_solution.survive_time = 0
            self.leader_neighbors.append(new_solution)
        self.leader_neighbors = self.sort_population.execute(self.leader_neighbors)
        if self.comparator.compare(self.leader_neighbors[0], self.solutions[0]) == -1:
            self.solutions[0] = self.leader_neighbors[0]

    def population_improvement(self) -> None:
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 1])
        for j in range(1, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                else:
                    new_solution.survive_time = 0
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 2])
        for j in range(2, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                else:
                    new_solution.survive_time = 0
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        self.leader_neighbors.clear()
        self.individual_neighbors.clear()

    def move_leader(self) -> None:
        old_solutions = copy.deepcopy(self.solutions)
        if random.random() < 0.5:
            self.solutions[0] = old_solutions[1]
            for j in range(3, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 2] = old_solutions[0]
        else:
            self.solutions[0] = old_solutions[2]
            for j in range(4, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 1] = old_solutions[0]

    def scout_phase(self) -> None:
        for j in range(0, self.population_size):
            if self.solutions[j].survive_time >= self.scout_time:
                neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in range(0, self.neighbor_size)]
                neighbors = self.evaluate(neighbors)
                neighbors = self.sort_population.execute(neighbors)
                for solution in neighbors[0:]:
                    if self.identical_solutions_comparator.compare(solution, self.solutions[j]) != 0:
                        self.solutions[j] = solution
                        break
                self.offspring_population_size += self.neighbor_size

    def evolve(self) -> None:
        self.offspring_population_size = 0
        for _ in range(0, self.para_m):
            self.offspring_population_size += self.para_k + (self.para_k - self.para_x) * (self.population_size - 1)
            for j in range(0, self.population_size):
                self.solutions[j].survive_time += 1
            self.leader_improvement()
            self.population_improvement()
            self.scout_phase()
        self.move_leader()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Migrating birds optimization with simulated annealing-based acceptance
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MBOSA(SingleObjectiveSwarmRoot[S, R]):
    """
    Migrating birds optimization with simulated annealing-based acceptance
    References:
    [1] Janardhanan, Mukund Nilakantan, Zixiang Li, and Peter Nielsen. 2019. "Model and migrating birds optimization algorithm for two-sided assembly line worker assignment and balancing problem."
    Soft Computing 23 (21):11263-76. doi: 10.1007/s00500-018-03684-8.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MBOSA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Migrating birds optimization with simulated annealing-based acceptance"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.para_k = 11
        self.para_x = 5
        self.para_m = 20
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.leader_neighbors = []
        self.previous_neighbors = []
        self.individual_neighbors = []

    def leader_improvement(self) -> None:
        neighbor = self.solutions[0]
        for k in range(0, self.para_k):
            new_solution = self.mutation_operator.execute(neighbor)
            new_solution = self.evaluate_solution(new_solution)
            if self.comparator.compare(new_solution, neighbor) <= 0:
                neighbor = new_solution
            if self.identical_solutions_comparator.compare(new_solution, self.solutions[0]) == 0:
                for i in range(0, self.problem.number_of_objectives):
                    new_solution.objectives[i] = float("inf")
            self.leader_neighbors.append(new_solution)
        self.leader_neighbors = self.sort_population.execute(self.leader_neighbors)
        acceptance_probability = self.compute_acceptance_probability(
            self.solutions[0].objectives[0], self.leader_neighbors[0].objectives[0], self.temperature
        )
        if self.comparator.compare(self.leader_neighbors[0], self.solutions[0]) == -1:
            self.solutions[0] = self.leader_neighbors[0]
        elif acceptance_probability > random.random():
            self.solutions[0] = self.leader_neighbors[0]

    def population_improvement(self) -> None:
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 1])
        for j in range(1, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            acceptance_probability = self.compute_acceptance_probability(
                self.solutions[j].objectives[0], self.individual_neighbors[0].objectives[0], self.temperature
            )
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            elif acceptance_probability > random.random():
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 2])
        for j in range(2, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            acceptance_probability = self.compute_acceptance_probability(
                self.solutions[j].objectives[0], self.individual_neighbors[0].objectives[0], self.temperature
            )
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            elif acceptance_probability > random.random():
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        self.leader_neighbors.clear()
        self.individual_neighbors.clear()

    def move_leader(self) -> None:
        old_solutions = copy.deepcopy(self.solutions)
        if random.random() < 0.5:
            self.solutions[0] = old_solutions[1]
            for j in range(3, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 2] = old_solutions[0]
        else:
            self.solutions[0] = old_solutions[2]
            for j in range(4, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 1] = old_solutions[0]

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def evolve(self) -> None:
        self.offspring_population_size = 0
        for _ in range(0, self.para_m):
            self.offspring_population_size += self.para_k + (self.para_k - self.para_x) * (self.population_size - 1)
            self.leader_improvement()
            self.population_improvement()
            self.temperature *= self.cooling_rate
        self.move_leader()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class IMBOSA(SingleObjectiveSwarmRoot[S, R]):
    """
    Improved migrating birds optimization with simulated annealing-based acceptance
    References:
    [1] Janardhanan, Mukund Nilakantan, Zixiang Li, and Peter Nielsen. 2019. "Model and migrating birds optimization algorithm for two-sided assembly line worker assignment and balancing problem."
    Soft Computing 23 (21):11263-76. doi: 10.1007/s00500-018-03684-8.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IMBOSA, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Improved migrating birds optimization with simulated annealing-based acceptance"
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.para_k = 11
        self.para_x = 5
        self.para_m = 20
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.scout_time = 100
        self.neighbor_size = 10
        self.leader_neighbors = []
        self.previous_neighbors = []
        self.individual_neighbors = []

    def leader_improvement(self) -> None:
        neighbor = self.solutions[0]
        for k in range(0, self.para_k):
            new_solution = self.mutation_operator.execute(neighbor)
            new_solution = self.evaluate_solution(new_solution)
            if self.comparator.compare(new_solution, neighbor) <= 0:
                neighbor = new_solution
            if self.identical_solutions_comparator.compare(new_solution, self.solutions[0]) == 0:
                for i in range(0, self.problem.number_of_objectives):
                    new_solution.objectives[i] = float("inf")
            else:
                new_solution.survive_time = 0
            self.leader_neighbors.append(new_solution)
        self.leader_neighbors = self.sort_population.execute(self.leader_neighbors)
        acceptance_probability = self.compute_acceptance_probability(
            self.solutions[0].objectives[0], self.leader_neighbors[0].objectives[0], self.temperature
        )
        if self.comparator.compare(self.leader_neighbors[0], self.solutions[0]) == -1:
            self.solutions[0] = self.leader_neighbors[0]
        elif acceptance_probability > random.random():
            self.solutions[0] = self.leader_neighbors[0]

    def population_improvement(self) -> None:
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 1])
        for j in range(1, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                else:
                    new_solution.survive_time = 0
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            acceptance_probability = self.compute_acceptance_probability(
                self.solutions[j].objectives[0], self.individual_neighbors[0].objectives[0], self.temperature
            )
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            elif acceptance_probability > random.random():
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        for k in range(0, self.para_x):
            self.previous_neighbors.append(self.leader_neighbors[2 * k + 2])
        for j in range(2, self.population_size, 2):
            self.individual_neighbors = copy.deepcopy(self.previous_neighbors)
            neighbor = self.solutions[j]
            for k in range(self.para_x, self.para_k):
                new_solution = self.mutation_operator.execute(neighbor)
                new_solution = self.evaluate_solution(new_solution)
                if self.comparator.compare(new_solution, neighbor) <= 0:
                    neighbor = new_solution
                if self.identical_solutions_comparator.compare(new_solution, self.solutions[j]) == 0:
                    for i in range(0, self.problem.number_of_objectives):
                        new_solution.objectives[i] = float("inf")
                else:
                    new_solution.survive_time = 0
                self.individual_neighbors.append(new_solution)
            self.individual_neighbors = self.sort_population.execute(self.individual_neighbors)
            acceptance_probability = self.compute_acceptance_probability(
                self.solutions[j].objectives[0], self.individual_neighbors[0].objectives[0], self.temperature
            )
            if self.comparator.compare(self.individual_neighbors[0], self.solutions[j]) == -1:
                self.solutions[j] = self.individual_neighbors[0]
            elif acceptance_probability > random.random():
                self.solutions[j] = self.individual_neighbors[0]
            self.previous_neighbors = copy.deepcopy(self.individual_neighbors[1:1 + self.para_x])
        self.previous_neighbors.clear()
        self.leader_neighbors.clear()
        self.individual_neighbors.clear()

    def move_leader(self) -> None:
        old_solutions = copy.deepcopy(self.solutions)
        if random.random() < 0.5:
            self.solutions[0] = old_solutions[1]
            for j in range(3, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 2] = old_solutions[0]
        else:
            self.solutions[0] = old_solutions[2]
            for j in range(4, self.population_size, 2):
                self.solutions[j - 2] = old_solutions[j]
            self.solutions[self.population_size - 1] = old_solutions[0]

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return numpy.exp(-1.0 * value)

    def scout_phase(self) -> None:
        for j in range(0, self.population_size):
            if hasattr(self.solutions[j], 'survive_time') and self.solutions[j].survive_time >= self.scout_time:
                neighbors = [self.mutation_operator.execute(self.solutions[j]) for _ in range(0, self.neighbor_size)]
                neighbors = self.evaluate(neighbors)
                neighbors = self.sort_population.execute(neighbors)
                for solution in neighbors[0:]:
                    if self.identical_solutions_comparator.compare(solution, self.solutions[j]) != 0:
                        self.solutions[j] = solution
                        break
                self.offspring_population_size += self.neighbor_size

    def evolve(self) -> None:
        self.offspring_population_size = 0
        for _ in range(0, self.para_m):
            self.offspring_population_size += self.para_k + (self.para_k - self.para_x) * (self.population_size - 1)
            for j in range(0, self.population_size):
                if hasattr(self.solutions[j], 'survive_time'):
                    self.solutions[j].survive_time += 1
                else:
                    self.solutions[j].survive_time = 1
            self.leader_improvement()
            self.population_improvement()
            self.scout_phase()
            self.temperature *= self.cooling_rate
        self.move_leader()

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
