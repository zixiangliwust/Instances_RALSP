# -*- coding: utf-8 -*-
import random
import copy
import math
import numpy as np
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from evolu.config import store
from evolu.core.algorithm import SingleObjectiveSwarmRoot
from evolu.core.solution import Solution
from evolu.core.problem import FloatProblem
from evolu.util.comparator import Comparator
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.core.operator import Crossover, Mutation, Selection
from evolu.util.termination_criterion import TerminationCriterion
from evolu.util.neighborhood import Neighborhood
from evolu.util.distance import EuclideanDistance
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Particle swarm optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PSOBase(SingleObjectiveSwarmRoot[S, R]):
    """
    Particle swarm optimization
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] https://github.com/jMetal/jMetalPy, Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
    [3] https://github.com/7ossam81/EvoloPy, Hossam Faris etc., hossam.faris@ju.edu.jo (H. Faris)
    [4] Kennedy, J., and R. Eberhart. 1995. Particle swarm optimization. Paper presented at the Proceedings of ICNN'95 - International Conference on Neural Networks, 27 Nov.-1 Dec. 1995.
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PSOBase, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Particle swarm optimization"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.c1 = 1.0
        self.c2 = 2.0
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.w_min = 0.1
        self.w_max = 0.5
        self.v_max = []
        self.v_min = []
        self.velocity = np.zeros((self.population_size, self.problem.number_of_variables), dtype=float)
        self.local_best_solutions: List[S] = []

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = copy.deepcopy(self.solutions)
        self.v_max = [0.5 * (self.problem.upper_bound[i] - self.problem.lower_bound[i]) for i in
                      range(self.problem.number_of_variables)]
        self.v_min = [-self.v_max[i] for i in range(0, len(self.v_max))]
        for j in range(self.population_size):
            for i in range(self.problem.number_of_variables):
                self.velocity[j][i] = random.uniform(self.v_min[i], self.v_max[i])

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        wk = self.w_max - self.iterations * ((self.w_max - self.w_min) / self.max_iterations)
        for j in range(self.population_size):
            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            for i in range(offsprings[j].number_of_variables):
                self.velocity[j][i] = (
                        wk * self.velocity[j][i]
                        + (self.c1 * r1 * (self.local_best_solutions[j].variables[i] - offsprings[j].variables[i]))
                        + (self.c2 * r2 * (self.g_best.variables[i] - offsprings[j].variables[i]))
                )
                if self.velocity[j][i] < self.v_min[i]:
                    self.velocity[j][i] = self.v_min[i]
                if self.velocity[j][i] > self.v_max[i]:
                    self.velocity[j][i] = self.v_max[i]
        for j in range(self.population_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
        return offsprings

    def update_local_best(self, population: List[S]) -> None:
        for j in range(self.population_size):
            flag = self.comparator.compare(population[j], self.local_best_solutions[j])
            if flag != 1:
                self.local_best_solutions[j] = copy.deepcopy(population[j])

    def evolve(self) -> None:
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class AdaptiveRandomNeighborhood(Neighborhood[Solution], ABC):
    def __init__(self,
                 number_of_neighborhood_vectors: int,
                 neighborhood_size: int,
                 ):
        self.number_of_neighborhood_vectors = number_of_neighborhood_vectors
        self.neighborhood_size = neighborhood_size
        self.neighborhood = []
        for i in range(self.number_of_neighborhood_vectors):
            idx_list = [i]
            self.neighborhood.append(idx_list)
        for i in range(self.number_of_neighborhood_vectors):
            for j in range(self.neighborhood_size):
                idx_random = random.randint(0, self.number_of_neighborhood_vectors - 1)
                if i not in self.neighborhood[idx_random]:
                    self.neighborhood[idx_random].append(i)

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        neighbors_indexes = self.neighborhood[index]
        if any(i > len(solution_list) for i in neighbors_indexes):
            raise IndexError("Neighbor index out of range")
        return [solution_list[i] for i in neighbors_indexes]

    def get_best_neighbors(self, comparator: Comparator, solution_list: List[Solution]) -> List[Solution]:
        result = []
        for i in range(len(0, self.neighborhood)):
            best_solution = solution_list[self.neighborhood[i][0]]
            for j in range(1, len(self.neighborhood[i])):
                if comparator.compare(best_solution, solution_list[self.neighborhood[i][j]]) == 1:
                    best_solution = solution_list[self.neighborhood[i][j]]
            result.append(best_solution)
        return result

    def recompute(self) -> None:
        self.neighborhood.clear()
        for i in range(self.number_of_neighborhood_vectors):
            idx_list = [i]
            self.neighborhood.append(idx_list)
        for i in range(self.number_of_neighborhood_vectors):
            for j in range(self.neighborhood_size):
                idx_random = random.randint(0, self.number_of_neighborhood_vectors - 1)
                if i not in self.neighborhood[idx_random]:
                    self.neighborhood[idx_random].append(i)

    def get_neighborhood(self) -> List[List]:
        return self.neighborhood


"""
Module: Standard particle swarm optimization in 2007
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PSO2007(SingleObjectiveSwarmRoot[S, R]):
    """
    Standard particle swarm optimization in 2007
    References:
    [1] Initial code built based on https:// github.com/adajani/jMetalCpp by Antonio J. Nebro <antonio@lcc.uma.es>,
    Juan J. Durillo <durillo@lcc.uma.es>,Esteban López-Camacho <esteban@lcc.uma.es>
    [2] Bratton, D., and J. Kennedy. 2007. Defining a Standard for Particle Swarm Optimization.
    Paper presented at the 2007 IEEE Swarm Intelligence Symposium, 1-5 April 2007.
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PSO2007, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Particle swarm optimization 2007"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.W = 1.0 / (2.0 * math.log(2))
        self.C = 1.0 / 2.0 + math.log(2)
        self.neighborhood_size = 3
        self.adaptive_random_neighborhood = AdaptiveRandomNeighborhood(self.population_size, self.neighborhood_size)
        self.local_best_solutions: List[S] = []
        self.best_neighbor_solutions: List[S] = []
        self.velocity = np.zeros((self.population_size, self.problem.number_of_variables), dtype=float)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = copy.deepcopy(self.solutions)
        for j in range(0, self.population_size):
            self.best_neighbor_solutions.append(self.get_best_neighbor(j))
        for j in range(self.population_size):
            for i in range(self.problem.number_of_variables):
                self.velocity[j][i] = (random.uniform(self.solutions[j].lower_bound[i],
                                                      self.solutions[j].upper_bound[i]) - self.solutions[j].variables[
                                           i]) / 2.0

    def get_best_neighbor(self, index: int) -> Solution:
        best_local_best_solution = None
        for i in self.adaptive_random_neighborhood.neighborhood[index]:
            if best_local_best_solution is None or self.comparator.compare(best_local_best_solution,
                                                                           self.local_best_solutions[i]) == 1:
                best_local_best_solution = self.local_best_solutions[i]
        return best_local_best_solution

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(self.population_size):
            r1 = random.uniform(0, self.C)
            r2 = random.uniform(0, self.C)
            if self.local_best_solutions[j] != self.best_neighbor_solutions[j]:
                for i in range(offsprings[j].number_of_variables):
                    self.velocity[j][i] = (
                            self.W * self.velocity[j][i]
                            + r1 * (self.local_best_solutions[j].variables[i] - offsprings[j].variables[i])
                            + r2 * (self.best_neighbor_solutions[j].variables[i] - offsprings[j].variables[i])
                    )
            else:
                for i in range(offsprings[j].number_of_variables):
                    self.velocity[j][i] = (
                            self.W * self.velocity[j][i]
                            + r1 * (self.local_best_solutions[j].variables[i] - offsprings[j].variables[i])
                    )
        for j in range(self.population_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
                if offsprings[j].variables[i] < self.problem.lower_bound[i]:
                    offsprings[j].variables[i] = self.problem.lower_bound[i]
                    self.velocity[j][i] = 0
                if offsprings[j].variables[i] > self.problem.upper_bound[i]:
                    offsprings[j].variables[i] = self.problem.upper_bound[i]
                    self.velocity[j][i] = 0
        return offsprings

    def update_local_best(self, population: List[S]) -> None:
        for j in range(self.population_size):
            flag = self.comparator.compare(population[j], self.local_best_solutions[j])
            if flag != 1:
                self.local_best_solutions[j] = copy.deepcopy(population[j])

    def evolve(self) -> None:
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def after_evolve(self) -> None:
        for j in range(0, self.population_size):
            self.best_neighbor_solutions[j] = self.get_best_neighbor(j)
        if self.comparator.compare(self.p_best, self.g_best) == 0:
            self.adaptive_random_neighborhood.recompute()
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Standard particle swarm optimization in 2011
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PSO2011(SingleObjectiveSwarmRoot[S, R]):
    """
    Standard particle swarm optimization in 2011
    References:
    [1] Initial code built based on https:// github.com/adajani/jMetalCpp by Antonio J. Nebro <antonio@lcc.uma.es>,
    Juan J. Durillo <durillo@lcc.uma.es>,Esteban López-Camacho <esteban@lcc.uma.es>
    [2] Clerc, M. (2011). Standard particle swarm optimisation. Technical Report.
    http://clerc.maurice.free.fr/pso/SPSO_descriptions.pdf
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PSO2011, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Particle swarm optimization 2011"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.W = 1.0 / (2.0 * math.log(2))
        self.C = 1.0 / 2.0 + math.log(2)
        self.ChVel = -0.5
        self.neighborhood_size = 3
        self.adaptive_random_neighborhood = AdaptiveRandomNeighborhood(self.population_size, self.neighborhood_size)
        self.local_best_solutions: List[S] = []
        self.best_neighbor_solutions: List[S] = []
        self.velocity = np.zeros((self.population_size, self.problem.number_of_variables), dtype=float)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = copy.deepcopy(self.solutions)
        for j in range(0, self.population_size):
            self.best_neighbor_solutions.append(self.get_best_neighbor(j))
        for j in range(self.population_size):
            for i in range(self.problem.number_of_variables):
                self.velocity[j][i] = random.uniform(self.solutions[j].lower_bound[i] - self.solutions[j].variables[i],
                                                     self.solutions[j].upper_bound[i] - self.solutions[j].variables[i])

    def get_best_neighbor(self, index: int) -> Solution:
        best_local_best_solution = None
        for i in self.adaptive_random_neighborhood.neighborhood[index]:
            if best_local_best_solution is None or self.comparator.compare(best_local_best_solution,
                                                                           self.local_best_solutions[i]) == 1:
                best_local_best_solution = self.local_best_solutions[i]
        return best_local_best_solution

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(self.population_size):
            gravity_center_solution = self.create_solution()
            random_solution = copy.deepcopy(offsprings[j])
            if self.local_best_solutions[j] != self.best_neighbor_solutions[j]:
                for i in range(offsprings[j].number_of_variables):
                    gravity_center_solution.variables[i] = offsprings[j].variables[i] + self.C * (
                            self.local_best_solutions[j].variables[i] + self.best_neighbor_solutions[j].variables[
                        i] - 2 * offsprings[j].variables[i]) / 3.0
            else:
                for i in range(offsprings[j].number_of_variables):
                    gravity_center_solution.variables[i] = offsprings[j].variables[i] + self.C * (
                            self.local_best_solutions[j].variables[i] - offsprings[j].variables[i]) / 2.0
            radius = EuclideanDistance().get_distance(gravity_center_solution.variables, offsprings[j].variables)
            random_list = self.rand_sphere(offsprings[j].number_of_variables)
            for i in range(offsprings[j].number_of_variables):
                random_solution.variables[i] = gravity_center_solution.variables[i] + radius * random_list[i]
            for i in range(offsprings[j].number_of_variables):
                self.velocity[j][i] = self.W * self.velocity[j][i] + random_solution.variables[i] - \
                                      offsprings[j].variables[i]
        for j in range(self.population_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
                if offsprings[j].variables[i] < self.problem.lower_bound[i]:
                    offsprings[j].variables[i] = self.problem.lower_bound[i]
                    self.velocity[j][i] = self.ChVel * self.velocity[j][i]
                if offsprings[j].variables[i] > self.problem.upper_bound[i]:
                    offsprings[j].variables[i] = self.problem.upper_bound[i]
                    self.velocity[j][i] = self.ChVel * self.velocity[j][i]
        return offsprings

    def rand_sphere(self, dimension: int) -> list:
        x = [0.0 for _ in range(0, dimension)]
        length = 0.0
        for i in range(0, dimension):
            x[i] = np.random.normal(0, 1)
            length += length + x[i] * x[i]
        length = math.sqrt(length)
        r = random.random()
        for i in range(0, dimension):
            x[i] = r * x[i] / length
        return x

    def update_local_best(self, population: List[S]) -> None:
        for j in range(self.population_size):
            flag = self.comparator.compare(population[j], self.local_best_solutions[j])
            if flag != 1:
                self.local_best_solutions[j] = copy.deepcopy(population[j])

    def evolve(self) -> None:
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def after_evolve(self) -> None:
        for j in range(0, self.population_size):
            self.best_neighbor_solutions[j] = self.get_best_neighbor(j)
        if self.comparator.compare(self.p_best, self.g_best) == 0:
            self.adaptive_random_neighborhood.recompute()
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Phasor particle swarm optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class PSOPhasor(PSOBase):
    """
    Phasor particle swarm optimization
    References:
    [1] Initial code built based on https://github.com/thieu1995/mealpy, Nguyen Van Thieu,nguyenthieu2102@gmail.com
    [2] Code by Nguyen Van Thieu is converted from matlab code (sent from author: Ebrahim Akbari)
    [3] Ghasemi, Mojtaba, Ebrahim Akbari, Abolfazl Rahimnejad, Seyed Ehsan Razavi, Sahand Ghavidel, and Li Li. 2019. "Phasor particle swarm optimization: a simple and efficient variant of PSO."
    Soft Computing 23 (19):9701-18. doi: 10.1007/s00500-018-3536-8.
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(PSOPhasor, self).__init__(problem=problem,
                                        population_size=population_size,
                                        population_generator=population_generator,
                                        population_evaluator=population_evaluator,
                                        termination_criterion=termination_criterion,
                                        )
        self.algorithm_name = "Phasor particle swarm optimization"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.delta_list = []
        self.v_max_list = np.zeros((self.population_size, self.problem.number_of_variables), dtype=float)
        self.v_min_list = np.zeros((self.population_size, self.problem.number_of_variables), dtype=float)

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = copy.deepcopy(self.solutions)
        for j in range(self.population_size):
            self.v_max_list[j] = [0.5 * (self.problem.upper_bound[i] - self.problem.lower_bound[i]) for i in
                                  range(0, self.problem.number_of_variables)]
            self.v_min_list[j] = [-self.v_max_list[j][i] for i in range(0, self.problem.number_of_variables)]
        for j in range(self.population_size):
            for i in range(self.problem.number_of_variables):
                self.velocity[j][i] = random.uniform(self.v_min_list[j][i], self.v_max_list[j][i])
        self.delta_list = [np.random.uniform(0, 2 * np.pi) for _ in range(0, self.population_size)]

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(self.solutions)
        for j in range(0, self.population_size):
            p_delta = np.abs(np.cos(self.delta_list[j])) ** (2 * (np.sin(self.delta_list[j])))
            g_delta = np.abs(np.sin(self.delta_list[j])) ** (2 * (np.cos(self.delta_list[j])))
            for i in range(self.problem.number_of_variables):
                self.velocity[j][i] = p_delta * (
                        self.local_best_solutions[j].variables[i] - offsprings[j].variables[i]) + g_delta * (
                                              self.g_best.variables[i] - offsprings[j].variables[i])
            for i in range(offsprings[j].number_of_variables):
                if self.velocity[j][i] < self.v_min_list[j][i]:
                    self.velocity[j][i] = self.v_min_list[j][i]
                if self.velocity[j][i] > self.v_max_list[j][i]:
                    self.velocity[j][i] = self.v_max_list[j][i]
        for j in range(0, self.population_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
        for j in range(0, self.population_size):
            self.delta_list[j] += np.abs(np.cos(self.delta_list[j]) + np.sin(self.delta_list[j])) * (2 * np.pi)
            for i in range(offsprings[j].number_of_variables):
                self.v_max_list[j][i] = (np.abs(np.cos(self.delta_list[j])) ** 2) * (
                        self.problem.upper_bound[i] - self.problem.lower_bound[i])
            self.v_min_list[j] = [-self.v_max_list[j][i] for i in range(0, self.problem.number_of_variables)]
        return offsprings

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: Discrete particle swarm optimization
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class DiscretePSO(SingleObjectiveSwarmRoot[S, R]):
    """
    Discrete particle swarm optimization
    References:
    [1] Li, Zixiang, Mukund Nilakantan Janardhanan, Qiuhua Tang, and Peter Nielsen. 2016.
    "Co-evolutionary particle swarm optimization algorithm for two-sided robotic assembly line balancing problem."
    Advances in Mechanical Engineering 8 (9):1-14. doi: 10.1177/1687814016667907.
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 crossover: Crossover,
                 mutation: Mutation,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(DiscretePSO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "Discrete particle swarm optimization"
        self.crossover_operator = crossover
        self.mutation_operator = mutation
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.w_k = 0.2
        self.c1 = 0.1
        self.local_best_solutions: List[S] = []

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = []
        for j in range(0, self.population_size):
            if random.random() < self.w_k:
                new_solutions = self.crossover_operator.execute([population[j], self.g_best])
            else:
                new_solutions = self.crossover_operator.execute([population[j], self.local_best_solutions[j]])
            if random.random() < self.c1:
                offsprings.append(self.mutation_operator.execute(new_solutions[0]))
            else:
                offsprings.append(new_solutions[0])
        return offsprings

    def update_local_best(self, population: List[S]) -> None:
        for j in range(self.population_size):
            flag = self.comparator.compare(population[j], self.local_best_solutions[j])
            if flag != 1:
                self.local_best_solutions[j] = copy.deepcopy(population[j])

    def evolve(self) -> None:
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
