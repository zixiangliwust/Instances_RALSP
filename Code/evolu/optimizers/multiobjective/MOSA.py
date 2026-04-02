# -*- coding: utf-8 -*-
import copy
import random
import math
import time
from typing import List, TypeVar
from evolu.config import store
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.core.solution import Solution
from evolu.util.comparator import Comparator, ObjectiveComparator, DominanceWithConstraintsComparator, \
    EpsilonDominanceComparator
from evolu.util.archive import NonDominatedSolutionsArchive, ModifiedNonDominatedSolutionsArchive
from evolu.util.density_estimator import ModifiedCrowdingDistance
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger
from evolu.core.algorithm import MultiObjectiveSwarmRoot

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
Module: Multi-objective simulated annealing
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MSAA(MultiObjectiveSwarmRoot[S, R]):
    """
    Multi-objective simulated annealing
    References:
    [1] Cakir, Burcin, Fulya Altiparmak, and Berna Dengiz. 2011. "Multi-objective optimization of a stochastic assembly line balancing: A hybrid simulated annealing algorithm."
    Computers & Industrial Engineering 60 (3):376-84. doi: http://dx.doi.org/10.1016/j.cie.2010.08.013.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MSAA, self).__init__(problem=problem)
        self.algorithm_name = "Multi-objective simulated annealing"
        self.population_size = 1  # SA typically works with single solution
        self.offspring_population_size = 1
        self.mutation_operator = mutation
        self.comparator = comparator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.leaders_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.offspring_population_size = self.sub_max_iteration
        self.non_improvement_time_limit = 10
        self.non_improvement_time = 0
        self.solution = None  # Initialize solution attribute

    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)
        
    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = 1
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.leaders_archive.add(self.solution)

    def evolve(self) -> None:
        for _ in range(0, self.sub_max_iteration):
            self.non_improvement_time += 1
            new_solution: Solution = self.mutation_operator.execute(self.solution)
            new_solution = self.evaluate_solution(new_solution)
            success = self.leaders_archive.add(new_solution)
            if success:
                self.non_improvement_time = 0
                self.solution = new_solution
            else:
                objective_index = random.randint(0, self.solution.number_of_objectives - 1)
                acceptance_probability = self.compute_acceptance_probability(
                    self.solution.objectives[objective_index], new_solution.objectives[objective_index],
                    self.temperature
                )
                self.comparator = ObjectiveComparator(objective_index)
                if self.comparator.compare(new_solution, self.solution) == -1:
                    self.solution = new_solution
                elif acceptance_probability > random.random():
                    self.solution = new_solution
            self.restart()
        self.temperature *= self.cooling_rate

    def restart(self):
        if self.non_improvement_time > self.non_improvement_time_limit:
            idx = random.randint(0, len(self.leaders_archive.solution_list) - 1)
            self.solution = copy.deepcopy(self.leaders_archive.solution_list[idx])
            self.non_improvement_time = 0

    def compute_acceptance_probability(self, current_score: float, new_score: float, temperature: float) -> float:
        if new_score <= current_score:
            return 1.0
        else:
            t = temperature if temperature > self.minimum_temperature else self.minimum_temperature
            if current_score == 0.0:
                current_score = 1.0e-14
            value = (new_score - current_score) / (t * current_score)
            return math.exp(-1.0 * value)

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name



"""
Module: Restarted simulated annealing
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MRSA(MSAA[S, R]):
    """
    Restarted simulated annealing
    References:
    [1] Li, Zixiang, Qiuhua Tang, and LiPing Zhang. 2016. "Minimizing energy consumption and cycle time in two-sided robotic assembly line systems using restarted simulated annealing algorithm."
    Journal of Cleaner Production 135:508-22. doi: http://dx.doi.org/10.1016/j.jclepro.2016.06.131.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(MRSA, self).__init__(problem=problem, mutation=mutation, comparator=comparator,
                                   termination_criterion=termination_criterion, )
        # Note: MSAA parent class handles operator assignments properly
        self.population_size = 1  # SA typically works with single solution
        self.offspring_population_size = 1
        self.algorithm_name = "Restarted simulated annealing"
        self.leaders_archive = ModifiedNonDominatedSolutionsArchive()
        self.density_estimator = ModifiedCrowdingDistance()
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.non_improvement_time_limit = 10
        self.non_improvement_time = 0
        self.offspring_population_size = self.sub_max_iteration
        
    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)

    def restart(self):
        if self.non_improvement_time > self.non_improvement_time_limit:
            self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
            sorted_population = self.leaders_archive.solution_list
            self.density_estimator.sort(sorted_population)
            self.solution = copy.deepcopy(sorted_population[0])
            sorted_population[0].survive_time += 1
            self.non_improvement_time = 0

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name




"""
Module: Improved restarted simulated annealing
Creator: Zixiang Li, zixiangliwust@gmail.com;`n[1] Chenyu Zheng, Wuhan University of Science and Technology, z173570209@163.com;
[2] Zixiang Li, Wuhan University of Science and Technology, https://www.researchgate.net/profile/Zixiang-Li-2, zixiangliwust@gmail.com;
"""

class IMRSA(MSAA[S, R]):
    """
    Improved restarted simulated annealing
    References:
    [1] Zheng Chenyu, Li, Zixiang, Zhang, Zikai, LiPing Zhang and Tang, Qiuhua. 2024. "Multi-objective restarted simulated annealing algorithm for assembly line balancing problem with collaborative robots considering ergonomics risks."
    Flexible Service and Manufacturing Journal, under review.
    """

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 comparator: Comparator = store.default_comparator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        super(IMRSA, self).__init__(problem=problem, mutation=mutation, comparator=comparator,
                                   termination_criterion=termination_criterion, )
        # Note: MSAA parent class handles operator assignments properly
        self.population_size = 1  # SA typically works with single solution
        self.offspring_population_size = 1
        self.algorithm_name = "Improved Restarted simulated annealing"
        self.leaders_archive = ModifiedNonDominatedSolutionsArchive()
        self.density_estimator = ModifiedCrowdingDistance()
        self.temperature_start = 1.0
        self.temperature = self.temperature_start
        self.sub_max_iteration = 5
        self.minimum_temperature = 0.000001
        self.cooling_rate = 0.95
        self.offspring_population_size = self.sub_max_iteration
        self.non_improvement_time_limit = 10
        self.non_improvement_time = 0
        
    def initialization(self) -> None:
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solution = self.create_solution()
        self.solution = self.evaluate_solution(self.solution)

    def evolve(self) -> None:
        for _ in range(0, self.sub_max_iteration):
            self.non_improvement_time += 1
            new_solution: Solution = self.mutation_operator.execute(self.solution)
            new_solution = self.evaluate_solution(new_solution)
            success = self.leaders_archive.add(new_solution)
            if success:
                self.non_improvement_time = 0
                self.solution = new_solution
            else:
                # IMRSA核心改进：对多个目标计算接受概率并相�?
                acceptance_probability = 1.0
                # 对除最后一个外的所有目标计算接受概率并相乘
                for objective_index in range(0, self.solution.number_of_objectives - 1):
                    acceptance_probability *= self.compute_acceptance_probability(
                        self.solution.objectives[objective_index], new_solution.objectives[objective_index],
                        self.temperature
                    )
                
                if acceptance_probability > random.random():
                    self.solution = new_solution
            self.restart()
        self.temperature *= self.cooling_rate

    def restart(self):
        if self.non_improvement_time > self.non_improvement_time_limit:
            self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
            sorted_population = self.leaders_archive.solution_list
            self.density_estimator.sort(sorted_population)
            self.solution = copy.deepcopy(sorted_population[0])
            sorted_population[0].survive_time += 1
            self.non_improvement_time = 0

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


