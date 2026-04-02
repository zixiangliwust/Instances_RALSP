import random
import copy
import time
import threading
from math import sqrt
from typing import List, Optional, TypeVar

from evolu.config import store
from evolu.core.algorithm import DynamicAlgorithm
from evolu.ml import QLearning
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.problem import FloatProblem, DynamicProblem
from evolu.core.solution import FloatSolution
from evolu.core.operator import Mutation
from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation, \
    FloatPermutationIntegerSwapMutation, FloatPermutationIntegerInversionMutation, \
    FloatPermutationIntegerScrambleMutation, FloatOppositionMutation
from evolu.util.archive import BoundedArchive, NonDominatedSolutionsArchive, ArchiveWithReferencePoint
from evolu.util.comparator import DominanceWithConstraintsComparator, EpsilonDominanceComparator
from evolu.util.density_estimator import ModifiedCrowdingDistance
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = List[S]
"""
Module: OMOPSO (Multi-objective particle swarm optimization)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class OMOPSO(MultiObjectiveSwarmRoot[S, R]):
    """
    OMOPSO (Multi-objective particle swarm optimization)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Coello, C. A. C., G. T. Pulido, and M. S. Lechuga. 2004. "Handling multiple objectives with particle swarm optimization."
    IEEE transactions on evolutionary computation 8 (3):256-79. doi: 10.1109/TEVC.2004.826067.
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 uniform_mutation: FloatUniformMutation,
                 non_uniform_mutation: FloatNonUniformMutation,
                 leaders_archive: Optional[BoundedArchive],
                 epsilon: float,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        """
        super(OMOPSO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "OMOPSO"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion) 
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.__EPS = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.c1_min = 1.5
        self.c1_max = 2.0
        self.c2_min = 1.5
        self.c2_max = 2.0
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.w_min = 0.1
        self.w_max = 0.5
        self.change_velocity1 = -1
        self.change_velocity2 = -1
        self.velocity = [[0.0 for _ in range(self.problem.number_of_variables)] for _ in range(self.population_size)]
        self.local_best_solutions: List[S] = []
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        for i in range(self.population_size):
            for j in range(self.problem.number_of_variables):
                self.velocity[i][j] = 0.0
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
        self.leaders_archive.compute_density_estimator()
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(self.population_size):
            g_best = self.select_global_best()
            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)
            w = round(random.uniform(self.w_min, self.w_max), 1)
            for i in range(offsprings[j].number_of_variables):
                self.velocity[j][i] = (
                        w * self.velocity[j][i]
                        + (c1 * r1 * (self.local_best_solutions[j].variables[i] - offsprings[j].variables[i]))
                        + (c2 * r2 * (g_best.variables[i] - offsprings[j].variables[i]))
                )
        for j in range(self.population_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
                if offsprings[j].variables[i] < self.problem.lower_bound[i]:
                    offsprings[j].variables[i] = self.problem.lower_bound[i]
                    self.velocity[j][i] *= self.change_velocity1
                if offsprings[j].variables[i] > self.problem.upper_bound[i]:
                    offsprings[j].variables[i] = self.problem.upper_bound[i]
                    self.velocity[j][i] *= self.change_velocity2
        return offsprings

    def update_local_best(self, population: List[S]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(population[i], self.local_best_solutions[i])
            if flag != 1:
                self.local_best_solutions[i] = copy.deepcopy(population[i])

    def perturbation(self, population: List[S]) -> List[S]:
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        for i in range(self.population_size):
            if (i % 3) == 0:
                population[i] = self.non_uniform_mutation.execute(population[i])
            else:
                population[i] = self.uniform_mutation.execute(population[i])
        return population

    def select_global_best(self) -> FloatSolution:
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) > 2:
            solutions = random.sample(leaders_archive, 2)
            if self.leaders_archive.comparator.compare(solutions[0], solutions[1]) < 1:
                g_best = copy.deepcopy(solutions[0])
            else:
                g_best = copy.deepcopy(solutions[1])
        else:
            g_best = copy.deepcopy(self.leaders_archive.solution_list[0])
        return g_best

    def evolve(self):
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
        self.leaders_archive.compute_density_estimator()
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
Module: SMPSO (A new PSO-based metaheuristic)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class SMPSO(MultiObjectiveSwarmRoot[S, R]):
    """
    SMPSO (A new PSO-based metaheuristic)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Nebro, A. J., J. J. Durillo, J. Garcia-Nieto, C. A. Coello Coello, F. Luna, and E. Alba. 2009.
    SMPSO: A new PSO-based metaheuristic for multi-objective optimization. Paper presented at the 2009 IEEE Symposium on
    Computational Intelligence in Multi-Criteria Decision-Making(MCDM), 30 March-2 April 2009.
    """

    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 mutation: Mutation,
                 leaders_archive: Optional[BoundedArchive],
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param max_evaluations: Maximum number of evaluations/iterations.
        param mutation: Mutation operator (see :py:mod:`evolu.operator.mutation`).
        param leaders_archive: Archive for leaders.
        """
        super(SMPSO, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "SMPSO"
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.mutation_operator = mutation
        self.leaders_archive = leaders_archive
        self.c1_min = 1.5
        self.c1_max = 2.5
        self.c2_min = 1.5
        self.c2_max = 2.5
        self.r1_min = 0.0
        self.r1_max = 1.0
        self.r2_min = 0.0
        self.r2_max = 1.0
        self.w_min = 0.1
        self.w_max = 0.1
        self.change_velocity1 = -1
        self.change_velocity2 = -1
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.velocity = [[0.0 for _ in range(self.problem.number_of_variables)] for _ in range(self.population_size)]
        self.v_max, self.v_min = ([0.0] * problem.number_of_variables, [0.0] * problem.number_of_variables,)
        self.local_best_solutions: List[S] = []
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.leaders_archive.compute_density_estimator()
        self.local_best_solutions = copy.deepcopy(self.solutions)
        for solution in self.solutions:
            self.leaders_archive.add(solution)
        for i in range(self.problem.number_of_variables):
            self.v_max[i] = (self.problem.upper_bound[i] - self.problem.lower_bound[i]) / 2.0
        self.v_min = [-x for x in self.v_max]

    def __inertia_weight(self, wmax: float):
        return wmax

    def __constriction_coefficient(self, c1: float, c2: float) -> float:
        rho = c1 + c2
        if rho <= 4:
            result = 1.0
        else:
            result = 2.0 / (2.0 - rho - sqrt(pow(rho, 2.0) - 4.0 * rho))
        return result

    def __velocity_constriction(self, value: float, v_max: [], v_min: [], variable_index: int) -> float:
        result = value
        if value > v_max[variable_index]:
            result = v_max[variable_index]
        if value < v_min[variable_index]:
            result = v_min[variable_index]
        return result

    def update_position(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        for j in range(self.population_size):
            g_best = self.select_global_best()
            r1 = round(random.uniform(self.r1_min, self.r1_max), 1)
            r2 = round(random.uniform(self.r2_min, self.r2_max), 1)
            c1 = round(random.uniform(self.c1_min, self.c1_max), 1)
            c2 = round(random.uniform(self.c2_min, self.c2_max), 1)
            for i in range(offsprings[j].number_of_variables):
                self.velocity[j][i] = self.__velocity_constriction(
                    self.__constriction_coefficient(c1, c2)
                    * ((self.__inertia_weight(self.w_max) * self.velocity[j][i])
                       + (c1 * r1 * (self.local_best_solutions[j].variables[i] - offsprings[j].variables[i]))
                       + (c2 * r2 * (g_best.variables[i] - offsprings[j].variables[i]))
                       ), self.v_max, self.v_min, i, )
        for j in range(self.population_size):
            for i in range(offsprings[j].number_of_variables):
                offsprings[j].variables[i] += self.velocity[j][i]
                if offsprings[j].variables[i] < self.problem.lower_bound[i]:
                    offsprings[j].variables[i] = self.problem.lower_bound[i]
                    self.velocity[j][i] *= self.change_velocity1
                if offsprings[j].variables[i] > self.problem.upper_bound[i]:
                    offsprings[j].variables[i] = self.problem.upper_bound[i]
                    self.velocity[j][i] *= self.change_velocity2
        return offsprings

    def update_local_best(self, population: List[S]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(population[i], self.local_best_solutions[i])
            if flag != 1:
                self.local_best_solutions[i] = copy.deepcopy(population[i])

    def perturbation(self, population: List[S]) -> List[S]:
        for i in range(self.population_size):
            if (i % 6) == 0:
                population[i] = self.mutation_operator.execute(population[i])
        return population

    def select_global_best(self) -> FloatSolution:
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) > 2:
            solutions = random.sample(leaders_archive, 2)
            if self.leaders_archive.comparator.compare(solutions[0], solutions[1]) < 1:
                g_best = copy.deepcopy(solutions[0])
            else:
                g_best = copy.deepcopy(solutions[1])
        else:
            g_best = copy.deepcopy(self.leaders_archive.solution_list[0])
        return g_best

    def evolve(self):
        self.solutions = self.update_position(self.solutions)
        self.solutions = self.perturbation(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.leaders_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            self.leaders_archive.add(solution)
        self.leaders_archive.compute_density_estimator()
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class DynamicSMPSO(SMPSO, DynamicAlgorithm):
    def __init__(self,
                 problem: DynamicProblem[FloatSolution],
                 population_size: int,
                 mutation: Mutation,
                 leaders_archive: BoundedArchive,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 ):
        super(DynamicSMPSO, self).__init__(
            problem=problem,
            population_size=population_size,
            mutation=mutation,
            leaders_archive=leaders_archive,
            termination_criterion=termination_criterion,
            population_generator=population_generator,
            population_evaluator=population_evaluator,
        )
        self.completed_iterations = 0

    def restart(self) -> None:
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)
        self.leaders_archive.__init__(self.leaders_archive.maximum_size)
        for i in range(self.problem.number_of_variables):
            self.v_max[i] = (self.problem.upper_bound[i] - self.problem.lower_bound[i]) / 2.0
        self.v_min = [-x for x in self.v_max]
        self.local_best_solutions = copy.deepcopy(self.solutions)
        for solution in self.solutions:
            self.leaders_archive.add(solution)
        self.init_progress()

    def update_progress(self):
        if self.problem.the_problem_has_changed():
            self.restart()
            self.problem.clear_changed()
        self.evaluations += self.offspring_population_size
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.leaders_archive.compute_density_estimator()

    def stopping_condition_is_met(self):
        if self.termination_criterion.is_met:
            observable_data = self.get_observable_data()
            observable_data["termination_criterion_is_met"] = True
            self.observable.notify_all(**observable_data)
            self.restart()
            self.init_progress()
            self.completed_iterations += 1
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class SMPSORP(SMPSO):
    def __init__(self,
                 problem: FloatProblem,
                 population_size: int,
                 mutation: Mutation,
                 reference_points: List[List[float]],
                 leaders_archive: List[ArchiveWithReferencePoint],
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 ):
        """This class implements the SMPSORP algorithm.
     param problem: The problem to solve.
     param population_size:
     param mutation:
     param leaders_archive: List of bounded archives.
     param population_evaluator: An evaluator object to evaluate the solutions in the population.
        """
        super(SMPSORP, self).__init__(
            problem=problem,
            population_size=population_size,
            mutation=mutation,
            leaders_archive=None,
            population_generator=population_generator,
            population_evaluator=population_evaluator,
            termination_criterion=termination_criterion,
        )
        self.leaders_archive = leaders_archive
        self.reference_points = reference_points
        self.lock = threading.Lock()
        thread = threading.Thread(target=_change_reference_point, args=(self,))
        thread.start()

    def select_global_best(self) -> FloatSolution:
        selected = False
        selected_swarm_index = 0
        while not selected:
            selected_swarm_index = random.randint(0, len(self.leaders_archive) - 1)
            if len(self.leaders_archive[selected_swarm_index].solution_list) != 0:
                selected = True
        leaders_archive = self.leaders_archive[selected_swarm_index].solution_list
        if len(leaders_archive) > 2:
            solutions = random.sample(leaders_archive, 2)
            if self.leaders_archive[selected_swarm_index].comparator.compare(solutions[0], solutions[1]) < 1:
                g_best = copy.deepcopy(solutions[0])
            else:
                g_best = copy.deepcopy(solutions[1])
        else:
            g_best = copy.deepcopy(self.leaders_archive[selected_swarm_index].solution_list[0])
        return g_best

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        for i in range(self.problem.number_of_variables):
            self.v_max[i] = (self.problem.upper_bound[i] - self.problem.lower_bound[i]) / 2.0
        self.v_min = [-x for x in self.v_max]
        self.local_best_solutions = copy.deepcopy(self.solutions)
        for leader in self.leaders_archive:
            leader.compute_density_estimator()
        for solution in self.solutions:
            for leader in self.leaders_archive:
                leader.add(copy.deepcopy(solution))

    def update_progress(self) -> None:
        self.evaluations += self.offspring_population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["REFERENCE_POINT"] = self.get_reference_point()
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            for leader in self.leaders_archive:
                leader.add(copy.deepcopy(solution))
        for leader in self.leaders_archive:
            leader.filter()
            leader.compute_density_estimator()

    def update_reference_point(self, new_reference_points: list):
        with self.lock:
            self.reference_points = new_reference_points
            for index, archive in enumerate(self.leaders_archive):
                archive.update_reference_point(new_reference_points[index])

    def get_reference_point(self):
        with self.lock:
            return self.reference_points

    def get_name(self) -> str:
        return "SMPSORP"
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


def _change_reference_point(algorithm: SMPSORP):
    """Auxiliar function to read new reference points from the keyboard for the SMPSORP algorithm"""
    number_of_reference_points = len(algorithm.reference_points)
    number_of_objectives = algorithm.problem.number_of_objectives
    while True:
        print(f"Enter {number_of_reference_points}-points of dimension {number_of_objectives}: ")
        read = [float(x) for x in input().split()]
        # Update reference points
        reference_points = []
        for i in range(0, len(read), number_of_objectives):
            reference_points.append(read[i: i + number_of_objectives])
        algorithm.update_reference_point(reference_points)
