import copy
import random
from math import ceil
from typing import Generator, List, TypeVar
from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.operator import Mutation
from evolu.core.problem import Problem
from evolu.operator.crossover import FloatDifferentialEvolutionCrossover
from evolu.operator.selection import NaryRandomSolutionSelection
from evolu.util.archive import NonDominatedSolutionsArchive
from evolu.util.aggregation_function import AggregationFunction
from evolu.util.comparator import DominanceWithConstraintsComparator, EpsilonDominanceComparator
from evolu.util.constraint_handling import feasibility_ratio, is_feasible, overall_constraint_violation_degree
from evolu.util.density_estimator import CrowdingDistance
from evolu.util.evaluator import Evaluator
from evolu.util.neighborhood import WeightVectorNeighborhood
from evolu.util.ranking import FastNonDominatedRanking
from evolu.util.termination_criterion import StoppingByEvaluations, TerminationCriterion
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")
R = List[S]


class Permutation:
    def __init__(self, length: int):
        self.counter = 0
        self.length = length
        self.permutation = list(range(length))
        random.shuffle(self.permutation)

    def get_next_value(self):
        next_value = self.permutation[self.counter]
        self.counter += 1
        if self.counter == self.length:
            self.permutation = list(range(self.length))
            random.shuffle(self.permutation)
            self.counter = 0
        return next_value

    def get_permutation(self):
        return self.permutation


"""
Module: MOEAD (Multi-objective evolutionary algorithm based on decomposition)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOEAD(MultiObjectiveSwarmRoot[S, R]):
    """
    MOEAD (Multi-objective evolutionary algorithm based on decomposition)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Zhang, Q., and H. Li. 2007. "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition."
    IEEE transactions on evolutionary computation 11 (6):712-31. doi: 10.1109/TEVC.2007.892759.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 crossover: FloatDifferentialEvolutionCrossover,
                 mutation: Mutation,
                 aggregation_function: AggregationFunction,
                 neighborhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 neighbor_size: int,
                 weight_files_path: str,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 ):
        """
        Input parameters:
        max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        neighborhood_selection_probability: Probability of mating with a solution in the neighborhood rather
        than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEAD, self).__init__(problem=problem, population_size=population_size, )
        self.algorithm_name = "MOEAD"
        self.offspring_population_size = 1
        self.selection_operator = NaryRandomSolutionSelection(2)
        self.crossover_operator = crossover
        self.mutation_operator = mutation
        self.objective_function = aggregation_function
        self.neighborhood_selection_probability = neighborhood_selection_probability
        self.max_number_of_replaced_solutions = max_number_of_replaced_solutions
        self.neighborhood = WeightVectorNeighborhood(
            number_of_weight_vectors=population_size,
            neighborhood_size=neighbor_size,
            weight_vector_size=problem.number_of_objectives,
            weights_path=weight_files_path,
        )
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.permutation = None
        self.current_solution_id = 0
        self.neighbor_type = None
        self.result_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(0.0075))

    def init_progress(self) -> None:
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.permutation = Permutation(self.population_size)
        for solution in self.solutions:
            self.objective_function.update(solution.objectives)

    def selection(self, population: List[S]):
        rnd = random.random()
        if rnd < self.neighborhood_selection_probability:
            self.neighbor_type = "NEIGHBOR"
        else:
            self.neighbor_type = "POPULATION"
        self.current_solution_id = self.permutation.get_next_value()
        if self.neighbor_type == "NEIGHBOR":
            neighbors = self.neighborhood.get_neighbors(self.current_solution_id, population)
            selected_solutions = self.selection_operator.execute(neighbors)
        else:
            selected_solutions = self.selection_operator.execute(population)
        selected_solutions.append(population[self.current_solution_id])
        return selected_solutions

    def reproduction(self, population: List[S]) -> List[S]:
        offsprings = []
        offspring = self.crossover_operator.execute(self.solutions[self.current_solution_id], population)
        offsprings.append(offspring)
        new_solution = self.mutation_operator.execute(offsprings[0])
        offsprings[0] = new_solution
        return offsprings

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        new_solution = offsprings[0]
        self.objective_function.update(new_solution.objectives)
        new_solutions = self.update_current_subproblem_neighborhood(new_solution, population)
        return new_solutions

    def update_current_subproblem_neighborhood(self, new_solution, population):
        if self.neighbor_type == "NEIGHBOR":
            neighbors = self.neighborhood.get_neighborhood()[self.current_solution_id]
            permuted_neighbors_indexes = copy.deepcopy(neighbors.tolist())
        else:
            permuted_neighbors_indexes = Permutation(self.population_size).get_permutation()
        # Create a deep copy of population to match C++ behavior
        new_solutions = copy.deepcopy(population)
        replacements = 0
        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]
            f1 = self.objective_function.compute(population[k].objectives, self.neighborhood.weight_vectors[k])
            f2 = self.objective_function.compute(new_solution.objectives, self.neighborhood.weight_vectors[k])
            if f2 < f1:
                new_solutions[k] = copy.deepcopy(new_solution)
                replacements += 1
            if replacements >= self.max_number_of_replaced_solutions:
                break
        return new_solutions

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
MOEAD_DRA (MOEA/D with dynamical resource allocation)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOEAD_DRA(MOEAD):
    """
    MOEAD_DRA (MOEA/D with dynamical resource allocation)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Zhang, Q., W. Liu, and H. Li. 2009. The performance of a new version of MOEA/D on CEC09 unconstrained MOP test instances.
    Paper presented at the 2009 IEEE Congress on Evolutionary Computation, 18-21 May 2009.
    """

    def __init__(self,
                 problem,
                 population_size,
                 crossover,
                 mutation,
                 aggregation_function,
                 neighborhood_selection_probability,
                 max_number_of_replaced_solutions,
                 neighbor_size,
                 weight_files_path,
                 population_generator=store.default_generator,
                 population_evaluator=store.default_evaluator,
                 termination_criterion=store.default_termination_criteria,
                 ):
        super(MOEAD_DRA, self).__init__(
            problem,
            population_size,
            crossover,
            mutation,
            aggregation_function,
            neighborhood_selection_probability,
            max_number_of_replaced_solutions,
            neighbor_size,
            weight_files_path,
            termination_criterion=termination_criterion,
            population_generator=population_generator,
            population_evaluator=population_evaluator,
        )
        self.saved_solutions = []
        self.utility = [1.0 for _ in range(population_size)]
        self.frequency = [0.0 for _ in range(population_size)]
        self.generation_counter = 0
        self.order = []
        self.current_order_index = 0

    def init_progress(self):
        super().init_progress()
        self.saved_solutions = [copy.deepcopy(solution) for solution in self.solutions]
        self.evaluations = self.population_size
        for solution in self.solutions:
            self.objective_function.update(solution.objectives)
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        self.order = self.__tour_selection(10)
        self.current_order_index = 0

    def update_progress(self):
        super().update_progress()
        self.current_order_index += 1
        if self.current_order_index == (len(self.order)):
            self.order = self.__tour_selection(10)
            self.current_order_index = 0
        self.generation_counter += 1
        if self.generation_counter % 30 == 0:
            self.__utility_function()

    def selection(self, population: List[S]):
        self.current_solution_id = self.order[self.current_order_index]
        self.current_order_index += 1
        self.frequency[self.current_solution_id] += 1
        rnd = random.random()
        if rnd < self.neighborhood_selection_probability:
            self.neighbor_type = "NEIGHBOR"
        else:
            self.neighbor_type = "POPULATION"
        if self.neighbor_type == "NEIGHBOR":
            neighbors = self.neighborhood.get_neighbors(self.current_solution_id, population)
            selected_solutions = self.selection_operator.execute(neighbors)
        else:
            selected_solutions = self.selection_operator.execute(population)
        selected_solutions.append(population[self.current_solution_id])
        return selected_solutions

    def __utility_function(self):
        for i in range(len(self.solutions)):
            f1 = self.objective_function.compute(self.solutions[i].objectives, self.neighborhood.weight_vectors[i])
            f2 = self.objective_function.compute(self.saved_solutions[i].objectives,
                                                 self.neighborhood.weight_vectors[i])
            delta = f2 - f1
            if delta > 0.001:
                self.utility[i] = 1.0
            else:
                utility_value = (0.95 + (0.05 * delta / 0.001)) * self.utility[i]
                self.utility[i] = utility_value if utility_value < 1.0 else 1.0
            self.saved_solutions[i] = copy.deepcopy(self.solutions[i])

    def __tour_selection(self, depth):
        selected = [i for i in range(self.problem.number_of_objectives)]
        candidate = [i for i in range(self.problem.number_of_objectives, self.population_size)]
        while len(selected) < int(self.population_size / 5.0):
            best_idd = int(random.random() * len(candidate))
            best_sub = candidate[best_idd]
            for i in range(1, depth):
                i2 = int(random.random() * len(candidate))
                s2 = candidate[i2]
                if self.utility[s2] > self.utility[best_sub]:
                    best_idd = i2
                    best_sub = s2
            selected.append(best_sub)
            del candidate[best_idd]
        return selected

    def get_name(self):
        return "MOEAD-DRA"

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


"""
MOEADIEpsilon (Epsilon constraint-handling method in MOEA/D)
Creator: Zixiang Li, zixiangliwust@gmail.com;`n"""


class MOEADIEpsilon(MOEAD):
    """
    MOEADIEpsilon (Epsilon constraint-handling method in MOEA/D)
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
    [2] Fan, Zhun, Wenji Li, Xinye Cai, Han Huang, Yi Fang, Yugen You, Jiajie Mo, Caimin Wei, and Erik Goodman. 2019.
    "An improved epsilon constraint-handling method in MOEA/D for CMOPs with large infeasible regions."
    Soft Computing 23 (23):12491-510. doi: 10.1007/s00500-019-03794-x.
    """

    def __init__(self,
                 problem: Problem,
                 population_size: int,
                 crossover: FloatDifferentialEvolutionCrossover,
                 mutation: Mutation,
                 aggregation_function: AggregationFunction,
                 neighborhood_selection_probability: float,
                 max_number_of_replaced_solutions: int,
                 neighbor_size: int,
                 weight_files_path: str,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = StoppingByEvaluations(300000),
                 ):
        """
        param max_number_of_replaced_solutions: (eta in Zhang & Li paper).
        param neighborhood_selection_probability: Probability of mating with a solution in the neighborhood rather
        than the entire population (Delta in Zhang & Li paper).
        """
        super(MOEADIEpsilon, self).__init__(
            problem=problem,
            population_size=population_size,
            crossover=crossover,
            mutation=mutation,
            aggregation_function=aggregation_function,
            neighborhood_selection_probability=neighborhood_selection_probability,
            max_number_of_replaced_solutions=max_number_of_replaced_solutions,
            neighbor_size=neighbor_size,
            weight_files_path=weight_files_path,
            population_evaluator=population_evaluator,
            population_generator=population_generator,
            termination_criterion=termination_criterion,
        )
        self.constraints = []
        self.epsilon_k = 0
        self.phi_max = -1e30
        self.epsilon_zero = 0
        self.tc = 800
        self.tao = 0.05
        self.rk = 0
        self.generation_counter = 0
        self.archive = []

    def init_progress(self) -> None:
        super().init_progress()
        # for i in range(self.population_size):
        #    self.constraints[i] = get_overall_constraint_violation_degree(self.permutation[i])
        self.constraints = [
            overall_constraint_violation_degree(self.solutions[i]) for i in range(0, self.population_size)
        ]
        sorted(self.constraints)
        self.epsilon_zero = abs(self.constraints[int(ceil(0.05 * self.population_size))])
        if self.phi_max < abs(self.constraints[0]):
            self.phi_max = abs(self.constraints[0])
        self.rk = feasibility_ratio(self.solutions)
        self.epsilon_k = self.epsilon_zero

    def update_progress(self) -> None:
        super().update_progress()
        if self.evaluations % self.population_size == 0:
            self.update_external_archive()
            self.generation_counter += 1
            self.rk = feasibility_ratio(self.solutions)
            if self.generation_counter >= self.tc:
                self.epsilon_k = 0
            else:
                if self.rk < 0.95:
                    self.epsilon_k = (1 - self.tao) * self.epsilon_k
                else:
                    self.epsilon_k = self.phi_max * (1 + self.tao)

    def update_current_subproblem_neighborhood(self, new_solution, population):
        if self.phi_max < overall_constraint_violation_degree(new_solution):
            self.phi_max = overall_constraint_violation_degree(new_solution)
        if self.neighbor_type == "NEIGHBOR":
            neighbors = self.neighborhood.get_neighborhood()[self.current_solution_id]
            permuted_neighbors_indexes = copy.deepcopy(neighbors.tolist())
        else:
            permuted_neighbors_indexes = Permutation(self.population_size).get_permutation()
        replacements = 0
        for i in range(len(permuted_neighbors_indexes)):
            k = permuted_neighbors_indexes[i]
            f1 = self.objective_function.compute(population[k].objectives, self.neighborhood.weight_vectors[k])
            f2 = self.objective_function.compute(new_solution.objectives, self.neighborhood.weight_vectors[k])
            cons1 = abs(overall_constraint_violation_degree(self.solutions[k]))
            cons2 = abs(overall_constraint_violation_degree(new_solution))
            if cons1 < self.epsilon_k and cons2 <= self.epsilon_k:
                if f2 < f1:
                    population[k] = copy.deepcopy(new_solution)
                    replacements += 1
            elif cons1 == cons2:
                if f2 < f1:
                    population[k] = copy.deepcopy(new_solution)
                    replacements += 1
            elif cons2 < cons1:
                population[k] = copy.deepcopy(new_solution)
                replacements += 1
            if replacements >= self.max_number_of_replaced_solutions:
                break
        return population

    def update_external_archive(self):
        feasible_solutions = []
        for solution in self.solutions:
            if is_feasible(solution):
                feasible_solutions.append(copy.deepcopy(solution))
        if len(feasible_solutions) > 0:
            feasible_solutions = feasible_solutions + self.archive
            ranking = FastNonDominatedRanking()
            ranking.compute_ranking(feasible_solutions)
            first_rank_solutions = ranking.get_sub_front(0)
            if len(first_rank_solutions) <= self.population_size:
                self.archive = []
                for solution in first_rank_solutions:
                    self.archive.append(copy.deepcopy(solution))
            else:
                crowding_distance = CrowdingDistance()
                while len(first_rank_solutions) > self.population_size:
                    crowding_distance.compute_density_estimator(first_rank_solutions)
                    first_rank_solutions = sorted(
                        first_rank_solutions, key=lambda x: x.attributes["crowding_distance"], reverse=True
                    )
                    first_rank_solutions.pop()
                self.archive = []
                for solution in first_rank_solutions:
                    self.archive.append(copy.deepcopy(solution))

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
