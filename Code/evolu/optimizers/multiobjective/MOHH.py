import random
import copy
import math
import time
from typing import List, TypeVar, Optional

from evolu.config import store
from evolu.core.algorithm import MultiObjectiveSwarmRoot
from evolu.core.problem import FloatProblem
from evolu.core.quality_indicator import HyperVolume
from evolu.core.solution import FloatSolution
from evolu.operator.mutation import FloatUniformMutation, FloatNonUniformMutation
from evolu.util.archive import BoundedArchive, NonDominatedSolutionsArchive
from evolu.util.comparator import DominanceWithConstraintsComparator, EpsilonDominanceComparator
from evolu.util.density_estimator import ModifiedCrowdingDistance
from evolu.util.evaluator import Evaluator
from evolu.util.generator import Generator
from evolu.util.termination_criterion import TerminationCriterion
from evolu.logger import get_logger

# Import DQN and Q-learning implementations
from evolu.ml.dqn import DQN
from evolu.ml.q_learning import QLearning

logger = get_logger(__name__)
S = TypeVar("S")
R = List[S]


class DQNMOHH(MultiObjectiveSwarmRoot[S, R]):
    """
    DQNMOHH (Deep Q-Network Multi-objective Hyper-Heuristic)
    A multi-objective hyper-heuristic algorithm that uses Deep Q-Network (DQN) for adaptive operator selection.
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
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
                 dqn_state_dim: int = 5,
                 dqn_learning_rate: float = 0.001,
                 target_update_frequency: int = 10
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        param dqn_state_dim: Dimension of the DQN state vector.
        param dqn_learning_rate: Learning rate for DQN.
        param target_update_frequency: How often to update the target network.
        """        
        super(DQNMOHH, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "DQNMOHH"
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.__EPS = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.density_estimator = ModifiedCrowdingDistance()
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
        
        # Define state and action sets for DQN
        # Define actions (operators) - corresponding to the eight operator names
        self.action_set = [
            "PSO_POSITION_UPDATE",    # For PSO_Position_Update
            "PSO_PLUS_UNIFORM_MUTATION",# For PSO_Plus_Uniform_Mutation
            "PSO_PLUS_NON_UNIFORM_MUTATION", # For PSO_Plus_Non_Uniform_Mutation
            "TLBO_TEACHING_PHASE",    # For TLBO_Teaching_Phase
            "TLBO_LEARNING_PHASE",    # For TLBO_Learning_Phase
            "TLBO_MIXED_PHASE",       # For TLBO_Mixed_Phase
            "WOA_OPERATOR",           # For WOA Operator
            "GWO_OPERATOR"            # For GWO Operator
        ]
        # Define state set based on algorithm progress (0.0 to 1.0)
        self.state_set = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        
        # DQN-related attributes
        self.dqn_learning_rate = dqn_learning_rate
        self.target_update_frequency = target_update_frequency
        self.epsilon_start = 1.0  # For epsilon-greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Initialize DQN
        self.dqn_agent = DQN(
            state_dim=len(self.state_set),# Use number of states as state dimension
            action_dim=len(self.action_set),# Use number of actions as action dimension
            learning_rate=self.dqn_learning_rate,
            epsilon=self.epsilon_start,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            memory_size=2000,
            batch_size=32
        )

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
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def pso_position_update(self, population: List[S]) -> List[S]:
        """Apply PSO position update mechanism."""
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
    
    def pso_plus_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply uniform mutation
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop
    
    def pso_plus_non_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by non-uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply non-uniform mutation
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.non_uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop

    def tlbo_teaching_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO teaching phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        # Calculate mean position element-wise
        mean_position = [0.0] * self.problem.number_of_variables
        for i in range(self.problem.number_of_variables):
            sum_var = 0.0
            for item in population:
                sum_var += item.variables[i]
            mean_position[i] = sum_var / self.population_size
        
        for j in range(self.population_size):
            g_best = self.select_global_best()
            # Teaching Factor (TF) is randomly 1 or 2
            TF = random.randint(1, 2)  # 1 or 2 (never 3)
            
            # Update each solution based on the teacher (global best) and mean element-wise
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0.0, 1.0) * (
                        g_best.variables[i] - TF * mean_position[i])
      
        return offsprings

    def tlbo_learning_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO learning phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        for j in range(self.population_size):
            # Randomly select another solution for comparison (different from current)
            idx = random.randint(0, self.population_size - 1)
            while idx == j:
                idx = random.randint(0, self.population_size - 1)
            
            # Compare the two solutions
            dominance_result = self.dominance_comparator.compare(population[j], population[idx])
            
            # Update based on comparison element-wise
            if dominance_result <= 0:  # Current solution is worse or equal
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[j].variables[i] - population[idx].variables[i])
            else:  # Current solution is better
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[idx].variables[i] - population[j].variables[i])
             
        return offsprings

    def tlbo_mixed_phase(self, population: List[S]) -> List[S]:
        """Apply mixed TLBO mechanism combining both teaching and learning phases."""
        # First apply teaching phase
        taught_pop = self.tlbo_teaching_phase(population)
        # Then apply learning phase
        learned_pop = self.tlbo_learning_phase(taught_pop)
        return learned_pop
    
    def woa_operator(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            r1 = random.uniform(0, 1)  # r1 is a random number in [0,1]
            r2 = random.uniform(0, 1)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if random.uniform(0, 1) < 0.5:
                if abs(a) < 1:
                    for i in range(self.problem.number_of_variables):
                        d_leader_i = abs(c * g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = g_best.variables[i] - a * d_leader_i
                elif abs(a) >= 1:
                    idx = random.randint(0, self.population_size - 1)
                    for i in range(self.problem.number_of_variables):
                        d_idx_i = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx_i
            else:
                l = - 2 * random.uniform(0, 1) + 1  # parameters in Eq. (2.5)
                for i in range(self.problem.number_of_variables):
                    d_leader2_i = abs(g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2_i * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + g_best.variables[i]
        return offsprings
    
    def gwo_operator(self, population: List[S]) -> List[S]:
        """Apply Multi-Objective Grey Wolf Optimizer mechanism."""
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) >= 3:
            solutions = random.sample(leaders_archive, 3)
            alpha_solution = copy.deepcopy(solutions[0])
            beta_solution = copy.deepcopy(solutions[1])
            delta_solution = copy.deepcopy(solutions[2])
        elif len(leaders_archive) >= 2:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            beta_solution = copy.deepcopy(leaders_archive[1])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            delta_solution = copy.deepcopy(self.solutions[0])
        else:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            beta_solution = copy.deepcopy(self.solutions[0])
            delta_solution = copy.deepcopy(self.solutions[1])
        
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            a1 = a * (2 * random.uniform(0, 1) - 1)
            a2 = a * (2 * random.uniform(0, 1) - 1)
            a3 = a * (2 * random.uniform(0, 1) - 1)
            c1 = 2 * random.uniform(0, 1)
            c2 = 2 * random.uniform(0, 1)
            c3 = 2 * random.uniform(0, 1)
            for i in range(self.problem.number_of_variables):
                d_alpha_i = abs(c1 * alpha_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 1
                x1_i = alpha_solution.variables[i] - a1 * d_alpha_i  # Equation (3.6)-part 1
                d_beta_i = abs(c2 * beta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 2
                x2_i = beta_solution.variables[i] - a2 * d_beta_i  # Equation (3.6)-part 2
                d_delta_i = abs(c3 * delta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 3
                x3_i = delta_solution.variables[i] - a3 * d_delta_i  # Equation (3.5)-part 3
                offsprings[j].variables[i] = (x1_i + x2_i + x3_i) / 3  # Equation (3.7)
        # Calculate objective function for each search solution
        return offsprings

    def select_global_best(self) -> FloatSolution:
        self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
        sorted_population = self.leaders_archive.solution_list
        self.density_estimator.sort(sorted_population)
        g_best = copy.deepcopy(sorted_population[0])
        sorted_population[0].survive_time += 1
        return g_best
    
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

    def get_state_vector(self):
        """Get the current state vector based on algorithm progress."""
        progress = self.evaluations / self.termination_criterion.max_evaluations
        
        # Create a state vector with normalized progress
        state_vector = [progress]  # Progress in the algorithm
        
        # Add other relevant state information
        archive_size = len(self.leaders_archive.solution_list)
        # Check if the archive has maximum_size attribute (bounded archives) or use a default
        max_archive_size = getattr(self.leaders_archive, 'maximum_size', 100)
        archive_density = archive_size / max_archive_size if max_archive_size > 0 else 0
        state_vector.append(archive_density)
        
        # Add iteration information
        state_vector.append(self.iterations / (self.termination_criterion.max_evaluations / self.population_size))
        
        # Pad with zeros if needed to match expected state dimension
        while len(state_vector) < len(self.state_set):
            state_vector.append(0.0)
        
        # Trim if too long
        state_vector = state_vector[:len(self.state_set)]
        
        return state_vector

    def evolve(self):
        # Select operator using DQN
        state = self.get_state_vector()
        operator_idx = self.dqn_agent.select_action(state)
        
        if operator_idx == 0:  # PSO Position Update
            self.solutions = self.pso_position_update(self.solutions)
        elif operator_idx == 1:  # PSO Plus Uniform Mutation
            self.solutions = self.pso_plus_uniform_mutation_operator(self.solutions)
        elif operator_idx == 2:  # PSO Plus Non-Uniform Mutation
            self.solutions = self.pso_plus_non_uniform_mutation_operator(self.solutions)
        elif operator_idx == 3:  # TLBO Teaching Phase
            self.solutions = self.tlbo_teaching_phase(self.solutions)
        elif operator_idx == 4:  # TLBO Learning Phase
            self.solutions = self.tlbo_learning_phase(self.solutions)
        elif operator_idx == 5:  # TLBO Mixed Phase
            self.solutions = self.tlbo_mixed_phase(self.solutions)
        elif operator_idx == 6:  # WOA Operator
            self.solutions = self.woa_operator(self.solutions)
        elif operator_idx == 7:  # GWO Operator
            self.solutions = self.gwo_operator(self.solutions)
        
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)
        
        # Calculate reward based on improvement in the archive        
        reward_value = 0
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
                reward_value += 1
        
        # If no new solutions were added, assign a small negative reward
        if reward_value == 0:
            reward_value = -1
        
        # Get current state for storing transition
        current_state = self.get_state_vector()
        next_state = self.get_state_vector()
        
        # Determine if episode is done based on termination criterion
        done = False
        if hasattr(self.termination_criterion, 'is_met'):
            if self.termination_criterion.is_met:
                done = True
        
        # Store transition in DQN's memory
        self.dqn_agent.remember(current_state, operator_idx, reward_value, next_state, done)
        
        # Train DQN agent
        self.dqn_agent.replay()
        
        # Update target network periodically
        if self.iterations % self.target_update_frequency == 0:
            self.dqn_agent.update_target_network()

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class MOHHv1(MultiObjectiveSwarmRoot[S, R]):
    """
    MOHHv1: Uses only 3 evolutionary mechanisms including PSO_POSITION_UPDATE, 
    PSO_PLUS_UNIFORM_MUTATION, and PSO_PLUS_NON_UNIFORM_MUTATION.
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
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        """        
        super(MOHHv1, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "MOHHv1"
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.__EPS = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.density_estimator = ModifiedCrowdingDistance()
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
        
        # Define actions (operators) - only 3 operators for MOHHv1
        self.action_set = [
            "PSO_POSITION_UPDATE",    # For PSO_Position_Update
            "PSO_PLUS_UNIFORM_MUTATION",# For PSO_Plus_Uniform_Mutation
            "PSO_PLUS_NON_UNIFORM_MUTATION", # For PSO_Plus_Non_Uniform_Mutation
        ]

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
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def pso_position_update(self, population: List[S]) -> List[S]:
        """Apply PSO position update mechanism."""
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
    
    def pso_plus_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply uniform mutation
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop
    
    def pso_plus_non_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by non-uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply non-uniform mutation
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.non_uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop

    def select_global_best(self) -> FloatSolution:
        self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
        sorted_population = self.leaders_archive.solution_list
        self.density_estimator.sort(sorted_population)
        g_best = copy.deepcopy(sorted_population[0])
        sorted_population[0].survive_time += 1
        return g_best
    
    def update_local_best(self, population: List[S]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(population[i], self.local_best_solutions[i])
            if flag != 1:
                self.local_best_solutions[i] = copy.deepcopy(population[i])

    def evolve(self):
        # Randomly select one of the 3 operators
        selected_action = random.choice(self.action_set)
        
        if selected_action == "PSO_POSITION_UPDATE":
            self.solutions = self.pso_position_update(self.solutions)
        elif selected_action == "PSO_PLUS_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_uniform_mutation_operator(self.solutions)
        elif selected_action == "PSO_PLUS_NON_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_non_uniform_mutation_operator(self.solutions)
        
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)
        
        # Update archive
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class MOHHv2(MultiObjectiveSwarmRoot[S, R]):
    """
    MOHHv2: Without the Q learning mechanism (operator selection is random).
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
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        """        
        super(MOHHv2, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "MOHHv2"
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.__EPS = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.density_estimator = ModifiedCrowdingDistance()
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
        
        # Define actions (operators) - all 8 operators but selected randomly
        self.action_set = [
            "PSO_POSITION_UPDATE",    # For PSO_Position_Update
            "PSO_PLUS_UNIFORM_MUTATION",# For PSO_Plus_Uniform_Mutation
            "PSO_PLUS_NON_UNIFORM_MUTATION", # For PSO_Plus_Non_Uniform_Mutation
            "TLBO_TEACHING_PHASE",    # For TLBO_Teaching_Phase
            "TLBO_LEARNING_PHASE",    # For TLBO_Learning_Phase
            "TLBO_MIXED_PHASE",       # For TLBO_Mixed_Phase
            "WOA_OPERATOR",           # For WOA Operator
            "GWO_OPERATOR"            # For GWO Operator
        ]

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
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def pso_position_update(self, population: List[S]) -> List[S]:
        """Apply PSO position update mechanism."""
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
    
    def pso_plus_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply uniform mutation
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop
    
    def pso_plus_non_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by non-uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply non-uniform mutation
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.non_uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop

    def tlbo_teaching_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO teaching phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        # Calculate mean position element-wise
        mean_position = [0.0] * self.problem.number_of_variables
        for i in range(self.problem.number_of_variables):
            sum_var = 0.0
            for item in population:
                sum_var += item.variables[i]
            mean_position[i] = sum_var / self.population_size
        
        for j in range(self.population_size):
            g_best = self.select_global_best()
            # Teaching Factor (TF) is randomly 1 or 2
            TF = random.randint(1, 2)  # 1 or 2 (never 3)
            
            # Update each solution based on the teacher (global best) and mean element-wise
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0.0, 1.0) * (
                        g_best.variables[i] - TF * mean_position[i])
      
        return offsprings

    def tlbo_learning_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO learning phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        for j in range(self.population_size):
            # Randomly select another solution for comparison (different from current)
            idx = random.randint(0, self.population_size - 1)
            while idx == j:
                idx = random.randint(0, self.population_size - 1)
            
            # Compare the two solutions
            dominance_result = self.dominance_comparator.compare(population[j], population[idx])
            
            # Update based on comparison element-wise
            if dominance_result <= 0:  # Current solution is worse or equal
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[j].variables[i] - population[idx].variables[i])
            else:  # Current solution is better
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[idx].variables[i] - population[j].variables[i])
             
        return offsprings

    def tlbo_mixed_phase(self, population: List[S]) -> List[S]:
        """Apply mixed TLBO mechanism combining both teaching and learning phases."""
        # First apply teaching phase
        taught_pop = self.tlbo_teaching_phase(population)
        # Then apply learning phase
        learned_pop = self.tlbo_learning_phase(taught_pop)
        return learned_pop
    
    def woa_operator(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            r1 = random.uniform(0, 1)  # r1 is a random number in [0,1]
            r2 = random.uniform(0, 1)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if random.uniform(0, 1) < 0.5:
                if abs(a) < 1:
                    for i in range(self.problem.number_of_variables):
                        d_leader_i = abs(c * g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = g_best.variables[i] - a * d_leader_i
                elif abs(a) >= 1:
                    idx = random.randint(0, self.population_size - 1)
                    for i in range(self.problem.number_of_variables):
                        d_idx_i = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx_i
            else:
                l = - 2 * random.uniform(0, 1) + 1  # parameters in Eq. (2.5)
                for i in range(self.problem.number_of_variables):
                    d_leader2_i = abs(g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2_i * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + g_best.variables[i]
        return offsprings
    
    def gwo_operator(self, population: List[S]) -> List[S]:
        """Apply Multi-Objective Grey Wolf Optimizer mechanism."""
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) >= 3:
            solutions = random.sample(leaders_archive, 3)
            alpha_solution = copy.deepcopy(solutions[0])
            beta_solution = copy.deepcopy(solutions[1])
            delta_solution = copy.deepcopy(solutions[2])
        elif len(leaders_archive) >= 2:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            beta_solution = copy.deepcopy(leaders_archive[1])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            delta_solution = copy.deepcopy(self.solutions[0])
        else:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            beta_solution = copy.deepcopy(self.solutions[0])
            delta_solution = copy.deepcopy(self.solutions[1])
        
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            a1 = a * (2 * random.uniform(0, 1) - 1)
            a2 = a * (2 * random.uniform(0, 1) - 1)
            a3 = a * (2 * random.uniform(0, 1) - 1)
            c1 = 2 * random.uniform(0, 1)
            c2 = 2 * random.uniform(0, 1)
            c3 = 2 * random.uniform(0, 1)
            for i in range(self.problem.number_of_variables):
                d_alpha_i = abs(c1 * alpha_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 1
                x1_i = alpha_solution.variables[i] - a1 * d_alpha_i  # Equation (3.6)-part 1
                d_beta_i = abs(c2 * beta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 2
                x2_i = beta_solution.variables[i] - a2 * d_beta_i  # Equation (3.6)-part 2
                d_delta_i = abs(c3 * delta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 3
                x3_i = delta_solution.variables[i] - a3 * d_delta_i  # Equation (3.5)-part 3
                offsprings[j].variables[i] = (x1_i + x2_i + x3_i) / 3  # Equation (3.7)
        # Calculate objective function for each search solution
        return offsprings

    def select_global_best(self) -> FloatSolution:
        self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
        sorted_population = self.leaders_archive.solution_list
        self.density_estimator.sort(sorted_population)
        g_best = copy.deepcopy(sorted_population[0])
        sorted_population[0].survive_time += 1
        return g_best
    
    def update_local_best(self, population: List[S]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(population[i], self.local_best_solutions[i])
            if flag != 1:
                self.local_best_solutions[i] = copy.deepcopy(population[i])

    def evolve(self):
        # Randomly select one of the 8 operators (without Q-learning)
        selected_action = random.choice(self.action_set)
        
        if selected_action == "PSO_POSITION_UPDATE":
            self.solutions = self.pso_position_update(self.solutions)
        elif selected_action == "PSO_PLUS_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_uniform_mutation_operator(self.solutions)
        elif selected_action == "PSO_PLUS_NON_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_non_uniform_mutation_operator(self.solutions)
        elif selected_action == "TLBO_TEACHING_PHASE":
            self.solutions = self.tlbo_teaching_phase(self.solutions)
        elif selected_action == "TLBO_LEARNING_PHASE":
            self.solutions = self.tlbo_learning_phase(self.solutions)
        elif selected_action == "TLBO_MIXED_PHASE":
            self.solutions = self.tlbo_mixed_phase(self.solutions)
        elif selected_action == "WOA_OPERATOR":
            self.solutions = self.woa_operator(self.solutions)
        elif selected_action == "GWO_OPERATOR":
            self.solutions = self.gwo_operator(self.solutions)
        
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)
        
        # Update archive
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class MOHHv3(MultiObjectiveSwarmRoot[S, R]):
    """
    MOHHv3: Uses DQN for parameter selection.
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
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        """        
        super(MOHHv3, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "MOHHv3"
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.__EPS = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.density_estimator = ModifiedCrowdingDistance()
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
        
        # Define state and action sets for DQN
        # Define actions (operators) - all 8 operators
        self.action_set = [
            "PSO_POSITION_UPDATE",    # For PSO_Position_Update
            "PSO_PLUS_UNIFORM_MUTATION",# For PSO_Plus_Uniform_Mutation
            "PSO_PLUS_NON_UNIFORM_MUTATION", # For PSO_Plus_Non_Uniform_Mutation
            "TLBO_TEACHING_PHASE",    # For TLBO_Teaching_Phase
            "TLBO_LEARNING_PHASE",    # For TLBO_Learning_Phase
            "TLBO_MIXED_PHASE",       # For TLBO_Mixed_Phase
            "WOA_OPERATOR",           # For WOA Operator
            "GWO_OPERATOR"            # For GWO Operator
        ]
        # Define state set based on algorithm progress (0.0 to 1.0)
        self.state_set = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        
        # DQN-related attributes
        self.dqn_learning_rate = 0.001
        self.target_update_frequency = 10
        self.epsilon_start = 1.0  # For epsilon-greedy
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95  # Discount factor
        
        # Initialize DQN
        self.dqn_agent = DQN(
            state_dim=len(self.state_set),# Use number of states as state dimension
            action_dim=len(self.action_set),# Use number of actions as action dimension
            learning_rate=self.dqn_learning_rate,
            epsilon=self.epsilon_start,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            memory_size=2000,
            batch_size=32
        )

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
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def pso_position_update(self, population: List[S]) -> List[S]:
        """Apply PSO position update mechanism."""
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
    
    def pso_plus_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply uniform mutation
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop
    
    def pso_plus_non_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by non-uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply non-uniform mutation
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.non_uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop

    def tlbo_teaching_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO teaching phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        # Calculate mean position element-wise
        mean_position = [0.0] * self.problem.number_of_variables
        for i in range(self.problem.number_of_variables):
            sum_var = 0.0
            for item in population:
                sum_var += item.variables[i]
            mean_position[i] = sum_var / self.population_size
        
        for j in range(self.population_size):
            g_best = self.select_global_best()
            # Teaching Factor (TF) is randomly 1 or 2
            TF = random.randint(1, 2)  # 1 or 2 (never 3)
            
            # Update each solution based on the teacher (global best) and mean element-wise
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0.0, 1.0) * (
                        g_best.variables[i] - TF * mean_position[i])
      
        return offsprings

    def tlbo_learning_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO learning phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        for j in range(self.population_size):
            # Randomly select another solution for comparison (different from current)
            idx = random.randint(0, self.population_size - 1)
            while idx == j:
                idx = random.randint(0, self.population_size - 1)
            
            # Compare the two solutions
            dominance_result = self.dominance_comparator.compare(population[j], population[idx])
            
            # Update based on comparison element-wise
            if dominance_result <= 0:  # Current solution is worse or equal
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[j].variables[i] - population[idx].variables[i])
            else:  # Current solution is better
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[idx].variables[i] - population[j].variables[i])
             
        return offsprings

    def tlbo_mixed_phase(self, population: List[S]) -> List[S]:
        """Apply mixed TLBO mechanism combining both teaching and learning phases."""
        # First apply teaching phase
        taught_pop = self.tlbo_teaching_phase(population)
        # Then apply learning phase
        learned_pop = self.tlbo_learning_phase(taught_pop)
        return learned_pop
    
    def woa_operator(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            r1 = random.uniform(0, 1)  # r1 is a random number in [0,1]
            r2 = random.uniform(0, 1)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if random.uniform(0, 1) < 0.5:
                if abs(a) < 1:
                    for i in range(self.problem.number_of_variables):
                        d_leader_i = abs(c * g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = g_best.variables[i] - a * d_leader_i
                elif abs(a) >= 1:
                    idx = random.randint(0, self.population_size - 1)
                    for i in range(self.problem.number_of_variables):
                        d_idx_i = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx_i
            else:
                l = - 2 * random.uniform(0, 1) + 1  # parameters in Eq. (2.5)
                for i in range(self.problem.number_of_variables):
                    d_leader2_i = abs(g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2_i * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + g_best.variables[i]
        return offsprings
    
    def gwo_operator(self, population: List[S]) -> List[S]:
        """Apply Multi-Objective Grey Wolf Optimizer mechanism."""
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) >= 3:
            solutions = random.sample(leaders_archive, 3)
            alpha_solution = copy.deepcopy(solutions[0])
            beta_solution = copy.deepcopy(solutions[1])
            delta_solution = copy.deepcopy(solutions[2])
        elif len(leaders_archive) >= 2:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            beta_solution = copy.deepcopy(leaders_archive[1])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            delta_solution = copy.deepcopy(self.solutions[0])
        else:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            beta_solution = copy.deepcopy(self.solutions[0])
            delta_solution = copy.deepcopy(self.solutions[1])
        
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            a1 = a * (2 * random.uniform(0, 1) - 1)
            a2 = a * (2 * random.uniform(0, 1) - 1)
            a3 = a * (2 * random.uniform(0, 1) - 1)
            c1 = 2 * random.uniform(0, 1)
            c2 = 2 * random.uniform(0, 1)
            c3 = 2 * random.uniform(0, 1)
            for i in range(self.problem.number_of_variables):
                d_alpha_i = abs(c1 * alpha_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 1
                x1_i = alpha_solution.variables[i] - a1 * d_alpha_i  # Equation (3.6)-part 1
                d_beta_i = abs(c2 * beta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 2
                x2_i = beta_solution.variables[i] - a2 * d_beta_i  # Equation (3.6)-part 2
                d_delta_i = abs(c3 * delta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 3
                x3_i = delta_solution.variables[i] - a3 * d_delta_i  # Equation (3.5)-part 3
                offsprings[j].variables[i] = (x1_i + x2_i + x3_i) / 3  # Equation (3.7)
        # Calculate objective function for each search solution
        return offsprings

    def select_global_best(self) -> FloatSolution:
        self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
        sorted_population = self.leaders_archive.solution_list
        self.density_estimator.sort(sorted_population)
        g_best = copy.deepcopy(sorted_population[0])
        sorted_population[0].survive_time += 1
        return g_best
    
    def update_local_best(self, population: List[S]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(population[i], self.local_best_solutions[i])
            if flag != 1:
                self.local_best_solutions[i] = copy.deepcopy(population[i])

    def get_state_vector(self):
        """Get the current state vector based on algorithm progress."""
        progress = self.evaluations / self.termination_criterion.max_evaluations
        
        # Create a state vector with normalized progress
        state_vector = [progress]  # Progress in the algorithm
        
        # Add other relevant state information
        archive_size = len(self.leaders_archive.solution_list)
        # Check if the archive has maximum_size attribute (bounded archives) or use a default
        max_archive_size = getattr(self.leaders_archive, 'maximum_size', 100)
        archive_density = archive_size / max_archive_size if max_archive_size > 0 else 0
        state_vector.append(archive_density)
        
        # Add iteration information
        state_vector.append(self.iterations / (self.termination_criterion.max_evaluations / self.population_size))
        
        # Pad with zeros if needed to match expected state dimension
        while len(state_vector) < len(self.state_set):
            state_vector.append(0.0)
        
        # Trim if too long
        state_vector = state_vector[:len(self.state_set)]
        
        return state_vector

    def evolve(self):
        # Select operator using DQN
        state = self.get_state_vector()
        operator_idx = self.dqn_agent.select_action(state)
        
        if operator_idx == 0:  # PSO Position Update
            self.solutions = self.pso_position_update(self.solutions)
        elif operator_idx == 1:  # PSO Plus Uniform Mutation
            self.solutions = self.pso_plus_uniform_mutation_operator(self.solutions)
        elif operator_idx == 2:  # PSO Plus Non-Uniform Mutation
            self.solutions = self.pso_plus_non_uniform_mutation_operator(self.solutions)
        elif operator_idx == 3:  # TLBO Teaching Phase
            self.solutions = self.tlbo_teaching_phase(self.solutions)
        elif operator_idx == 4:  # TLBO Learning Phase
            self.solutions = self.tlbo_learning_phase(self.solutions)
        elif operator_idx == 5:  # TLBO Mixed Phase
            self.solutions = self.tlbo_mixed_phase(self.solutions)
        elif operator_idx == 6:  # WOA Operator
            self.solutions = self.woa_operator(self.solutions)
        elif operator_idx == 7:  # GWO Operator
            self.solutions = self.gwo_operator(self.solutions)
        
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)
        
        # Calculate reward based on improvement in the archive        
        reward_value = 0
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
                reward_value += 1
        
        # If no new solutions were added, assign a small negative reward
        if reward_value == 0:
            reward_value = -1
        
        # Get current state for storing transition
        current_state = self.get_state_vector()
        next_state = self.get_state_vector()
        
        # Determine if episode is done based on termination criterion
        done = False
        if hasattr(self.termination_criterion, 'is_met'):
            if self.termination_criterion.is_met:
                done = True
        
        # Store transition in DQN's memory
        self.dqn_agent.remember(current_state, operator_idx, reward_value, next_state, done)
        
        # Train DQN agent
        self.dqn_agent.replay()
        
        # Update target network periodically
        if self.iterations % self.target_update_frequency == 0:
            self.dqn_agent.update_target_network()

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
    
    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class MOHHv4(MultiObjectiveSwarmRoot[S, R]):
    """
    MOHHv4: Uses MOPSO's global best selection strategy and archive maintenance mechanism.
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
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        """        
        super(MOHHv4, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "MOHHv4"
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
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
        
        # Define actions (operators) - all 8 operators
        self.action_set = [
            "PSO_POSITION_UPDATE",    # For PSO_Position_Update
            "PSO_PLUS_UNIFORM_MUTATION",# For PSO_Plus_Uniform_Mutation
            "PSO_PLUS_NON_UNIFORM_MUTATION", # For PSO_Plus_Non_Uniform_Mutation
            "TLBO_TEACHING_PHASE",    # For TLBO_Teaching_Phase
            "TLBO_LEARNING_PHASE",    # For TLBO_Learning_Phase
            "TLBO_MIXED_PHASE",       # For TLBO_Mixed_Phase
            "WOA_OPERATOR",           # For WOA Operator
            "GWO_OPERATOR"            # For GWO Operator
        ]

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

    def pso_position_update(self, population: List[S]) -> List[S]:
        """Apply PSO position update mechanism."""
        offsprings = copy.deepcopy(population)
        for j in range(self.population_size):
            # Use MOPSO-style global best selection
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
    
    def pso_plus_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply uniform mutation
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop
    
    def pso_plus_non_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by non-uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply non-uniform mutation
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.non_uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop

    def tlbo_teaching_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO teaching phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        # Calculate mean position element-wise
        mean_position = [0.0] * self.problem.number_of_variables
        for i in range(self.problem.number_of_variables):
            sum_var = 0.0
            for item in population:
                sum_var += item.variables[i]
            mean_position[i] = sum_var / self.population_size
        
        for j in range(self.population_size):
            # Use MOPSO-style global best selection
            g_best = self.select_global_best()
            # Teaching Factor (TF) is randomly 1 or 2
            TF = random.randint(1, 2)  # 1 or 2 (never 3)
            
            # Update each solution based on the teacher (global best) and mean element-wise
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0.0, 1.0) * (
                        g_best.variables[i] - TF * mean_position[i])
      
        return offsprings

    def tlbo_learning_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO learning phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        for j in range(self.population_size):
            # Randomly select another solution for comparison (different from current)
            idx = random.randint(0, self.population_size - 1)
            while idx == j:
                idx = random.randint(0, self.population_size - 1)
            
            # Compare the two solutions
            dominance_result = self.dominance_comparator.compare(population[j], population[idx])
            
            # Update based on comparison element-wise
            if dominance_result <= 0:  # Current solution is worse or equal
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[j].variables[i] - population[idx].variables[i])
            else:  # Current solution is better
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[idx].variables[i] - population[j].variables[i])
             
        return offsprings

    def tlbo_mixed_phase(self, population: List[S]) -> List[S]:
        """Apply mixed TLBO mechanism combining both teaching and learning phases."""
        # First apply teaching phase
        taught_pop = self.tlbo_teaching_phase(population)
        # Then apply learning phase
        learned_pop = self.tlbo_learning_phase(taught_pop)
        return learned_pop
    
    def woa_operator(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            # Use MOPSO-style global best selection
            g_best = self.select_global_best()
            r1 = random.uniform(0, 1)  # r1 is a random number in [0,1]
            r2 = random.uniform(0, 1)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if random.uniform(0, 1) < 0.5:
                if abs(a) < 1:
                    for i in range(self.problem.number_of_variables):
                        d_leader_i = abs(c * g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = g_best.variables[i] - a * d_leader_i
                elif abs(a) >= 1:
                    idx = random.randint(0, self.population_size - 1)
                    for i in range(self.problem.number_of_variables):
                        d_idx_i = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx_i
            else:
                l = - 2 * random.uniform(0, 1) + 1  # parameters in Eq. (2.5)
                for i in range(self.problem.number_of_variables):
                    d_leader2_i = abs(g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2_i * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + g_best.variables[i]
        return offsprings
    
    def gwo_operator(self, population: List[S]) -> List[S]:
        """Apply Multi-Objective Grey Wolf Optimizer mechanism."""
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) >= 3:
            solutions = random.sample(leaders_archive, 3)
            alpha_solution = copy.deepcopy(solutions[0])
            beta_solution = copy.deepcopy(solutions[1])
            delta_solution = copy.deepcopy(solutions[2])
        elif len(leaders_archive) >= 2:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            beta_solution = copy.deepcopy(leaders_archive[1])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            delta_solution = copy.deepcopy(self.solutions[0])
        else:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            beta_solution = copy.deepcopy(self.solutions[0])
            delta_solution = copy.deepcopy(self.solutions[1])
        
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            a1 = a * (2 * random.uniform(0, 1) - 1)
            a2 = a * (2 * random.uniform(0, 1) - 1)
            a3 = a * (2 * random.uniform(0, 1) - 1)
            c1 = 2 * random.uniform(0, 1)
            c2 = 2 * random.uniform(0, 1)
            c3 = 2 * random.uniform(0, 1)
            for i in range(self.problem.number_of_variables):
                d_alpha_i = abs(c1 * alpha_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 1
                x1_i = alpha_solution.variables[i] - a1 * d_alpha_i  # Equation (3.6)-part 1
                d_beta_i = abs(c2 * beta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 2
                x2_i = beta_solution.variables[i] - a2 * d_beta_i  # Equation (3.6)-part 2
                d_delta_i = abs(c3 * delta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 3
                x3_i = delta_solution.variables[i] - a3 * d_delta_i  # Equation (3.5)-part 3
                offsprings[j].variables[i] = (x1_i + x2_i + x3_i) / 3  # Equation (3.7)
        # Calculate objective function for each search solution
        return offsprings

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
    
    def update_local_best(self, population: List[S]) -> None:
        for i in range(self.population_size):
            flag = self.dominance_comparator.compare(population[i], self.local_best_solutions[i])
            if flag != 1:
                self.local_best_solutions[i] = copy.deepcopy(population[i])

    def evolve(self):
        # Randomly select one of the 8 operators
        selected_action = random.choice(self.action_set)
        
        if selected_action == "PSO_POSITION_UPDATE":
            self.solutions = self.pso_position_update(self.solutions)
        elif selected_action == "PSO_PLUS_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_uniform_mutation_operator(self.solutions)
        elif selected_action == "PSO_PLUS_NON_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_non_uniform_mutation_operator(self.solutions)
        elif selected_action == "TLBO_TEACHING_PHASE":
            self.solutions = self.tlbo_teaching_phase(self.solutions)
        elif selected_action == "TLBO_LEARNING_PHASE":
            self.solutions = self.tlbo_learning_phase(self.solutions)
        elif selected_action == "TLBO_MIXED_PHASE":
            self.solutions = self.tlbo_mixed_phase(self.solutions)
        elif selected_action == "WOA_OPERATOR":
            self.solutions = self.woa_operator(self.solutions)
        elif selected_action == "GWO_OPERATOR":
            self.solutions = self.gwo_operator(self.solutions)
        
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.population_size
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


class QMOHH(MultiObjectiveSwarmRoot[S, R]):
    """
    QMOHH (Q-Learning Multi-objective Hyper-Heuristic)
    A multi-objective hyper-heuristic algorithm that uses Q-Learning for adaptive operator selection.
    References:
    [1] Initial code built based on https://github.com/jMetal/jMetalPy; Antonio Benítez-Hidalgo <antonio.b@uma.es>, Julian Blank <blankjul@egr.msu.edu>
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
                 termination_criterion: TerminationCriterion = store.default_termination_criteria
                 ):
        """
        param problem: The problem to solve.
        param population_size: Size of the population.
        param leaders_archive: Archive for leaders.
        param q_learning_state_dim: Dimension of the Q-learning state vector.
        """        
        super(QMOHH, self).__init__(problem=problem, population_size=population_size)
        self.algorithm_name = "QMOHH"
        self.uniform_mutation = uniform_mutation
        self.non_uniform_mutation = non_uniform_mutation
        self.leaders_archive = leaders_archive
        self.population_generator = population_generator
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)        
        self.__EPS = epsilon
        self.epsilon_archive = NonDominatedSolutionsArchive(EpsilonDominanceComparator(epsilon))
        self.dominance_comparator = DominanceWithConstraintsComparator()
        self.density_estimator = ModifiedCrowdingDistance()
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
        
        # Define state and action sets for Q-learning
        # Define actions (operators) - corresponding to the eight operator names
        self.action_set = [
            "PSO_POSITION_UPDATE",    # For PSO_Position_Update
            "PSO_PLUS_UNIFORM_MUTATION",# For PSO_Plus_Uniform_Mutation
            "PSO_PLUS_NON_UNIFORM_MUTATION", # For PSO_Plus_Non_Uniform_Mutation
            "TLBO_TEACHING_PHASE",    # For TLBO_Teaching_Phase
            "TLBO_LEARNING_PHASE",    # For TLBO_Learning_Phase
            "TLBO_MIXED_PHASE",       # For TLBO_Mixed_Phase
            "WOA_OPERATOR",           # For WOA Operator
            "GWO_OPERATOR"            # For MOGWO Operator
        ]
        # Define state set based on algorithm progress (0.0 to 1.0)
        self.state_set = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9
        ]
        
        # Initialize Q-learning agent
        self.q_learning_agent = QLearning(
            action_set=self.action_set,
            state_set=self.state_set
        )

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
        self.local_best_solutions = copy.deepcopy(self.solutions)

    def pso_position_update(self, population: List[S]) -> List[S]:
        """Apply PSO position update mechanism."""
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
    
    def pso_plus_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply uniform mutation
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop
    
    def pso_plus_non_uniform_mutation_operator(self, population: List[S]) -> List[S]:
        """Apply PSO update followed by non-uniform mutation operator."""
        # First apply PSO position update
        pso_updated = self.pso_position_update(population)
        # Then apply non-uniform mutation
        self.non_uniform_mutation.set_current_iteration(self.evaluations / self.population_size)
        mutated_pop = []
        for solution in pso_updated:
            mutated_solution = self.non_uniform_mutation.execute(solution)
            mutated_pop.append(mutated_solution)
        return mutated_pop

    def tlbo_teaching_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO teaching phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        # Calculate mean position element-wise
        mean_position = [0.0] * self.problem.number_of_variables
        for i in range(self.problem.number_of_variables):
            sum_var = 0.0
            for item in population:
                sum_var += item.variables[i]
            mean_position[i] = sum_var / self.population_size
        
        for j in range(self.population_size):
            g_best = self.select_global_best()
            # Teaching Factor (TF) is randomly 1 or 2
            TF = random.randint(1, 2)  # 1 or 2 (never 3)
            
            # Update each solution based on the teacher (global best) and mean element-wise
            for i in range(self.problem.number_of_variables):
                offsprings[j].variables[i] = population[j].variables[i] + random.uniform(0.0, 1.0) * (
                        g_best.variables[i] - TF * mean_position[i])
      
        return offsprings

    def tlbo_learning_phase(self, population: List[S]) -> List[S]:
        """Apply TLBO learning phase mechanism based on MOTLBO implementation."""
        offsprings = copy.deepcopy(population)
        
        for j in range(self.population_size):
            # Randomly select another solution for comparison (different from current)
            idx = random.randint(0, self.population_size - 1)
            while idx == j:
                idx = random.randint(0, self.population_size - 1)
            
            # Compare the two solutions
            dominance_result = self.dominance_comparator.compare(population[j], population[idx])
            
            # Update based on comparison element-wise
            if dominance_result <= 0:  # Current solution is worse or equal
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[j].variables[i] - population[idx].variables[i])
            else:  # Current solution is better
                for i in range(self.problem.number_of_variables):
                    offsprings[j].variables[i] += random.uniform(0.0, 1.0) * (
                            population[idx].variables[i] - population[j].variables[i])
             
        return offsprings

    def tlbo_mixed_phase(self, population: List[S]) -> List[S]:
        """Apply mixed TLBO mechanism combining both teaching and learning phases."""
        # First apply teaching phase
        taught_pop = self.tlbo_teaching_phase(population)
        # Then apply learning phase
        learned_pop = self.tlbo_learning_phase(taught_pop)
        return learned_pop
    
    def woa_operator(self, population: List[S]) -> List[S]:
        offsprings = copy.deepcopy(population)
        alpha = 2 - self.iterations * (2 / self.max_iterations)
        # Update the Position of search agents
        for j in range(0, self.population_size):
            g_best = self.select_global_best()
            r1 = random.uniform(0, 1)  # r1 is a random number in [0,1]
            r2 = random.uniform(0, 1)  # r2 is a random number in [0,1]
            a = 2 * alpha * r1 - alpha  # Eq. (2.3) in the paper
            c = 2 * r2  # Eq. (2.4) in the paper
            b = 1  # parameters in Eq. (2.5)
            if random.uniform(0, 1) < 0.5:
                if abs(a) < 1:
                    for i in range(self.problem.number_of_variables):
                        d_leader_i = abs(c * g_best.variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = g_best.variables[i] - a * d_leader_i
                elif abs(a) >= 1:
                    idx = random.randint(0, self.population_size - 1)
                    for i in range(self.problem.number_of_variables):
                        d_idx_i = abs(c * population[idx].variables[i] - population[j].variables[i])
                        offsprings[j].variables[i] = population[idx].variables[i] - a * d_idx_i
            else:
                l = - 2 * random.uniform(0, 1) + 1  # parameters in Eq. (2.5)
                for i in range(self.problem.number_of_variables):
                    d_leader2_i = abs(g_best.variables[i] - population[j].variables[i])  # Eq.(2.5)
                    offsprings[j].variables[i] = d_leader2_i * math.exp(b * l) * math.cos(
                        l * 2 * math.pi) + g_best.variables[i]
        return offsprings
    
    def gwo_operator(self, population: List[S]) -> List[S]:
        """Apply Multi-Objective Grey Wolf Optimizer mechanism."""
        leaders_archive = self.leaders_archive.solution_list
        if len(leaders_archive) >= 3:
            solutions = random.sample(leaders_archive, 3)
            alpha_solution = copy.deepcopy(solutions[0])
            beta_solution = copy.deepcopy(solutions[1])
            delta_solution = copy.deepcopy(solutions[2])
        elif len(leaders_archive) >= 2:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            beta_solution = copy.deepcopy(leaders_archive[1])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            delta_solution = copy.deepcopy(self.solutions[0])
        else:
            alpha_solution = copy.deepcopy(leaders_archive[0])
            self.solutions = RankingAndDensityEstimatorSelection(self.population_size).execute(self.solutions)
            self.solutions = self.sort_population.execute(self.solutions)
            beta_solution = copy.deepcopy(self.solutions[0])
            delta_solution = copy.deepcopy(self.solutions[1])
        
        a = 2 - self.iterations * (2 / self.max_iterations)  # a decreases linearly from 2 to 0
        # Update the Position of search agents including omegas
        offsprings = copy.deepcopy(population)
        for j in range(0, self.population_size):
            a1 = a * (2 * random.uniform(0, 1) - 1)
            a2 = a * (2 * random.uniform(0, 1) - 1)
            a3 = a * (2 * random.uniform(0, 1) - 1)
            c1 = 2 * random.uniform(0, 1)
            c2 = 2 * random.uniform(0, 1)
            c3 = 2 * random.uniform(0, 1)
            for i in range(self.problem.number_of_variables):
                d_alpha_i = abs(c1 * alpha_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 1
                x1_i = alpha_solution.variables[i] - a1 * d_alpha_i  # Equation (3.6)-part 1
                d_beta_i = abs(c2 * beta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 2
                x2_i = beta_solution.variables[i] - a2 * d_beta_i  # Equation (3.6)-part 2
                d_delta_i = abs(c3 * delta_solution.variables[i] - population[j].variables[i])  # Equation (3.5)-part 3
                x3_i = delta_solution.variables[i] - a3 * d_delta_i  # Equation (3.5)-part 3
                offsprings[j].variables[i] = (x1_i + x2_i + x3_i) / 3  # Equation (3.7)
        # Calculate objective function for each search solution
        return offsprings

    def select_global_best(self) -> FloatSolution:
        self.density_estimator.compute_density_estimator(self.leaders_archive.solution_list)
        sorted_population = self.leaders_archive.solution_list
        self.density_estimator.sort(sorted_population)
        g_best = copy.deepcopy(sorted_population[0])
        sorted_population[0].survive_time += 1
        return g_best

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

    def evolve(self):
        # Determine current state based on evaluation progress
        # Since evaluations will be incremented after this evolve call in update_progress, 
        # we use current evals to determine the state for this iteration
        current_progress = self.evaluations / self.termination_criterion.max_evaluations
        current_state_index = min(int(current_progress * 10), len(self.state_set) - 1)
        
        # Select action using Q-learning
        selected_action_index = self.q_learning_agent.select_action_and_return_index(current_state_index)
        selected_action = self.action_set[selected_action_index]
        
        if selected_action == "PSO_POSITION_UPDATE":
            self.solutions = self.pso_position_update(self.solutions)
        elif selected_action == "PSO_PLUS_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_uniform_mutation_operator(self.solutions)
        elif selected_action == "PSO_PLUS_NON_UNIFORM_MUTATION":
            self.solutions = self.pso_plus_non_uniform_mutation_operator(self.solutions)
        elif selected_action == "TLBO_TEACHING_PHASE":
            self.solutions = self.tlbo_teaching_phase(self.solutions)
        elif selected_action == "TLBO_LEARNING_PHASE":
            self.solutions = self.tlbo_learning_phase(self.solutions)
        elif selected_action == "TLBO_MIXED_PHASE":
            self.solutions = self.tlbo_mixed_phase(self.solutions)
        elif selected_action == "WOA_OPERATOR":
            self.solutions = self.woa_operator(self.solutions)
        elif selected_action == "GWO_OPERATOR":
            self.solutions = self.gwo_operator(self.solutions)
        
        self.solutions = self.evaluate(self.solutions)
        self.update_local_best(self.solutions)

        # Calculate reward based on improvement in the archive        
        reward_value = 0
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
                reward_value += 1
        
        # If no new solutions were added, assign a small negative reward
        if reward_value == 0:
            reward_value = -1

        # Determine next state based on evaluation progress
        # Calculate the progress after this iteration (evaluations will be increased in update_progress)
        next_progress = (self.evaluations + self.population_size) / self.termination_criterion.max_evaluations
        next_state_index = min(int(next_progress * 10), len(self.state_set) - 1)
        
        # Update Q-learning table
        self.q_learning_agent.update_q_table(selected_action_index, current_state_index, next_state_index, reward_value)

    def update_progress(self) -> None:
        self.evaluations += self.population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.epsilon_archive.solution_list
        self.observable.notify_all(**observable_data)
        for solution in self.solutions:
            if self.leaders_archive.add(solution):
                self.epsilon_archive.add(solution)
    

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
    