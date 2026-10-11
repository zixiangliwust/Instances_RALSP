# -*- coding: utf-8 -*-
import copy
import time
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

from evolu.config import store
from evolu.core.problem import Problem
from evolu.logger import get_logger
from evolu.operator.selection import BestSolutionSelection, WorstSolutionSelection
from evolu.util.comparator import ObjectiveComparator, IdenticalSolutionsComparator
from evolu.operator.replacement import GreedyPopulationReplacement
from evolu.util.restart import SimpleReplaceDuplicatedSolution
from evolu.util.sort_population import SortPopulation

logger = get_logger(__name__)
S = TypeVar("S")
R = TypeVar("R")
"""
module:: algorithm
synopsis: Templates for algorithms.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Algorithm(Generic[S, R], ABC):
    """Base class for metaheuristic algorithms.
    
    This class defines the common interface and lifecycle for all optimization algorithms.
    Subclasses must implement the abstract methods to define specific optimization logic.
    
    Attributes:
        problem (Problem): The problem instance to be solved.
        algorithm_name (str): Name of the algorithm.
        solutions (List[S]): Current population or list of solutions.
        evaluations (int): Number of objective function evaluations performed.
        iterations (int): Number of iterations performed.
        start_computing_time (float): Timestamp when algorithm started computing.
        total_computing_time (float): Total computing time in seconds.
        population_evaluator (Evaluator): Population evaluator for solution evaluation.
        observable (Observable): Observable object for notifying observers.
    """

    def __init__(self, problem: Problem[S]) -> None:
        """Initialize the algorithm with a problem.
        
        Args:
            problem (Problem[S]): The problem instance to solve.
        """
        super().__init__()
        self.problem = problem
        self.algorithm_name = "Algorithm"
        self.solutions: List[S] = []
        self.evaluations = 0
        self.iterations = 0
        self.start_computing_time = 0
        self.total_computing_time = 0
        self.population_evaluator = store.default_evaluator
        self.observable = store.default_observable

    @abstractmethod
    def create_solution(self) -> S:
        """Create a new solution instance.
        
        Returns:
            S: A newly created solution instance.
        
        Note:
            Subclasses must implement this method to create solution instances
            of the appropriate type for their algorithm.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `create_solution`.")

    @abstractmethod
    def evaluate_solution(self, solution: S) -> S:
        """Evaluate the given solution.
        
        Args:
            solution (S): The solution to evaluate.
        
        Returns:
            S: The evaluated solution with objective values set.
        
        Note:
            This method typically delegates to problem.evaluate_solution().
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `evaluate_solution`.")

    def create_initial_solutions(self, population_size: Optional[int] = None) -> List[S]:
        """Create initial population of solutions.
        
        Args:
            population_size (Optional[int]): Size of the initial population.
                If None, uses default population size.
        
        Returns:
            List[S]: List of initial solution instances.
        """
        pass

    @abstractmethod
    def evaluate(self, solution_list: List[S]) -> List[S]:
        """Evaluate a list of solutions.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
        
        Returns:
            List[S]: List of evaluated solutions with objective values set.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `evaluate`.")

    @abstractmethod
    def initialization(self) -> None:
        """Initialize the algorithm.
        
        Sets up the algorithm by creating initial solutions and evaluating them.
        Also records the start time and logs algorithm and problem information.
        
        Note:
            Subclasses must implement this method to define initialization logic.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `initialization`.")

    @abstractmethod
    def init_progress(self) -> None:
        """Initialize progress tracking.
        
        Called once before the main algorithm loop begins.
        Used to set up progress tracking mechanisms.
        """
        logger.debug("Initializing progress...")
        raise NotImplementedError(f"{self.__class__.__name__} must implement `init_progress`.")

    @abstractmethod
    def stopping_condition_is_met(self) -> bool:
        """Check if the stopping condition is met.
        
        Returns:
            bool: True if the algorithm should stop, False otherwise.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `stopping_condition_is_met`.")

    @abstractmethod
    def evolve(self) -> None:
        """Perform one iteration/evolution step of the algorithm.
        
        This is the core method where the main optimization logic is implemented.
        Called repeatedly in the main algorithm loop until stopping condition is met.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `evolve`.")

    @abstractmethod
    def update_progress(self) -> None:
        """Update progress after each iteration.
        
        Called after each evolution step to update progress tracking,
        notify observers, and log progress information.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `update_progress`.")

    @abstractmethod
    def get_observable_data(self) -> dict:
        """Get observable data for observers.
        
        Returns:
            dict: Dictionary containing data to be sent to observers.
                Typically includes current iteration, evaluations, best solution, etc.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_observable_data`.")

    def run(self) -> None:
        """Execute the algorithm."""
        self.initialization()
        self.after_initialization()
        self.init_progress()
        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.evolve()
            self.after_evolve()
            self.update_progress()
        logger.debug("Finished!")
        self.total_computing_time = time.time() - self.start_computing_time

    @abstractmethod
    def after_initialization(self) -> None:
        """Perform actions after initial population creation.
        
        Called once after the initial population has been created and evaluated,
        but before the main algorithm loop begins. Can be used for algorithm-specific
        initialization tasks such as:
        - Setting initial best solutions
        - Initializing algorithm-specific data structures
        - Computing initial statistics
        
        Note:
            Subclasses must implement this method, even if it's a no-op.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `after_initialization`.")

    @abstractmethod
    def after_evolve(self) -> None:
        """Perform actions after each evolution step.
        
        Called after each iteration of the evolve() method, before
        update_progress(). Can be used for algorithm-specific post-processing.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `after_evolve`.")

    @abstractmethod
    def get_result(self) -> R:
        """Get the final result of the algorithm.
        
        Returns:
            R: The final result. For single-objective algorithms, this is typically
                the best solution. For multi-objective algorithms, this is typically
                a list of non-dominated solutions.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_result`.")

    def get_name(self) -> str:
        """Get the name of the algorithm."""
        return self.algorithm_name

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class SingleObjectiveSwarmRoot(Algorithm[S, R], ABC):
    """Base class for single-objective swarm intelligence algorithms.
    
    Swarm-based algorithms are population-based metaheuristics inspired by
    collective behavior of natural swarms (e.g., particle swarms, ant colonies,
    bird flocks). They maintain a population of candidate solutions and update
    them based on interactions between individuals.
    
    This class extends Algorithm with swarm-specific features:
    - Personal best (p_best) and global best (g_best) tracking
    - Population-based solution generation and evaluation
    - Selection, reproduction, and replacement operators
    
    Attributes:
        problem (Problem[S]): Problem instance to solve.
        population_size (int): Size of the population.
        offspring_population_size (int): Number of offspring generated per iteration.
        max_evaluations (Optional[int]): Maximum function evaluations limit.
        max_iterations (int): Maximum iteration limit.
        p_best (Optional[S]): Personal best solution (typically best in current population).
        g_best (Optional[S]): Global best solution (best solution found so far).
        comparator (Comparator): Comparator for solution comparison.
        population_generator (Generator): Generator for creating initial solutions.
        mutation_operator (Mutation): Mutation operator for creating variation.
        population_evaluator (Evaluator): Evaluator for batch solution evaluation.
        termination_criterion (TerminationCriterion): Stopping condition.
        best_solution_selection (Selection): Selection operator for finding best solution.
        sort_population (SortPopulation): Population sorting utility.
        replacement_operator (Replacement): Operator for population replacement.
        restart_operator (Restart): Operator for handling duplicate solutions.
    
    Note:
        Subclasses should implement selection(), reproduction(), and replacement()
        methods to define the specific swarm algorithm behavior (e.g., PSO, ABC, GWO).
    
    Example:
        >>> class PSO(SingleObjectiveSwarmRoot):
        ...     def selection(self, population):
        ...         # Implement particle selection
        ...         pass
        ...     
        ...     def reproduction(self, selected):
        ...         # Implement velocity and position update
        ...         pass
    """

    def __init__(self, problem: Problem[S], population_size: int = 100):
        """Initialize swarm-based algorithm.
        
        Args:
            problem (Problem[S]): Problem instance to solve.
            population_size (int, optional): Size of the swarm population.
                Defaults to 100.
        """
        super(SingleObjectiveSwarmRoot, self).__init__(problem)
        self.population_size = population_size
        self.offspring_population_size = self.population_size
        self.max_evaluations = None
        self.max_iterations = 1000
        self.p_best, self.g_best = None, None
        self.comparator = store.default_comparator
        self.identical_solutions_comparator = IdenticalSolutionsComparator()
        self.population_generator = store.default_generator
        self.mutation_operator = store.default_mutation["real"]
        self.population_evaluator = store.default_evaluator
        self.termination_criterion = store.default_termination_criteria
        self.best_solution_selection = BestSolutionSelection(comparator=ObjectiveComparator(0))
        self.sort_population = SortPopulation(comparator=ObjectiveComparator(0))
        self.replacement_operator = GreedyPopulationReplacement(comparator=ObjectiveComparator(0))
        self.restart_operator = SimpleReplaceDuplicatedSolution()

    def create_solution(self) -> S:
        """Create a new solution using the population generator.
        
        Returns:
            S: A newly created solution instance with random variable values.
        """
        new_solution = self.population_generator.create_solution(self.problem)
        return new_solution

    def evaluate_solution(self, solution) -> S:
        """Evaluate a solution.
        
        Args:
            solution (S): Solution to evaluate.
        
        Returns:
            S: Evaluated solution with objective values set.
        """
        solution = self.problem.evaluate_solution(solution)
        return solution

    def create_initial_solutions(self, population_size=None) -> List[S]:
        """Create initial population of solutions.
        
        Args:
            population_size (Optional[int]): Size of population to create.
                If None, uses self.population_size. Defaults to None.
        
        Returns:
            List[S]: List of randomly initialized solution instances.
        """
        if population_size is None:
            return [self.create_solution() for _ in range(self.population_size)]
        else:
            return [self.create_solution() for _ in range(population_size)]

    def evaluate(self, solution_list: List[S]):
        """Evaluate a list of solutions in batch.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
        
        Returns:
            List[S]: List of evaluated solutions with objective values set.
        
        Note:
            Uses population_evaluator for efficient batch evaluation.
        """
        solution_list = self.population_evaluator.evaluate(solution_list, self.problem)
        return solution_list

    def selection(self, population: List[S]) -> List[S]:
        """Select solutions for reproduction (parents).
        
        Args:
            population (List[S]): Current population.
        
        Returns:
            List[S]: Selected solutions for reproduction.
        
        Note:
            Default implementation returns the population as-is.
            Subclasses should override this method to implement selection strategies.
        """
        return population

    def reproduction(self, population: List[S]) -> List[S]:
        """Generate offspring solutions through variation operators.
        
        Args:
            population (List[S]): Parent solutions.
        
        Returns:
            List[S]: Offspring solutions generated from parents.
        
        Note:
            Default implementation does nothing. Subclasses should override
            this method to implement reproduction (e.g., position updates in
            PSO, crossover/mutation in genetic algorithms, etc.).
        """
        pass

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        """Replace population members with new solutions.
        
        Args:
            population (List[S]): Current population.
            offsprings (List[S]): New solutions to integrate.
        
        Returns:
            List[S]: Updated population after replacement.
        
        Note:
            Default implementation uses replacement_operator to replace solutions.
            Subclasses can override for custom replacement strategies.
        """
        return self.replacement_operator.replace(population, offsprings)

    def initialization(self) -> None:
        """Initialize the algorithm.
        
        Sets up the algorithm by recording start time, logging algorithm and problem info,
        creating initial solutions and evaluating them.
        """
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

    def stopping_condition_is_met(self) -> bool:
        """Check if the stopping condition is met.
        
        Returns:
            bool: True if algorithm should stop (optimality found or termination
                criterion met), False otherwise.
        
        Note:
            Checks both problem.optimality_found flag and termination_criterion.
        """
        if self.problem.optimality_found:
            return True
        return self.termination_criterion.is_met

    def get_observable_data(self) -> dict:
        """Get observable data dictionary for observers.
        
        Returns:
            dict: Dictionary containing:
                - "PROBLEM": Problem instance
                - "EVALUATIONS": Number of function evaluations
                - "SOLUTIONS": Current best solution(s)
                - "TOTAL_TIME": Total elapsed time in seconds
        """
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.get_result(),
            "TOTAL_TIME": time.time() - self.start_computing_time,
        }

    def init_progress(self) -> None:
        """Initialize progress tracking after initial population creation.
        
        Sets up evaluation counter, iteration counter, and initializes
        termination criteria limits based on the termination criterion type.
        Notifies observers with initial progress data.
        
        Note:
            Automatically calculates max_iterations based on termination criterion:
            - StoppingByIterations: Uses max_iterations directly
            - StoppingByEvaluations: Calculates iterations from evaluations
            - StoppingByTime: Estimates iterations from time limit and evaluation rate
        """
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        if self.termination_criterion.get_name() == "StoppingByIterations":
            self.max_iterations = self.termination_criterion.max_iterations
        if self.termination_criterion.get_name() == "StoppingByEvaluations":
            self.max_evaluations = self.termination_criterion.max_evaluations
            self.max_iterations = int(self.max_evaluations / self.offspring_population_size)
        if self.termination_criterion.get_name() == "StoppingByTime":
            self.max_evaluations = int(
                self.termination_criterion.max_seconds / (self.total_computing_time / self.evaluations))
            self.max_iterations = int(self.max_evaluations / self.offspring_population_size)

    def evolve(self) -> None:
        """Perform one iteration of the swarm algorithm.
        
        Executes the main evolutionary cycle:
        1. Selection: Choose solutions for reproduction
        2. Reproduction: Generate new solutions (offspring)
        3. Evaluation: Evaluate offspring solutions
        4. Replacement: Update population with new solutions
        
        Note:
            Subclasses typically override selection(), reproduction(), and
            replacement() methods rather than evolve() itself.
        """
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)

    def update_progress(self) -> None:
        """Update progress tracking after each iteration.
        
        Increments evaluation and iteration counters, updates computing time,
        notifies observers, and recalculates iteration limits if using time-based
        termination.
        
        Note:
            For StoppingByTime, recalculates max_evaluations and max_iterations
            based on current evaluation rate to adapt to actual computation speed.
        """
        self.evaluations += self.offspring_population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        if self.termination_criterion.get_name() == "StoppingByTime":
            self.max_evaluations = int(
                self.termination_criterion.max_seconds / (self.total_computing_time / self.evaluations))
            self.max_iterations = int(self.max_evaluations / self.offspring_population_size)

    def run(self) -> None:
        """Execute the complete swarm algorithm.
        
        Runs the algorithm through the full lifecycle:
        1. Initialization: Create and evaluate initial population
        2. After initialization: Set initial best solutions
        3. Initialize progress: Set up tracking and notify observers
        4. Main loop: Evolve until stopping condition is met
        5. Finalize: Record total computing time
        
        Note:
            The main loop calls evolve(), after_evolve(), and update_progress()
            repeatedly until stopping_condition_is_met() returns True.
        """
        self.initialization()
        self.after_initialization()
        self.init_progress()
        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.evolve()
            self.after_evolve()
            self.update_progress()
        logger.debug("Finished!")
        self.total_computing_time = time.time() - self.start_computing_time

    @property
    def label(self) -> str:
        """Get label identifying the algorithm and problem and problem combination.
        
        Returns:
            str: String in format "AlgorithmName.ProblemName" for identification.
        """
        return f"{self.get_name()}.{self.problem.get_name()}"

    def after_initialization(self) -> None:
        """Perform actions after initial population creation.
        
        Sets p_best and g_best to the best solution in the initial population,
        and stores g_best in problem.g_best for reference.
        
        Note:
            p_best and g_best start as the same solution but may diverge
            as the algorithm progresses.
        """
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        self.g_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        self.problem.g_best = self.g_best

    def after_evolve(self) -> None:
        """Perform actions after each evolution step.
        
        Updates p_best (best in current population) and g_best (global best).
        If p_best is better than g_best, updates g_best and stores it in problem.
        
        Note:
            This method is called after evolve() but before update_progress().
            It ensures g_best always tracks the best solution found so far.
        """
        self.p_best = copy.deepcopy(self.best_solution_selection.execute(self.solutions))
        if self.comparator.compare(self.p_best, self.g_best) == -1:
            self.g_best = copy.deepcopy(self.p_best)
            self.problem.g_best = self.g_best
        # self.solutions = self.restart_operator.execute(self.solutions)

    def get_result(self) -> R:
        """Get the final result of the single-objective algorithm.
        
        Returns:
            R: List containing the global best solution [g_best].
                For single-objective swarm algorithms, this is the best solution found.
        
        Note:
            Returns as a list to maintain consistency with multi-objective algorithms.
        """
        return [self.g_best]

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class MultiObjectiveSwarmRoot(Algorithm[S, R], ABC):
    """Base class for multi-objective swarm intelligence algorithms.
    
    This class extends the general Algorithm with multi-objective specific features.
    Unlike single-objective algorithms, multi-objective algorithms maintain an archive
    of non-dominated solutions as the result.
    
    Attributes:
        problem (Problem[S]): Problem instance to solve.
        population_size (int): Size of the population.
        offspring_population_size (int): Number of offspring generated per iteration.
        max_evaluations (Optional[int]): Maximum function evaluations limit.
        max_iterations (int): Maximum iteration limit.
        comparator (Comparator): Comparator for solution comparison.
        population_generator (Generator): Generator for creating initial solutions.
        mutation_operator (Mutation): Mutation operator for creating variation.
        population_evaluator (Evaluator): Evaluator for batch solution evaluation.
        termination_criterion (TerminationCriterion): Stopping condition.
        best_solution_selection (Selection): Selection operator for finding best solution.
        sort_population (SortPopulation): Population sorting utility.
        replacement_operator (Replacement): Operator for population replacement.
        restart_operator (Restart): Operator for handling duplicate solutions.
        result_archive (NonDominatedSolutionsArchive): Archive for storing non-dominated solutions.
        dominance_comparator (Comparator): Comparator for comparing solutions based on dominance.
    """
    def __init__(self, problem: Problem[S], population_size: int = 100):
        super(MultiObjectiveSwarmRoot, self).__init__(problem)
        self.population_size = population_size
        self.offspring_population_size = self.population_size
        self.max_evaluations = None
        self.max_iterations = 1000
        self.comparator = store.default_comparator
        self.identical_solutions_comparator = IdenticalSolutionsComparator()
        self.population_generator = store.default_generator
        self.mutation_operator = store.default_mutation["real"]
        self.population_evaluator = store.default_evaluator
        self.termination_criterion = store.default_termination_criteria
        self.best_solution_selection = BestSolutionSelection(comparator=ObjectiveComparator(0))
        self.sort_population = SortPopulation(comparator=ObjectiveComparator(0))
        self.replacement_operator = GreedyPopulationReplacement(comparator=ObjectiveComparator(0))
        self.restart_operator = SimpleReplaceDuplicatedSolution()
        from evolu.util.archive import NonDominatedSolutionsArchive
        from evolu.util.comparator import DominanceWithConstraintsComparator
        from evolu.util.evaluator import Evaluator
        from evolu.util.generator import Generator
        from evolu.util.termination_criterion import TerminationCriterion
        self.result_archive = NonDominatedSolutionsArchive(DominanceWithConstraintsComparator())
        self.dominance_comparator = DominanceWithConstraintsComparator()

    def create_solution(self) -> S:
        """Create a new solution using the population generator.
        
        Returns:
            S: A newly created solution instance with random variable values.
        """
        new_solution = self.population_generator.create_solution(self.problem)
        return new_solution

    def evaluate_solution(self, solution) -> S:
        """Evaluate a solution and optionally add to archive.
        
        Args:
            solution (S): Solution to evaluate.
        
        Returns:
            S: Evaluated solution with objective values set.
        
        Note:
            For multi-objective algorithms, the solution is added to the result archive.
        """
        solution = self.problem.evaluate_solution(solution)
        self.result_archive.add(solution)
        return solution

    def create_initial_solutions(self, population_size=None) -> List[S]:
        """Create initial population of solutions.
        
        Args:
            population_size (Optional[int]): Size of population to create.
                If None, uses self.population_size. Defaults to None.
        
        Returns:
            List[S]: List of randomly initialized solution instances.
        """
        if population_size is None:
            return [self.create_solution() for _ in range(self.population_size)]
        else:
            return [self.create_solution() for _ in range(population_size)]

    def evaluate(self, solution_list: List[S]):
        """Evaluate a list of solutions in batch.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
        
        Returns:
            List[S]: List of evaluated solutions with objective values set.
        
        Note:
            Uses population_evaluator for efficient batch evaluation.
            Solutions are added to result_archive.
        """
        solution_list = self.population_evaluator.evaluate(solution_list, self.problem)
        for solution in solution_list:
            self.result_archive.add(solution)
        return solution_list

    def selection(self, population: List[S]) -> List[S]:
        """Select solutions for reproduction (parents).
        
        Args:
            population (List[S]): Current population.
        
        Returns:
            List[S]: Selected solutions for reproduction.
        
        Note:
            Default implementation returns the population as-is.
            Subclasses should override this method to implement selection strategies.
        """
        return population

    def reproduction(self, population: List[S]) -> List[S]:
        """Generate offspring solutions through variation operators.
        
        Args:
            population (List[S]): Parent solutions.
        
        Returns:
            List[S]: Offspring solutions generated from parents.
        
        Note:
            Default implementation does nothing. Subclasses should override
            this method to implement reproduction (e.g., position updates in
            PSO, crossover/mutation in genetic algorithms, etc.).
        """
        pass

    def replacement(self, population: List[S], offsprings: List[S]) -> List[S]:
        """Replace population members with new solutions.
        
        Args:
            population (List[S]): Current population.
            offsprings (List[S]): New solutions to integrate.
        
        Returns:
            List[S]: Updated population after replacement.
        
        Note:
            Default implementation uses replacement_operator to replace solutions.
            Subclasses can override for custom replacement strategies.
        """
        return self.replacement_operator.replace(population, offsprings)

    def initialization(self) -> None:
        """Initialize the algorithm.
        
        Sets up the algorithm by recording start time, logging algorithm and problem info,
        creating initial solutions and evaluating them.
        """
        self.start_computing_time = time.time()
        logger.info("The running algorithm is: " + self.get_name())
        logger.info("The problem solved now is: " + self.problem.get_name())
        self.solutions = self.create_initial_solutions()
        self.solutions = self.evaluate(self.solutions)

    def stopping_condition_is_met(self) -> bool:
        """Check if the stopping condition is met.
        
        Returns:
            bool: True if algorithm should stop (optimality found or termination
                criterion met), False otherwise.
        
        Note:
            Checks both problem.optimality_found flag and termination_criterion.
        """
        if self.problem.optimality_found:
            return True
        return self.termination_criterion.is_met

    def get_observable_data(self) -> dict:
        """Get observable data dictionary for observers.
        
        Returns:
            dict: Dictionary containing:
                - "PROBLEM": Problem instance
                - "EVALUATIONS": Number of function evaluations
                - "SOLUTIONS": Current non-dominated solutions
                - "TOTAL_TIME": Total elapsed time in seconds
        """
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result_archive.solution_list,
            "TOTAL_TIME": time.time() - self.start_computing_time,
        }

    def init_progress(self) -> None:
        """Initialize progress tracking after initial population creation.
        
        Sets up evaluation counter, iteration counter, and initializes
        termination criteria limits based on the termination criterion type.
        Notifies observers with initial progress data.
        
        Note:
            Automatically calculates max_iterations based on termination criterion:
            - StoppingByIterations: Uses max_iterations directly
            - StoppingByEvaluations: Calculates iterations from evaluations
            - StoppingByTime: Estimates iterations from time limit and evaluation rate
        """
        logger.debug("Initializing progress...")
        self.evaluations = self.population_size
        self.iterations = 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)
        if self.termination_criterion.get_name() == "StoppingByIterations":
            self.max_iterations = self.termination_criterion.max_iterations
        if self.termination_criterion.get_name() == "StoppingByEvaluations":
            self.max_evaluations = self.termination_criterion.max_evaluations
            self.max_iterations = int(self.max_evaluations / self.offspring_population_size)
        if self.termination_criterion.get_name() == "StoppingByTime":
            self.max_evaluations = int(
                self.termination_criterion.max_seconds / (self.total_computing_time / self.evaluations))
            self.max_iterations = int(self.max_evaluations / self.offspring_population_size)

    def evolve(self) -> None:
        """Perform one iteration of the swarm algorithm.
        
        Executes the main evolutionary cycle:
        1. Selection: Choose solutions for reproduction
        2. Reproduction: Generate new solutions (offspring)
        3. Evaluation: Evaluate offspring solutions
        4. Replacement: Update population with new solutions
        
        Note:
            Subclasses typically override selection(), reproduction(), and
            replacement() methods rather than evolve() itself.
        """
        selected_solutions = self.selection(self.solutions)
        offsprings = self.reproduction(selected_solutions)
        offsprings = self.evaluate(offsprings)
        self.solutions = self.replacement(self.solutions, offsprings)

    def update_progress(self) -> None:
        """Update progress tracking after each iteration for multi-objective algorithms.
        
        Increments evaluation and iteration counters, updates computing time,
        notifies observers with the current non-dominated solutions.
        """
        self.evaluations += self.offspring_population_size
        self.iterations += 1
        self.total_computing_time = time.time() - self.start_computing_time
        observable_data = self.get_observable_data()
        observable_data["SOLUTIONS"] = self.result_archive.solution_list
        self.observable.notify_all(**observable_data)
        if self.termination_criterion.get_name() == "StoppingByTime":
            self.max_evaluations = int(
                self.termination_criterion.max_seconds / (self.total_computing_time / self.evaluations))
            self.max_iterations = int(self.max_evaluations / self.offspring_population_size)

    def run(self) -> None:
        """Execute the complete swarm algorithm.
        
        Runs the algorithm through the full lifecycle:
        1. Initialization: Create and evaluate initial population
        2. After initialization: Set initial best solutions
        3. Initialize progress: Set up tracking and notify observers
        4. Main loop: Evolve until stopping condition is met
        5. Finalize: Record total computing time
        
        Note:
            The main loop calls evolve(), after_evolve(), and update_progress()
            repeatedly until stopping_condition_is_met() returns True.
        """
        self.initialization()
        self.after_initialization()
        self.init_progress()
        logger.debug("Running main loop until termination criteria is met")
        while not self.stopping_condition_is_met():
            self.evolve()
            self.after_evolve()
            self.update_progress()
        logger.debug("Finished!")
        self.total_computing_time = time.time() - self.start_computing_time

    @property
    def label(self) -> str:
        """Get label identifying the algorithm and problem combination.
        
        Returns:
            str: String in format "AlgorithmName.ProblemName" for identification.
        """
        return f"{self.get_name()}.{self.problem.get_name()}"

    def after_initialization(self) -> None:
        pass

    def after_evolve(self) -> None:
        pass

    def get_result(self) -> R:
        """Get the final result of the multi-objective algorithm.
        
        Returns:
            R: List of non-dominated solutions found during the optimization process.
        """
        return self.result_archive.solution_list

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name


class DynamicAlgorithm(Algorithm[S, R], ABC):
    """Base class for dynamic optimization algorithms.
    
    Dynamic algorithms are designed to handle dynamic problems where the
    problem definition changes over time. This class extends Algorithm with
    restart capabilities to adapt to problem changes.
    
    Dynamic problems can change in various ways:
    - Objective functions change (time-varying objectives)
    - Constraints change
    - Optimal solutions move (moving peaks, changing environments)
    - Problem parameters change
    
    Subclasses should implement restart() to define how the algorithm adapts
    when problem changes are detected (typically through DynamicProblem.the_problem_has_changed()).
    
    Note:
        Dynamic algorithms often work with DynamicProblem instances that
        implement the Observer interface to notify algorithms of changes.
    
    Example:
        >>> class DynamicPSO(DynamicAlgorithm, SingleObjectiveSwarmRoot):
        ...     def restart(self) -> None:
        ...         # Reinitialize part of population or reset velocities
        ...         # when problem change is detected
        ...         pass
    """
    
    @abstractmethod
    def restart(self) -> None:
        """Restart or reinitialize the algorithm to adapt to problem changes.
        
        This method is called when a problem change is detected. The algorithm
        should implement appropriate adaptation strategies, such as:
        - Reinitializing part of the population
        - Resetting algorithm parameters
        - Preserving some solutions as memory
        - Adjusting operator parameters
        
        Note:
            Subclasses must implement this method to define restart behavior.
            The method should be efficient as it may be called multiple times
            during execution.
        
        Raises:
            NotImplementedError: If subclass doesn't implement this method.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `restart`.")

    def get_description(self) -> str:
        """Get the description of the algorithm."""
        return self.algorithm_name
