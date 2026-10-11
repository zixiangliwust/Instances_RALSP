# -*- coding: utf-8 -*-
import copy
import random
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar

from evolu.core.observer import Observer
from evolu.core.solution import BinaryArraySolution, FloatSolution, IntegerSolution, PermutationSolution
from evolu.logger import get_logger

logger = get_logger(__name__)
S = TypeVar("S")


class Problem(Generic[S], ABC):
    """Base class representing optimization problems.
    
    Defines the mathematical form of optimization problems. Subclasses must
    implement abstract methods to define specific problem types.
    
    Attributes:
        number_of_variables (int): Number of decision variables.
        number_of_objectives (int): Number of objective functions.
        number_of_constraints (int): Number of constraint functions.
        obj_directions (List[int]): Objective function directions (MINIMIZE=-1, MAXIMIZE=1).
        objective_weights (Optional[List[float]]): Weights for objective functions.
        problem_name (Optional[str]): Name of the problem.
        problem_file_path (Optional[str]): Path to problem instance file.
        solution_type (str): Type of solution encoding.
        reference_front (List[S]): Reference Pareto front (for multi-objective problems).
        directions (List[int]): Objective directions.
        labels (List[str]): Objective function labels.
        optimality_found (bool): Whether optimal solution has been found.
        g_best (Optional[S]): Global best solution found so far.
    
    Constants:
        MINIMIZE (int): Constant for minimization direction (-1).
        MAXIMIZE (int): Constant for maximization direction (1).
    """
    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self) -> None:
        """Initialize problem instance."""
        self.number_of_variables: int = 1
        self.number_of_objectives: int = 1
        self.number_of_constraints: int = 0
        self.obj_directions: List[int] = [self.MINIMIZE]
        self.objective_weights: Optional[List[float]] = None
        self.problem_name: Optional[str] = None
        self.problem_file_path: Optional[str] = None
        self.solution_type: str = ""
        self.reference_front: List[S] = []
        self.directions: List[int] = []
        self.labels: List[str] = []
        self.optimality_found: bool = False
        self.g_best: Optional[S] = None
        self.p_best: Optional[S] = None
        self.precision: List[int] = []
        self.length: List[int] = []

    def get_number_of_objectives(self) -> int:
        """Get the number of objective functions.
        
        Returns:
            int: Number of objectives in this problem.
        """
        return self.number_of_objectives

    def get_number_of_variables(self) -> int:
        """Get the number of decision variables.
        
        Returns:
            int: Number of decision variables in this problem.
        """
        return self.number_of_variables

    def get_number_of_constraints(self) -> int:
        """Get the number of constraint functions.
        
        Returns:
            int: Number of constraints in this problem (0 if unconstrained).
        """
        return self.number_of_constraints

    @abstractmethod
    def create_solution(self) -> S:
        """Create a random solution to the problem.
        
        Returns:
            S: A newly created solution instance with random variable values.
        
        Note:
            Subclasses must implement this method to create solution instances
            appropriate for the problem's encoding type (float, permutation, etc.).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `create_solution`.")

    @abstractmethod
    def evaluate_solution(self, solution: S) -> S:
        """Evaluate a solution.
        
        Computes objective function values and constraint violations for the given solution.
        
        Args:
            solution (S): The solution to evaluate.
        
        Returns:
            S: The evaluated solution with objectives and constraints set.
        
        Note:
            This framework assumes minimization by default. If a problem has
            maximization objectives, they should be negated during evaluation.
            For any new problem inheriting from Problem, this method must be
            implemented to define the specific evaluation logic.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `evaluate_solution`.")

    def get_name(self) -> str:
        return self.problem_name

    def evaluate_constraints(self, solution: S) -> S:
        """Evaluate constraints for a solution.
        
        Args:
            solution (S): The solution to evaluate constraints for.
        
        Returns:
            S: The solution with constraint values set.
        
        Note:
            Subclasses can override this method to implement custom constraint evaluation.
            Default implementation does nothing.
        """
        return solution

    def evaluate_stop_constraints(self, solution: S) -> bool:
        """Evaluate stopping constraints for a solution.
        
        Args:
            solution (S): The solution to check.
        
        Returns:
            bool: True if stopping constraints are met, False otherwise.
        
        Note:
            Subclasses can override this method to implement custom stopping criteria.
            Default implementation returns False.
        """
        return False

    def get_number_of_bits(self) -> int:
        """Get the number of bits (for binary representations).
        
        Returns:
            int: Number of bits. Default is 0 for non-binary problems.
        
        Note:
            Subclasses (e.g., BinaryArrayProblem) should override this method.
        """
        return 0

    def check_solution(self, solution: S) -> None:
        """Check solution validity.
        
        Args:
            solution (S): The solution to check.
        
        Note:
            Subclasses can override this method to implement custom validation logic.
            Default implementation does nothing.
        """
        pass


class FloatProblem(Problem[FloatSolution], ABC):
    """Base class for continuous optimization problems.
    
    This class represents optimization problems with continuous (floating-point)
    decision variables. Each variable is constrained to lie within specified
    lower and upper bounds.
    
    Attributes:
        lower_bound (List[float]): List of lower bounds for each decision variable.
        upper_bound (List[float]): List of upper bounds for each decision variable.
        solution_type (str): Set to "Float" for float problems.
    
    Example:
        >>> class MyFloatProblem(FloatProblem):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.lower_bound = [0.0, 0.0]
        ...         self.upper_bound = [10.0, 10.0]
        ...         self.number_of_variables = 2
        ...     
        ...     def evaluate_solution(self, solution):
        ...         # Implement evaluation logic
        ...         return solution
    """

    def __init__(self) -> None:
        """Initialize float problem."""
        super(FloatProblem, self).__init__()
        self.lower_bound: List[float] = []
        self.upper_bound: List[float] = []
        self.solution_type = "Float"

    def create_variables(self, lower_bound: Optional[List[float]] = None, upper_bound: Optional[List[float]] = None) -> List[float]:
        """Create random continuous variables within bounds.
        
        Generates random float values for each variable, uniformly distributed
        within the specified bounds. If bounds are not provided, uses the
        problem's default bounds.
        
        Args:
            lower_bound (Optional[List[float]]): Lower bounds for variables.
                If None, uses self.lower_bound. Defaults to None.
            upper_bound (Optional[List[float]]): Upper bounds for variables.
                If None, uses self.upper_bound. Defaults to None.
        
        Returns:
            List[float]: List of random float values, one for each variable.
        """
        if lower_bound is None or upper_bound is None:
            variables = [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0)
                         for i in range(self.number_of_variables)]
        else:
            variables = [random.uniform(lower_bound[i] * 1.0, upper_bound[i] * 1.0)
                         for i in range(self.number_of_variables)]
        return variables

    def create_solution(self) -> FloatSolution:
        """Create a new FloatSolution instance with random variables.
        
        Creates a FloatSolution with variables randomly initialized within
        the problem's bounds.
        
        Returns:
            FloatSolution: New solution instance with random variable values
                within bounds.
        """
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, self.number_of_variables, self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def remedy_solution(self, solution: FloatSolution) -> FloatSolution:
        """Repair solution variables to ensure they are within bounds.
        
        Clips any variables that are outside the allowed bounds to the nearest
        bound value. This is useful for constraint handling when operators
        might generate out-of-bounds values.
        
        Args:
            solution (FloatSolution): Solution to repair.
        
        Returns:
            FloatSolution: Solution with all variables clipped to bounds.
        
        Note:
            Variables below lower_bound are set to lower_bound.
            Variables above upper_bound are set to upper_bound.
        """
        variables = copy.deepcopy(solution.variables)
        for i in range(0, len(variables)):
            if variables[i] < self.lower_bound[i]:
                variables[i] = self.lower_bound[i]
            if variables[i] > self.upper_bound[i]:
                variables[i] = self.upper_bound[i]
        solution.variables = variables
        return solution


class PermutationProblem(Problem[PermutationSolution], ABC):
    """Base class for permutation-based optimization problems.
    
    This class represents optimization problems where solutions are represented
    as permutations (ordered sequences) of elements. Commonly used for problems
    like traveling salesman problem (TSP), job sequencing, etc.
    
    In permutation problems, variables represent positions in a permutation,
    and each variable contains a unique integer value from a set (typically
    [0, n-1] or [1, n]).
    
    Attributes:
        solution_type (str): Set to "Permutation" for permutation problems.
    
    Example:
        >>> class TSPProblem(PermutationProblem):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.number_of_variables = 10  # 10 cities
        ...     
        ...     def create_solution(self):
        ...         # Creates random permutation of city indices
        ...         return PermutationSolution(...)
        ...     
        ...     def evaluate_solution(self, solution):
        ...         # Evaluate tour length
        ...         return solution
    """

    def __init__(self) -> None:
        """Initialize permutation problem."""
        super(PermutationProblem, self).__init__()
        self.solution_type = "Permutation"


class IntegerProblem(Problem[IntegerSolution], ABC):
    """Base class for integer optimization problems.
    
    This class represents optimization problems with discrete integer decision
    variables. Each variable must be an integer value within specified lower
    and upper bounds.
    
    Attributes:
        lower_bound (List[int]): List of lower bounds for each decision variable.
        upper_bound (List[int]): List of upper bounds for each decision variable.
        solution_type (str): Set to "Integer" for integer problems.
    
    Example:
        >>> class KnapsackProblem(IntegerProblem):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.lower_bound = [0] * 10  # 10 items, each 0 or 1
        ...         self.upper_bound = [1] * 10
        ...         self.number_of_variables = 10
        ...     
        ...     def evaluate_solution(self, solution):
        ...         # Evaluate profit and weight
        ...         return solution
    """

    def __init__(self) -> None:
        """Initialize integer problem."""
        super(IntegerProblem, self).__init__()
        self.lower_bound: List[int] = []
        self.upper_bound: List[int] = []
        self.solution_type = "Integer"

    def create_variables(self, lower_bound: Optional[List[float]] = None, upper_bound: Optional[List[float]] = None) -> List[int]:
        """Create random integer variables within bounds.
        
        Generates random integer values for each variable, uniformly distributed
        within the specified bounds. Values are rounded to nearest integers.
        
        Args:
            lower_bound (Optional[List[float]]): Lower bounds for variables.
                If None, uses self.lower_bound. Defaults to None.
            upper_bound (Optional[List[float]]): Upper bounds for variables.
                If None, uses self.upper_bound. Defaults to None.
        
        Returns:
            List[int]: List of random integer values, one for each variable.
        
        Note:
            Floating-point bounds are accepted but values are rounded to integers.
        """
        if lower_bound is None or upper_bound is None:
            variables = [
                round(random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0))
                for i in range(self.number_of_variables)]
        else:
            variables = [
                round(random.uniform(lower_bound[i] * 1.0, upper_bound[i] * 1.0))
                for i in range(self.number_of_variables)]
        return variables

    def create_solution(self) -> IntegerSolution:
        """Create a new IntegerSolution instance with random variables.
        
        Creates an IntegerSolution with variables randomly initialized as
        integers within the problem's bounds.
        
        Returns:
            IntegerSolution: New solution instance with random integer variable
                values within bounds.
        """
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, self.number_of_objectives, self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def remedy_solution(self, solution: IntegerSolution) -> IntegerSolution:
        """Repair solution variables to ensure they are within bounds and are integers.
        
        Clips any variables that are outside the allowed bounds to the nearest
        bound value and rounds to nearest integer. This is useful for constraint 
        handling when operators might generate out-of-bounds or non-integer values.
        
        Args:
            solution (IntegerSolution): Solution to repair.
        
        Returns:
            IntegerSolution: Solution with all variables clipped to bounds and rounded.
        
        Note:
            Variables below lower_bound are set to lower_bound.
            Variables above upper_bound are set to upper_bound.
            All variables are rounded to nearest integer.
        """
        variables = copy.deepcopy(solution.variables)
        for i in range(0, len(variables)):
            if variables[i] < self.lower_bound[i]:
                variables[i] = self.lower_bound[i]
            if variables[i] > self.upper_bound[i]:
                variables[i] = self.upper_bound[i]
            variables[i] = round(variables[i])
        solution.variables = variables
        return solution


class BinaryArrayProblem(Problem[BinaryArraySolution], ABC):
    """Base class for binary optimization problems.
    
    This class represents optimization problems where solutions are encoded
    as binary arrays (bit strings). Each variable may be represented by one
    or more bits, allowing for different precision levels per variable.
    
    Binary encoding is commonly used for discrete optimization problems where
    decisions are yes/no or on/off (e.g., binary knapsack, feature selection).
    
    Attributes:
        solution_type (str): Set to "BinaryArray" for binary problems.
        number_of_bits_per_variable (List[int]): Number of bits used to encode
            each decision variable. Length should equal number_of_variables.
    
    Example:
        >>> class BinaryKnapsack(BinaryArrayProblem):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.number_of_bits_per_variable = [1] * 10  # 1 bit per item
        ...         self.number_of_variables = 10
        ...     
        ...     def create_solution(self):
        ...         # Creates binary solution
        ...         return BinaryArraySolution(...)
    """

    def __init__(self) -> None:
        """Initialize binary array problem."""
        super(BinaryArrayProblem, self).__init__()
        self.solution_type = "BinaryArray"
        self.number_of_bits_per_variable: List[int] = []

    def number_of_bits_per_variable_list(self) -> List[int]:
        """Get the list of bits per variable.
        
        Returns:
            List[int]: List where each element indicates the number of bits
                used to encode the corresponding decision variable.
        """
        return self.number_of_bits_per_variable

    def total_number_of_bits(self) -> int:
        """Get the total number of bits in the binary encoding.
        
        Returns:
            int: Sum of all bits across all variables. This is the total
                length of the binary array representation.
        """
        return sum(self.number_of_bits_per_variable)


class DynamicProblem(Problem[S], Observer, ABC):
    """Base class for dynamic optimization problems.
    
    Dynamic problems are those where the problem definition (objective functions,
    constraints, or optimal solutions) change over time during the optimization
    process. This class combines Problem and Observer interfaces to allow problems
    to notify algorithms about changes.
    
    Subclasses must implement methods to detect and report when the problem
    has changed, enabling algorithms to adapt to these changes.
    
    Note:
        This is useful for:
        - Time-varying objective functions
        - Changing constraint sets
        - Dynamic environments (e.g., moving targets)
        - Real-time optimization scenarios
    
    Example:
        >>> class DynamicTSP(DynamicProblem, PermutationProblem):
        ...     def the_problem_has_changed(self) -> bool:
        ...         # Check if city positions have changed
        ...         return self.cities_moved
        ...     
        ...     def clear_changed(self) -> None:
        ...         self.cities_moved = False
    """
    
    @abstractmethod
    def the_problem_has_changed(self) -> bool:
        """Check if the problem has changed since last check.
        
        Returns:
            bool: True if the problem definition has changed, False otherwise.
        
        Note:
            Algorithms can call this method to detect changes and trigger
            adaptation mechanisms (e.g., re-initialization, memory mechanisms).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `the_problem_has_changed`.")

    @abstractmethod
    def clear_changed(self) -> None:
        """Clear the problem change flag.
        
        Should be called after the algorithm has detected and handled the
        problem change, resetting the change status.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `clear_changed`.")


class OnTheFlyFloatProblem(FloatProblem):
    """ Class for defining float problems on the fly.
        Example:
        >>> # Defining problem Srinivas on the fly
        >>> def f1(x: [float]):
        >>>     return 2.0 + (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 1.0) * (x[1] - 1.0)
        >>>
        >>> def f2(x: [float]):
        >>>     return 9.0 * x[0] - (x[1] - 1.0) * (x[1] - 1.0)
        >>>
        >>> def c1(x: [float]):
        >>>     return 1.0 - (x[0] * x[0] + x[1] * x[1]) / 225.0
        >>>
        >>> def c2(x: [float]):
        >>>     return (3.0 * x[1] - x[0]) / 10.0 - 1.0
        >>>
        >>> problem = OnTheFlyFloatProblem()\
            .set_name("Srinivas")\
            .add_variable(-20.0, 20.0)\
            .add_variable(-20.0, 20.0)\
            .add_function(f1)\
            .add_function(f2)\
            .add_constraint(c1)\
            .add_constraint(c2)
    """

    def __init__(self) -> None:
        """Initialize on-the-fly float problem."""
        super(OnTheFlyFloatProblem, self).__init__()
        from typing import Callable
        self.functions: List[Callable[[List[float]], float]] = []
        self.constraints: List[Callable[[List[float]], float]] = []
        self.problem_name: Optional[str] = None

    def set_name(self, name: str) -> "OnTheFlyFloatProblem":
        """Set the problem name.
        
        Args:
            name (str): Name identifier for the problem.
        
        Returns:
            OnTheFlyFloatProblem: Self for method chaining.
        """
        self.problem_name = name
        return self

    def add_function(self, function) -> "OnTheFlyFloatProblem":
        """Add an objective function.
        
        Args:
            function (Callable[[List[float]], float]): Objective function that
                takes variable list and returns objective value.
        
        Returns:
            OnTheFlyFloatProblem: Self for method chaining.
        
        Note:
            Functions are added in order and correspond to objectives in order.
        """
        self.functions.append(function)
        self.number_of_objectives += 1
        return self

    def add_constraint(self, constraint) -> "OnTheFlyFloatProblem":
        """Add a constraint function.
        
        Args:
            constraint (Callable[[List[float]], float]): Constraint function that
                takes variable list and returns constraint violation value.
                Values <= 0 indicate satisfaction, > 0 indicates violation.
        
        Returns:
            OnTheFlyFloatProblem: Self for method chaining.
        
        Note:
            Constraints are added in order and correspond to constraint indices.
        """
        self.constraints.append(constraint)
        self.number_of_constraints += 1
        return self

    def add_variable(self, lower_bound: float, upper_bound: float) -> "OnTheFlyFloatProblem":
        """Add a decision variable with bounds.
        
        Args:
            lower_bound (float): Lower bound for the variable.
            upper_bound (float): Upper bound for the variable.
        
        Returns:
            OnTheFlyFloatProblem: Self for method chaining.
        
        Note:
            Variables are added in order. The order determines variable indices.
        """
        self.lower_bound.append(lower_bound)
        self.upper_bound.append(upper_bound)
        self.number_of_variables += 1
        return self

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        """Evaluate solution using registered functions and constraints.
        
        Calls each objective function and constraint function in order,
        storing results in the solution's objectives and constraints arrays.
        
        Args:
            solution (FloatSolution): Solution to evaluate.
        
        Returns:
            FloatSolution: Solution with objectives and constraints computed.
        """
        for i in range(self.number_of_objectives):
            solution.objectives[i] = self.functions[i](solution.variables)
        for i in range(self.number_of_constraints):
            solution.constraints[i] = self.constraints[i](solution.variables)
        return solution

    def get_name(self) -> str:
        """Get the problem name.
        
        Returns:
            str: Problem name, or None if not set.
        """
        return self.problem_name
