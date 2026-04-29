# -*- coding: utf-8 -*-
import numpy as np
import random
import copy
import math
from evolu.core.operator import Mutation
from evolu.core.solution import (
    BinaryArraySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)
from evolu.util.checking import Check

"""
module:: mutation
synopsis: Module implementing mutation operators.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""



class NullMutation(Mutation[Solution]):
    def __init__(self):
        super(NullMutation, self).__init__(probability=0)

    def execute(self, parent: Solution) -> Solution:
        """Return a deep copy of the parent without any modification.

        This operator is useful for debugging or when mutation needs to be
        disabled while keeping the algorithm structure unchanged.
        """
        return copy.deepcopy(parent)

    def get_name(self) -> str:
        """Get the name of the mutation operator."""
        return "NullMutation"




class FloatPolynomialMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(FloatPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parent: FloatSolution) -> FloatSolution:
        """Apply polynomial mutation to a float solution.

        The operator perturbs each decision variable using a polynomial
        probability distribution controlled by ``distribution_index``.

        Args:
            parent (FloatSolution): Parent solution to be mutated.

        Returns:
            FloatSolution: Mutated offspring solution.
        """
        child = copy.deepcopy(parent)
        Check.that(issubclass(type(child), FloatSolution), "Solution type invalid")
        for i in range(parent.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                y = child.variables[i]
                yl, yu = child.lower_bound[i], child.upper_bound[i]
                if yl == yu:
                    y = yl
                else:
                    rnd = random.random()
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = pow(val, mut_pow) - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (pow(xy, self.distribution_index + 1.0))
                        deltaq = 1.0 - pow(val, mut_pow)
                    y += deltaq * (yu - yl)
                    if y < child.lower_bound[i]:
                        y = child.lower_bound[i]
                    if y > child.upper_bound[i]:
                        y = child.upper_bound[i]
                child.variables[i] = y
        return child

    def get_name(self):
        return "FloatPolynomialMutation"




class FloatUniformMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5):
        super(FloatUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, parent: FloatSolution) -> FloatSolution:
        """Apply uniform mutation to a float solution.

        The operator adds a random value in a symmetric interval around each
        decision variable, bounded by the variable limits.
        """
        child = copy.deepcopy(parent)
        Check.that(type(child) is FloatSolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                value = (random.random() - 0.5) * self.perturbation
                value += child.variables[i]
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                elif value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = value
        return child

    def get_name(self):
        return "FloatUniformMutation"




class FloatNonUniformMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 100):
        super(FloatNonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        return y * (1.0 - pow(random.random(),
                              pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter)))

    def execute(self, parent: FloatSolution) -> FloatSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is FloatSolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            if random.random() <= self.probability:
                rand = random.random()
                if rand <= 0.5:
                    value = self.__delta(child.upper_bound[i] - child.variables[i], self.perturbation)
                else:
                    value = self.__delta(child.lower_bound[i] - child.variables[i], self.perturbation)
                value += child.variables[i]
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                elif value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = value
        return child

    def get_name(self):
        return "FloatNonUniformMutation"




class FloatSimpleFlipMutation(Mutation[Solution]):
    def __init__(self, probability=1.0):
        super(FloatSimpleFlipMutation, self).__init__(probability=probability)

    def execute(self, parent: FloatSolution) -> FloatSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is FloatSolution, "Solution type invalid")
        rand = random.random()
        if rand <= self.probability:
            idx = random.randint(0, len(parent.variables) - 1)
            child.variables[idx] = (
                    child.lower_bound[idx] + (child.upper_bound[idx] - child.lower_bound[idx]) * random.random()
            )
        return child

    def get_name(self):
        return "FloatSimpleFlipMutation"




class FloatSimpleRandomMutation(Mutation[FloatSolution]):
    def __init__(self, probability: float):
        super(FloatSimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, parent: FloatSolution) -> FloatSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is FloatSolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                child.variables[i] = (
                        child.lower_bound[i] + (child.upper_bound[i] - child.lower_bound[i]) * random.random()
                )
        return child

    def get_name(self):
        return "FloatSimpleRandomMutation"




class FloatOppositionMutation(Mutation[Solution]):
    def __init__(self, probability=1.0):
        super(FloatOppositionMutation, self).__init__(probability=probability)

    def execute(self, parent: Solution = None, best_solution: Solution = None) -> Solution:
        if parent is None or best_solution is None:
            raise ValueError("FloatOppositionMutation requires both parent and best_solution")
        child = copy.deepcopy(parent)
        rand = random.random()
        if rand <= self.probability:
            for i in range(0, len(child.variables)):
                child.variables[i] = parent.lower_bound[i] + parent.upper_bound[i] - best_solution.variables[
                    i] + random.uniform(0.0, 1.0) * (best_solution.variables[i] - parent.variables[i])
                if child.variables[i] < child.lower_bound[i]:
                    child.variables[i] = child.lower_bound[i]
                elif child.variables[i] > child.upper_bound[i]:
                    child.variables[i] = child.upper_bound[i]
        return child

    def get_name(self):
        return "FloatOppositionMutation"




class FloatPermutationIntegerSwapMutation(Mutation[Solution]):
    def __init__(self, probability: float = 1.0):
        super(FloatPermutationIntegerSwapMutation, self).__init__(probability=probability)

    def execute(self, parent: Solution) -> Solution:
        child = copy.deepcopy(parent)
        rand = random.random()
        if rand <= self.probability:
            if parent.number_of_variables >= 2:
                idx1, idx2 = random.sample(range(parent.number_of_variables), 2)
                number_of_times = 0
                while parent.variables[idx1] == parent.variables[idx2]:
                    idx1, idx2 = random.sample(range(parent.number_of_variables), 2)
                    number_of_times += 1
                    if number_of_times > parent.number_of_variables:
                        break
                child.variables[idx1], child.variables[idx2] = parent.variables[idx2], parent.variables[idx1]
        return child

    def get_name(self):
        return "FloatPermutationIntegerSwapMutation"




class FloatPermutationIntegerInsertionMutation(Mutation[Solution]):
    def __init__(self, probability: float = 1.0):
        super(FloatPermutationIntegerInsertionMutation, self).__init__(probability=probability)

    def execute(self, parent: Solution) -> Solution:
        child = copy.deepcopy(parent)
        rand = random.random()
        if rand <= self.probability:
            number_of_variables = parent.number_of_variables
            if number_of_variables >= 2:
                temp_solution = copy.deepcopy(parent)
                old_idx, new_idx = random.sample(range(0, number_of_variables), 2)
                while old_idx == new_idx:
                    old_idx, new_idx = random.sample(range(0, number_of_variables), 2)
                for i in range(old_idx, number_of_variables):
                    if i + 1 < number_of_variables:
                        temp_solution.variables[i] = parent.variables[i + 1]
                child.variables[new_idx] = parent.variables[old_idx]
                for i in range(0, number_of_variables):
                    if i < new_idx:
                        child.variables[i] = temp_solution.variables[i]
                    if i > new_idx:
                        child.variables[i] = temp_solution.variables[i - 1]
        return child

    def get_name(self):
        return "FloatPermutationIntegerInsertionMutation"




class FloatPermutationIntegerInversionMutation(Mutation[Solution]):
    def __init__(self, probability: float = 1.0):
        super(FloatPermutationIntegerInversionMutation, self).__init__(probability=probability)

    def execute(self, parent: Solution) -> Solution:
        child = copy.deepcopy(parent)
        rand = random.random()
        if rand <= self.probability:
            if parent.number_of_variables >= 2:
                idx1, idx2 = random.sample(range(0, len(parent.variables)), 2)
                while idx1 >= idx2:
                    idx1, idx2 = random.sample(range(0, len(parent.variables)), 2)
                    if idx1 > idx2:
                        idx1, idx2 = idx2, idx1
                temp_variable = child.variables[idx1: idx2]
                temp_variable = temp_variable[::-1]
                child.variables[idx1: idx2] = temp_variable
        return child

    def get_name(self):
        return "FloatPermutationIntegerInversionMutation"




class FloatPermutationIntegerScrambleMutation(Mutation[PermutationSolution]):
    def __init__(self, probability: float = 1.0):
        super(FloatPermutationIntegerScrambleMutation, self).__init__(probability=probability)

    def execute(self, parent: PermutationSolution) -> PermutationSolution:
        child = copy.deepcopy(parent)
        rand = random.random()
        if rand <= self.probability:
            if parent.number_of_variables >= 2:
                point1 = random.randint(0, parent.number_of_variables - 1)
                point2 = random.randint(0, parent.number_of_variables - 2)
                if point2 >= point1:
                    point2 += 1
                else:
                    point1, point2 = point2, point1
                if point2 - point1 >= 20:
                    point2 = point1 + 20
                values = child.variables[point1:point2]
                child.variables[point1:point2] = random.sample(values, len(values))
        return child

    def get_name(self):
        return "FloatPermutationIntegerScrambleMutation"




class IntegerPolynomialMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(IntegerPolynomialMutation, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parent: IntegerSolution) -> IntegerSolution:
        """Apply polynomial mutation to an integer solution.

        This is the integer counterpart of ``FloatPolynomialMutation`` and
        rounds mutated values to the nearest integer within bounds.
        """
        child = copy.deepcopy(parent)
        Check.that(issubclass(type(child), IntegerSolution), "Solution type invalid")
        for i in range(parent.number_of_variables):
            if random.random() <= self.probability:
                y = child.variables[i]
                yl, yu = child.lower_bound[i], child.upper_bound[i]
                if yl == yu:
                    y = yl
                else:
                    delta1 = (y - yl) / (yu - yl)
                    delta2 = (yu - y) / (yu - yl)
                    mut_pow = 1.0 / (self.distribution_index + 1.0)
                    rnd = random.random()
                    if rnd <= 0.5:
                        xy = 1.0 - delta1
                        val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (self.distribution_index + 1.0))
                        deltaq = val ** mut_pow - 1.0
                    else:
                        xy = 1.0 - delta2
                        val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (self.distribution_index + 1.0))
                        deltaq = 1.0 - val ** mut_pow
                    y += deltaq * (yu - yl)
                    if y < child.lower_bound[i]:
                        y = child.lower_bound[i]
                    if y > child.upper_bound[i]:
                        y = child.upper_bound[i]
                child.variables[i] = int(round(y))
        return child

    def get_name(self):
        return "IntegerPolynomialMutation"




class IntegerUniformMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5):
        super(IntegerUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation

    def execute(self, parent: IntegerSolution) -> IntegerSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is IntegerSolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                value = (random.random() - 0.5) * self.perturbation
                value += child.variables[i]
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                elif value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = int(round(value))
        return child

    def get_name(self):
        return "IntegerUniformMutation"




class IntegerNonUniformMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float, perturbation: float = 0.5, max_iterations: int = 100):
        super(IntegerNonUniformMutation, self).__init__(probability=probability)
        self.perturbation = perturbation
        self.max_iterations = max_iterations
        self.current_iteration = 0

    def set_current_iteration(self, current_iteration: int):
        self.current_iteration = current_iteration

    def __delta(self, y: float, b_mutation_parameter: float):
        return y * (1.0 - pow(random.random(),
                              pow((1.0 - 1.0 * self.current_iteration / self.max_iterations), b_mutation_parameter)))

    def execute(self, parent: IntegerSolution) -> IntegerSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is IntegerSolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            if random.random() <= self.probability:
                rand = random.random()
                if rand <= 0.5:
                    value = self.__delta(child.upper_bound[i] - child.variables[i], self.perturbation)
                else:
                    value = self.__delta(child.lower_bound[i] - child.variables[i], self.perturbation)
                value += child.variables[i]
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                elif value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = int(round(value))
        return child

    def get_name(self):
        return "IntegerNonUniformMutation"




class IntegerSimpleFlipMutation(Mutation[Solution]):
    def __init__(self, probability=1.0):
        super(IntegerSimpleFlipMutation, self).__init__(probability=probability)

    def execute(self, parent: IntegerSolution) -> IntegerSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is IntegerSolution, "Solution type invalid")
        rand = random.random()
        if rand <= self.probability:
            idx = random.randint(0, len(parent.variables) - 1)
            original_value = child.variables[idx]
            value = original_value
            # Try up to 10 times to get a different value (matches C++ implementation)
            for attempt in range(10):
                value = child.lower_bound[idx] + (child.upper_bound[idx] - child.lower_bound[idx]) * random.random()
                if value < child.lower_bound[idx]:
                    value = child.lower_bound[idx]
                elif value > child.upper_bound[idx]:
                    value = child.upper_bound[idx]
                if int(round(value)) != int(round(original_value)):
                    break
            # Only update if we found a different value
            if int(round(value)) != int(round(original_value)):
                child.variables[idx] = int(round(value))
        return child

    def get_name(self):
        return "IntegerSimpleFlipMutation"




class IntegerSimpleRandomMutation(Mutation[IntegerSolution]):
    def __init__(self, probability: float):
        super(IntegerSimpleRandomMutation, self).__init__(probability=probability)

    def execute(self, parent: IntegerSolution) -> IntegerSolution:
        child = copy.deepcopy(parent)
        Check.that(type(child) is IntegerSolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            rand = random.random()
            if rand <= self.probability:
                value = child.lower_bound[i] + (child.upper_bound[i] - child.lower_bound[i]) * random.random()
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                elif value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = int(round(value))
        return child

    def get_name(self):
        return "IntegerSimpleRandomMutation"




class IntegerOppositionMutation(Mutation[Solution]):
    def __init__(self, probability=1.0):
        super(IntegerOppositionMutation, self).__init__(probability=probability)

    def execute(self, parent: Solution = None, best_solution: Solution = None) -> Solution:
        if parent is None or best_solution is None:
            raise ValueError("IntegerOppositionMutation requires both parent and best_solution")
        child = copy.deepcopy(parent)
        rand = random.random()
        if rand <= self.probability:
            for i in range(0, len(child.variables)):
                value = parent.lower_bound[i] + parent.upper_bound[i] - best_solution.variables[
                    i] + random.uniform(0.0, 1.0) * (best_solution.variables[i] - parent.variables[i])
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                elif value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = int(round(value))
        return child

    def get_name(self):
        return "IntegerOppositionMutation"




class BitFlipMutation(Mutation[BinaryArraySolution]):
    def __init__(self, probability: float):
        super(BitFlipMutation, self).__init__(probability=probability)

    def execute(self, parent: BinaryArraySolution) -> BinaryArraySolution:
        """Flip bits of a binary array solution with a given probability.

        Args:
            parent (BinaryArraySolution): Parent solution to be mutated.

        Returns:
            BinaryArraySolution: Mutated offspring solution.
        """
        child = copy.deepcopy(parent)
        Check.that(type(parent) is BinaryArraySolution, "Solution type invalid")
        for i in range(parent.number_of_variables):
            for j in range(len(parent.variables[i])):
                rand = random.random()
                if rand <= self.probability:
                    child.variables[i][j] = True if parent.variables[i][j] is False else False
        return child

    def get_name(self):
        return "BitFlipMutation"




class AlternativeMutation(Mutation[Solution]):
    """Alternative mutation operator that selects one mutation operator probabilistically."""
    __EPS = 1.0e-14
    
    def __init__(self, mutation_operator_list: [Mutation]):
        super(AlternativeMutation, self).__init__(probability=1.0)
        Check.is_not_none(mutation_operator_list)
        Check.collection_is_not_empty(mutation_operator_list)
        self.mutation_operators_list = []
        for operator in mutation_operator_list:
            Check.that(issubclass(operator.__class__, Mutation), "Object is not a subclass of Mutation")
            self.mutation_operators_list.append(operator)

    def execute(self, parent: Solution) -> Solution:
        """Select and execute one mutation operator using roulette wheel selection.
        
        Args:
            parent (Solution): Parent solution to be mutated.
            
        Returns:
            Solution: Mutated solution from selected operator.
        """
        Check.is_not_none(parent)
        
        # Build probability list from operators
        probability_list = [op.probability for op in self.mutation_operators_list]
        
        # Roulette wheel selection
        maximum = sum(probability_list)
        rand_num = random.uniform(0.0, maximum)
        value = 0.0
        
        for i in range(len(probability_list)):
            value += probability_list[i]
            if value >= rand_num:
                return self.mutation_operators_list[i].execute(parent)
        
        # Fallback: return parent copy if no operator selected
        return copy.deepcopy(parent)

    def get_name(self) -> str:
        return "AlternativeMutation"




class AlternativeCompositeMutation(Mutation[CompositeSolution]):
    """Alternative composite mutation that selects ONE component to mutate."""
    __EPS = 1.0e-14
    
    def __init__(self, mutation_operator_list: [Mutation]):
        super(AlternativeCompositeMutation, self).__init__(probability=1.0)
        Check.is_not_none(mutation_operator_list)
        Check.collection_is_not_empty(mutation_operator_list)
        self.mutation_operators_list = []
        for operator in mutation_operator_list:
            Check.that(issubclass(operator.__class__, Mutation), "Object is not a subclass of Mutation")
            self.mutation_operators_list.append(operator)

    def execute(self, parent: CompositeSolution) -> CompositeSolution:
        """Select one component to mutate, copy others unchanged.
        
        Args:
            parent (CompositeSolution): Composite solution to mutate.
            
        Returns:
            CompositeSolution: Mutated composite solution with one component changed.
        """
        Check.is_not_none(parent)
        
        mutated_solution_components = []
        
        # Build probability list from operators
        probability_list = [op.probability for op in self.mutation_operators_list]
        
        # Roulette wheel selection to choose which component to mutate
        maximum = sum(probability_list)
        rand_num = random.uniform(0.0, maximum)
        value = 0.0
        
        number_of_components = parent.number_of_variables
        
        for i in range(number_of_components):
            value += probability_list[i]
            if value >= rand_num:
                # Mutate this component
                mutated_solution_components.append(self.mutation_operators_list[i].execute(parent.sub_solutions[i]))
                # Set rand_num to max to ensure no other component is selected
                rand_num = float('inf')
            else:
                # Copy this component unchanged from parent
                mutated_solution_components.append(copy.deepcopy(parent.sub_solutions[i]))
        
        return CompositeSolution(mutated_solution_components)

    def get_name(self) -> str:
        return "AlternativeCompositeMutation"




class CompositeMutation(Mutation[Solution]):
    def __init__(self, mutation_operator_list: [Mutation]):
        super(CompositeMutation, self).__init__(probability=1.0)
        Check.is_not_none(mutation_operator_list)
        Check.collection_is_not_empty(mutation_operator_list)
        self.mutation_operators_list = []
        for operator in mutation_operator_list:
            Check.that(issubclass(operator.__class__, Mutation), "Object is not a subclass of Mutation")
            self.mutation_operators_list.append(operator)

    def execute(self, parent: CompositeSolution) -> CompositeSolution:
        """Apply component-wise mutation to a composite solution.

        Each sub-solution is mutated using the corresponding mutation operator
        in ``mutation_operators_list``.

        Args:
            parent (CompositeSolution): Composite solution to mutate.

        Returns:
            CompositeSolution: Mutated composite solution.
        """
        child = copy.deepcopy(parent)
        Check.is_not_none(child)
        mutated_solution_components = []
        for i in range(parent.number_of_variables):
            mutated_solution_components.append(self.mutation_operators_list[i].execute(child.sub_solutions[i]))
        return CompositeSolution(mutated_solution_components)

    def get_name(self) -> str:
        return "CompositeMutation"




class LevyFlightOperator(Mutation[FloatSolution]):
    def __init__(self, probability=1.0):
        super(LevyFlightOperator, self).__init__(probability=probability)

    def levy_flight_step(self, beta=1.0, multiplier=0.001, case=0):
        """Compute a Lévy-flight step size.

        Args:
            beta (float): Stability parameter in \\([0, 2]\\). Values in
                \\([0, 1]\\) favor exploitation, and values in \\((1, 2]\\)
                favor exploration.
            multiplier (float): Scale factor applied to the step size.
            case (int): How to post-process the Lévy step. Supported values:
                ``0`` (uniform scaling), ``1`` (Gaussian scaling),
                ``-1`` (no additional randomness).

        Returns:
            float: The Lévy-flight step size.
        """
        # u and v are two random variables which follow np.random.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                    math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        u = np.random.normal(0, sigma_u ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = u / np.power(abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * np.random.uniform()
        elif case == 1:
            step = multiplier * s * np.random.normal(0, 1)
        else:
            step = multiplier * s
        return step

    def execute(self, iterations=None, parent=None, best_solution=None, step=0.001, levy_type=0):
        """Apply a Lévy-flight based perturbation to the parent.

        Args:
            iterations (int, optional): Current iteration number of the
                algorithm.
            parent (FloatSolution): Current solution to be perturbed.
            best_solution (FloatSolution): Current best-known solution.
            step (float): Base step size used in the Lévy-flight move.
            levy_type (int): Strategy used to combine the Lévy step with the
                current solution, controlling exploitation vs exploration.

        Returns:
            FloatSolution: Mutated solution after performing the Lévy-flight
            move.
        """
        if parent is None or best_solution is None:
            raise ValueError("LevyFlightOperator requires both parent and best_solution")
        variables, g_best_position = parent.variables, best_solution.variables
        beta = 1
        # muy and v are two random variables which follow np.random.normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(
            math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
                    math.gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = muy / np.power(abs(v), 1 / beta)
        # Use parent's bounds instead of self.problem
        lb = np.array(parent.lower_bound)
        ub = np.array(parent.upper_bound)
        # Generate uniform random values for each dimension
        uniform_rand = np.array([random.uniform(lb[i], ub[i]) for i in range(len(lb))])
        levy = uniform_rand * step * s * (np.array(variables) - np.array(g_best_position))
        if levy_type == 0:
            parent.variables = levy.tolist() if isinstance(levy, np.ndarray) else levy
            return parent
        elif levy_type == 1:
            if iterations is None:
                raise ValueError("LevyFlightOperator with levy_type=1 requires iterations parameter")
            parent.variables = (np.array(variables) + 1.0 / np.sqrt(iterations + 1) * np.sign(np.random.random() - 0.5) * levy).tolist()
            return parent
        elif levy_type == 2:
            parent.variables = (np.array(variables) + np.random.normal(0, 1, len(lb)) * levy).tolist()
            return parent
        elif levy_type == 3:
            parent.variables = (np.array(variables) + 0.01 * levy).tolist()
            return parent

    def get_name(self):
        return "LevyFlightOperator"


