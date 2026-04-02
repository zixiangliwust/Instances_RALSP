# -*- coding: utf-8 -*-
import copy
import random
from typing import List
from evolu.core.operator import Crossover
from evolu.core.solution import (
    FloatSolution,
    BinaryArraySolution,
    CompositeSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)
from evolu.core.exceptions import InvalidParentsException, InvalidVariantException
from evolu.util.checking import Check

"""
module:: crossover
synopsis: Module implementing crossover operators.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class NullCrossover(Crossover[Solution, Solution]):
    """Crossover operator that returns an exact copy of the parents.

    This operator is typically used for debugging or ablation studies where
    the effect of crossover needs to be disabled while keeping the rest of
    the algorithm unchanged.
    """

    def __init__(self):
        super(NullCrossover, self).__init__(probability=0.0)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        """Return deep copies of the two parent solutions.

        Args:
            parents (List[Solution]): List containing exactly two parent solutions.

        Returns:
            List[Solution]: Two offspring that are deep copies of the parents.

        Raises:
            InvalidParentsException: If the number of parents is not two.
        """
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        return copy.deepcopy(parents)

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        """Get the name of the crossover operator."""
        return "NullCrossover"


class FloatDifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """Differential evolution (DE) crossover operator for float solutions.

    This operator receives two parameters: the current individual and an array
    of three parent individuals. The *best* and *rand* variants depend on the
    third parent, according to whether it represents the current or the
    globally best individual. The implementation of the variants is the same;
    the difference lies in how parents are selected outside this operator.

    Implemented variants include, among others:

    - ``rand/1/bin`` / ``best/1/bin``
    - ``rand/1/exp`` / ``best/1/exp``
    - ``current-to-rand/1`` / ``current-to-best/1``
    - ``current-to-rand/1/bin`` / ``current-to-best/1/bin``
    - ``current-to-rand/1/exp`` / ``current-to-best/1/exp``
    """

    def __init__(
        self,
        CR: float = 0.5,
        F: float = 0.5,
        K: float = 0.5,
        DE_Variant: str = "rand/1/bin",
    ):
        super(FloatDifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K
        self.DE_Variant = DE_Variant
        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Execute the differential evolution crossover.

        Args:
            parents (List[FloatSolution]): List of three parent solutions used
                to build the trial solution.

        Returns:
            List[FloatSolution]: A list containing a single offspring solution.

        Raises:
            InvalidParentsException: If the number of parents is not the
                expected one for the configured DE variant.
            InvalidVariantException: If an unsupported DE variant is provided.
        """
        if len(parents) != self.get_number_of_parents():
            raise InvalidParentsException(
                f"The number of parents is not {self.get_number_of_parents()}: {len(parents)}"
            )
        child = copy.deepcopy(self.current_individual)
        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)
        if self.DE_Variant == "rand/1/bin" or self.DE_Variant == "best/1/bin":
            for i in range(number_of_variables):
                if random.random() < self.CR or i == rand:
                    value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])
                    if value < child.lower_bound[i]:
                        value = child.lower_bound[i]
                    if value > child.upper_bound[i]:
                        value = child.upper_bound[i]
                    child.variables[i] = value
                # else: keep original value (no assignment needed)
        elif self.DE_Variant == "rand/1/exp" or self.DE_Variant == "best/1/exp":
            cr_local = self.CR
            for i in range(number_of_variables):
                if random.random() < cr_local or i == rand:
                    value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])
                    if value < child.lower_bound[i]:
                        value = child.lower_bound[i]
                    if value > child.upper_bound[i]:
                        value = child.upper_bound[i]
                    child.variables[i] = value
                else:
                    cr_local = 0.0
                    # Keep original value
                    pass
        elif self.DE_Variant == "current-to-rand/1" or self.DE_Variant == "current-to-best/1":
            for i in range(number_of_variables):
                value = self.current_individual.variables[i] + self.K * (
                        parents[2].variables[i] - self.current_individual.variables[i]) + self.F * (
                                parents[0].variables[i] - parents[1].variables[i])
                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
                child.variables[i] = value
        elif self.DE_Variant == "current-to-rand/1/bin" or self.DE_Variant == "current-to-best/1/bin":
            for i in range(number_of_variables):
                if random.random() < self.CR or i == rand:
                    value = self.current_individual.variables[i] + self.K * (
                            parents[2].variables[i] - self.current_individual.variables[i]) + self.F * (
                                    parents[0].variables[i] - parents[1].variables[i])
                    if value < child.lower_bound[i]:
                        value = child.lower_bound[i]
                    if value > child.upper_bound[i]:
                        value = child.upper_bound[i]
                else:
                    value = child.variables[i]
                child.variables[i] = value
        elif self.DE_Variant == "current-to-rand/1/exp" or self.DE_Variant == "current-to-best/1/exp":
            cr_local = self.CR
            for i in range(number_of_variables):
                if random.random() < cr_local or i == rand:
                    value = self.current_individual.variables[i] + self.K * (
                            parents[2].variables[i] - self.current_individual.variables[i]) + self.F * (
                                    parents[0].variables[i] - parents[1].variables[i])
                    if value < child.lower_bound[i]:
                        value = child.lower_bound[i]
                    if value > child.upper_bound[i]:
                        value = child.upper_bound[i]
                    child.variables[i] = value
                else:
                    cr_local = 0.0
                    # Keep original value
                    pass
        else:
            raise InvalidVariantException(
                f"DifferentialEvolutionCrossover.execute: Invalid DE_Variant '{self.DE_Variant}'"
            )
        return [child]

    def get_number_of_parents(self) -> int:
        return 3

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return "FloatDifferentialEvolutionCrossover"


class FloatIntegerUniformCrossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self):
        super(FloatIntegerUniformCrossover, self).__init__(probability=1.0)

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            for i in range(0, len(parents[0].variables)):
                if random.random() < 0.5:
                    offsprings[0].variables[i] = parents[0].variables[i]
                    offsprings[1].variables[i] = parents[1].variables[i]
                else:
                    offsprings[0].variables[i] = parents[1].variables[i]
                    offsprings[1].variables[i] = parents[0].variables[i]
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "FloatIntegerUniformCrossover"


class FloatArithmeticCrossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self):
        super(FloatArithmeticCrossover, self).__init__(probability=1.0)

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            r = random.random()
            for i in range(0, len(parents[0].variables)):
                offsprings[0].variables[i] = parents[0].variables[i] * r + parents[1].variables[i] * (1.0 - r)
                offsprings[1].variables[i] = parents[1].variables[i] * r + parents[0].variables[i] * (1.0 - r)
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "FloatArithmeticCrossover"


class FloatSimulatedBinaryCrossover(Crossover[FloatSolution, FloatSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(FloatSimulatedBinaryCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        if distribution_index < 0:
            from evolu.core.exceptions import InvalidParameterException
            raise InvalidParameterException(
                f"The distribution index is negative: {distribution_index}"
            )

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Apply simulated binary crossover (SBX) to two float solutions.

        SBX simulates the behavior of single-point crossover on binary strings
        in the continuous domain and is widely used for real-coded
        evolutionary algorithms.

        Args:
            parents (List[FloatSolution]): Two parent solutions.

        Returns:
            List[FloatSolution]: Two offspring generated by SBX.
        """
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid: " + str(type(parents[0])))
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]
                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1
                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[0].upper_bound[i]
                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))
                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound
                        if random.random() <= 0.5:
                            offsprings[0].variables[i] = c2
                            offsprings[1].variables[i] = c1
                        else:
                            offsprings[0].variables[i] = c1
                            offsprings[1].variables[i] = c2
                    else:
                        offsprings[0].variables[i] = value_x1
                        offsprings[1].variables[i] = value_x2
                else:
                    offsprings[0].variables[i] = value_x1
                    offsprings[1].variables[i] = value_x2
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "FloatSimulatedBinaryCrossover"


class PermutationOnePointCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PermutationOnePointCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        """Apply one-point crossover to permutation solutions.

        The operator preserves permutation feasibility by copying a segment from
        the first parent and filling the remaining positions with genes from the
        second parent in order.
        """
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            if parents[0].number_of_variables >= 2:
                crossover_point = random.randint(1, parents[0].number_of_variables - 1) if parents[0].number_of_variables > 1 else 0
                for i in range(0, crossover_point):
                    offsprings[0].variables[i] = parents[0].variables[i]
                index_of_current_gen_to_change = crossover_point
                for i in range(0, parents[1].number_of_variables):
                    is_in_the_sub_string_of_parent1 = False
                    for h in range(0, crossover_point):
                        if parents[1].variables[i] == parents[0].variables[h]:
                            is_in_the_sub_string_of_parent1 = True
                    if not is_in_the_sub_string_of_parent1:
                        offsprings[0].variables[index_of_current_gen_to_change] = parents[1].variables[i]
                        index_of_current_gen_to_change += 1
                for i in range(0, crossover_point):
                    offsprings[1].variables[i] = parents[1].variables[i]
                index_of_current_gen_to_change = crossover_point
                for i in range(0, parents[0].number_of_variables):
                    is_in_the_sub_string_of_parent2 = False
                    for h in range(0, crossover_point):
                        if parents[0].variables[i] == parents[1].variables[h]:
                            is_in_the_sub_string_of_parent2 = True
                    if not is_in_the_sub_string_of_parent2:
                        offsprings[1].variables[index_of_current_gen_to_change] = parents[0].variables[i]
                        index_of_current_gen_to_change += 1
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "PermutationOnePointCrossover"


class PermutationTwoPointCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PermutationTwoPointCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            if parents[0].number_of_variables > 2:
                crossover_point1, crossover_point2 = random.sample(range(parents[0].number_of_variables), 2)
                if crossover_point1 > crossover_point2:
                    crossover_point1, crossover_point2 = crossover_point2, crossover_point1
                for i in range(0, crossover_point1):
                    offsprings[0].variables[i] = parents[0].variables[i]
                for i in range(crossover_point2, parents[0].number_of_variables):
                    offsprings[0].variables[i] = parents[0].variables[i]
                index_of_current_gen_to_change = crossover_point1
                for i in range(0, parents[1].number_of_variables):
                    is_in_the_sub_string_of_parent1 = False
                    for h in range(0, crossover_point1):
                        if parents[1].variables[i] == parents[0].variables[h]:
                            is_in_the_sub_string_of_parent1 = True
                    for h in range(crossover_point2, parents[0].number_of_variables):
                        if parents[1].variables[i] == parents[0].variables[h]:
                            is_in_the_sub_string_of_parent1 = True
                    if not is_in_the_sub_string_of_parent1:
                        offsprings[0].variables[index_of_current_gen_to_change] = parents[1].variables[i]
                        index_of_current_gen_to_change += 1
                for i in range(0, crossover_point1):
                    offsprings[1].variables[i] = parents[1].variables[i]
                for i in range(crossover_point2, parents[1].number_of_variables):
                    offsprings[1].variables[i] = parents[1].variables[i]
                index_of_current_gen_to_change = crossover_point1
                for i in range(0, parents[0].number_of_variables):
                    is_in_the_sub_string_of_parent2 = False
                    for h in range(0, crossover_point1):
                        if parents[0].variables[i] == parents[1].variables[h]:
                            is_in_the_sub_string_of_parent2 = True
                    for h in range(crossover_point2, parents[1].number_of_variables):
                        if parents[0].variables[i] == parents[1].variables[h]:
                            is_in_the_sub_string_of_parent2 = True
                    if not is_in_the_sub_string_of_parent2:
                        offsprings[1].variables[index_of_current_gen_to_change] = parents[0].variables[i]
                        index_of_current_gen_to_change += 1
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "PermutationTwoPointCrossover"


class PermutationOrderCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PermutationOrderCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            if parents[0].number_of_variables >= 2:
                crossover_point1, crossover_point2 = random.sample(range(parents[0].number_of_variables), 2)
                if crossover_point1 > crossover_point2:
                    crossover_point1, crossover_point2 = crossover_point2, crossover_point1
                index_of_current_gen_to_change = 0
                if index_of_current_gen_to_change == crossover_point1:
                    index_of_current_gen_to_change = crossover_point2
                for i in range(0, parents[0].number_of_variables):
                    is_in_the_sub_string_of_parent1 = False
                    for h in range(crossover_point1, crossover_point2):
                        if parents[1].variables[i] == parents[0].variables[h]:
                            is_in_the_sub_string_of_parent1 = True
                    if is_in_the_sub_string_of_parent1 is False:
                        offsprings[0].variables[index_of_current_gen_to_change] = parents[1].variables[i]
                        index_of_current_gen_to_change += 1
                    if index_of_current_gen_to_change == crossover_point1:
                        index_of_current_gen_to_change = crossover_point2
                index_of_current_gen_to_change = 0
                if index_of_current_gen_to_change == crossover_point1:
                    index_of_current_gen_to_change = crossover_point2
                for i in range(0, parents[0].number_of_variables):
                    is_in_the_sub_string_of_parent2 = False
                    for h in range(crossover_point1, crossover_point2):
                        if parents[0].variables[i] == parents[1].variables[h]:
                            is_in_the_sub_string_of_parent2 = True
                    if is_in_the_sub_string_of_parent2 is False:
                        offsprings[1].variables[index_of_current_gen_to_change] = parents[0].variables[i]
                        index_of_current_gen_to_change += 1
                    if index_of_current_gen_to_change == crossover_point1:
                        index_of_current_gen_to_change = crossover_point2
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "PermutationOrderCrossover"


class PermutationCycleCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PermutationCycleCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents[::-1])
        rand = random.random()
        if rand <= self.probability:
            idx = random.randint(0, parents[0].number_of_variables - 1)
            curr_idx = idx
            cycle = []
            while True:
                cycle.append(curr_idx)
                curr_idx = parents[0].variables.index(parents[1].variables[curr_idx])
                if curr_idx == idx:
                    break
            for i in range(len(parents[0].variables)):
                if i in cycle:
                    offsprings[0].variables[i] = parents[0].variables[i]
                    offsprings[1].variables[i] = parents[1].variables[i]
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "PermutationCycleCrossover"


class PermutationPartiallyMatchedCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PermutationPartiallyMatchedCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            cross_points = sorted(random.sample(range(0, parents[0].number_of_variables), 2))

            def _repeated(element, collection):
                c = 0
                for e in collection:
                    if e == element:
                        c += 1
                return c > 1

            def _swap(data_a, data_b, cross_points):
                c1, c2 = cross_points
                new_a = data_a[:c1] + data_b[c1:c2] + data_a[c2:]
                new_b = data_b[:c1] + data_a[c1:c2] + data_b[c2:]
                return new_a, new_b

            def _map(swapped, cross_points):
                n = len(swapped[0])
                c1, c2 = cross_points
                map_ = swapped[0][c1:c2], swapped[1][c1:c2]
                for i_chromosome in range(n):
                    if not c1 < i_chromosome < c2:
                        for i_son in range(2):
                            while _repeated(swapped[i_son][i_chromosome], swapped[i_son]):
                                map_index = map_[i_son].index(swapped[i_son][i_chromosome])
                                swapped[i_son][i_chromosome] = map_[1 - i_son][map_index]
                return swapped

            swapped = _swap(parents[0].variables, parents[1].variables, cross_points)
            mapped = _map(swapped, cross_points)
            offsprings[0].variables, offsprings[1].variables = mapped
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "PermutationPartiallyMatchedCrossover"


class PermutationPositionBasedCrossover(Crossover[PermutationSolution, PermutationSolution]):
    def __init__(self, probability: float):
        super(PermutationPositionBasedCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[PermutationSolution]) -> List[PermutationSolution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            number_of_random_genes = random.randint(0, parents[0].number_of_variables - 1)
            selected_indexes = random.sample(range(0, parents[0].number_of_variables), number_of_random_genes)
            empty_indexes = list(range(0, len(parents[0].variables)))
            for i in selected_indexes:
                empty_indexes.remove(i)
            idx_of_empty_indexes = 0
            for i in range(0, len(parents[0].variables)):
                found = False
                for h in range(0, len(selected_indexes)):
                    if parents[1].variables[i] == parents[0].variables[selected_indexes[h]]:
                        found = True
                        break
                if found is False:
                    offsprings[0].variables[empty_indexes[idx_of_empty_indexes]] = parents[1].variables[i]
                    idx_of_empty_indexes += 1
            idx_of_empty_indexes = 0
            for i in range(0, len(parents[0].variables)):
                found = False
                for h in range(0, len(selected_indexes)):
                    if parents[0].variables[i] == parents[1].variables[selected_indexes[h]]:
                        found = True
                        break
                if found is False:
                    offsprings[1].variables[empty_indexes[idx_of_empty_indexes]] = parents[0].variables[i]
                    idx_of_empty_indexes += 1
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "PermutationPositionBasedCrossover"


class RepeatedPermutationOnePointCrossover(Crossover[Solution, Solution]):
    def __init__(self, probability: float):
        super(RepeatedPermutationOnePointCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            if parents[0].number_of_variables >= 2:
                crossover_point = random.randint(1, parents[0].number_of_variables - 1) if parents[0].number_of_variables > 1 else 0
                dad_variables = copy.deepcopy(parents[0].variables)
                mom_variables = copy.deepcopy(parents[1].variables)
                dad_variables = dad_variables[0:crossover_point]
                for i in dad_variables:
                    mom_variables.remove(i)
                dad_variables = dad_variables + mom_variables
                offsprings[0].variables = copy.deepcopy(dad_variables)
                dad_variables = copy.deepcopy(parents[0].variables)
                mom_variables = copy.deepcopy(parents[1].variables)
                mom_variables = mom_variables[0:crossover_point]
                for i in mom_variables:
                    dad_variables.remove(i)
                mom_variables = mom_variables + dad_variables
                offsprings[1].variables = copy.deepcopy(mom_variables)
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "RepeatedPermutationOnePointCrossover"


class RepeatedPermutationTwoPointCrossover(Crossover[Solution, Solution]):
    def __init__(self, probability: float):
        super(RepeatedPermutationTwoPointCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[Solution]) -> List[Solution]:
        if len(parents) != 2:
            raise InvalidParentsException(
                f"The number of parents is not two: {len(parents)}"
            )
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            if parents[0].number_of_variables > 2:
                crossover_point1, crossover_point2 = random.sample(range(parents[0].number_of_variables), 2)
                if crossover_point1 > crossover_point2:
                    crossover_point1, crossover_point2 = crossover_point2, crossover_point1
                dad_variables = copy.deepcopy(parents[0].variables)
                mom_variables = copy.deepcopy(parents[1].variables)
                dad_variables = dad_variables[0:crossover_point1] + dad_variables[crossover_point2:]
                for i in dad_variables:
                    mom_variables.remove(i)
                dad_variables = dad_variables[0:crossover_point1] + mom_variables[:] + dad_variables[crossover_point1:]
                offsprings[0].variables = copy.deepcopy(dad_variables)
                dad_variables = copy.deepcopy(parents[0].variables)
                mom_variables = copy.deepcopy(parents[1].variables)
                mom_variables = mom_variables[0:crossover_point1] + mom_variables[crossover_point2:]
                for i in mom_variables:
                    dad_variables.remove(i)
                mom_variables = mom_variables[0:crossover_point1] + dad_variables[:] + mom_variables[crossover_point1:]
                offsprings[1].variables = copy.deepcopy(mom_variables)
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "RepeatedPermutationTwoPointCrossover"


class IntegerArithmeticCrossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self):
        super(IntegerArithmeticCrossover, self).__init__(probability=1.0)

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            r = random.random()
            for i in range(0, len(parents[0].variables)):
                value = parents[0].variables[i] * r + parents[1].variables[i] * (1.0 - r)
                offsprings[0].variables[i] = int(round(value))
                value = parents[1].variables[i] * r + parents[0].variables[i] * (1.0 - r)
                offsprings[1].variables[i] = int(round(value))
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return "IntegerArithmeticCrossover"


class IntegerSimulatedBinaryCrossover(Crossover[IntegerSolution, IntegerSolution]):
    __EPS = 1.0e-14

    def __init__(self, probability: float, distribution_index: float = 20.0):
        super(IntegerSimulatedBinaryCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index

    def execute(self, parents: List[IntegerSolution]) -> List[IntegerSolution]:
        Check.that(issubclass(type(parents[0]), IntegerSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), IntegerSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]
                if random.random() <= 0.5:
                    if abs(value_x1 - value_x2) > self.__EPS:
                        if value_x1 < value_x2:
                            y1, y2 = value_x1, value_x2
                        else:
                            y1, y2 = value_x2, value_x1
                        lower_bound, upper_bound = parents[0].lower_bound[i], parents[0].upper_bound[i]
                        beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))
                        rand = random.random()
                        if rand <= (1.0 / alpha):
                            betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                        c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                        beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))
                        if rand <= (1.0 / alpha):
                            betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                        else:
                            betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                        c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
                        if c1 < lower_bound:
                            c1 = lower_bound
                        if c2 < lower_bound:
                            c2 = lower_bound
                        if c1 > upper_bound:
                            c1 = upper_bound
                        if c2 > upper_bound:
                            c2 = upper_bound
                        if random.random() <= 0.5:
                            offsprings[0].variables[i] = int(c2)
                            offsprings[1].variables[i] = int(c1)
                        else:
                            offsprings[0].variables[i] = int(c1)
                            offsprings[1].variables[i] = int(c2)
                    else:
                        offsprings[0].variables[i] = value_x1
                        offsprings[1].variables[i] = value_x2
                else:
                    offsprings[0].variables[i] = value_x1
                    offsprings[1].variables[i] = value_x2
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "IntegerSimulatedBinaryCrossover"


class BinaryArraySinglePointCrossover(Crossover[BinaryArraySolution, BinaryArraySolution]):
    def __init__(self, probability: float):
        super(BinaryArraySinglePointCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinaryArraySolution]) -> List[BinaryArraySolution]:
        """Apply single-point crossover on binary array solutions.

        The operator selects a bit position across the concatenated bit
        representation and swaps the tails of both parents from that point.
        """
        Check.that(type(parents[0]) is BinaryArraySolution, "Solution type invalid")
        Check.that(type(parents[1]) is BinaryArraySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offspring = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            # 1. Get the total number of bits
            total_number_of_bits = parents[0].get_total_number_of_bits()
            # 2. Calculate the point to make the crossover
            if total_number_of_bits > 1:
                crossover_point = random.randrange(0, total_number_of_bits - 1)
            else:
                crossover_point = 0
            # 3. Compute the variable containing the crossover bit
            variable_to_cut = 0
            bits_count = len(parents[1].variables[variable_to_cut])
            while bits_count < (crossover_point + 1):
                variable_to_cut += 1
                bits_count += len(parents[1].variables[variable_to_cut])
            # 4. Compute the bit into the selected variable
            diff = bits_count - crossover_point
            crossover_point_in_variable = len(parents[1].variables[variable_to_cut]) - diff
            # 5. Apply the crossover to the variable
            bitset1 = copy.copy(parents[0].variables[variable_to_cut])
            bitset2 = copy.copy(parents[1].variables[variable_to_cut])
            for i in range(crossover_point_in_variable, len(bitset1)):
                swap = bitset1[i]
                bitset1[i] = bitset2[i]
                bitset2[i] = swap
            offspring[0].variables[variable_to_cut] = bitset1
            offspring[1].variables[variable_to_cut] = bitset2
            # 6. Apply the crossover to the other variables
            for i in range(variable_to_cut + 1, parents[0].number_of_variables):
                offspring[0].variables[i] = copy.deepcopy(parents[1].variables[i])
                offspring[1].variables[i] = copy.deepcopy(parents[0].variables[i])
        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "BinaryArraySinglePointCrossover"


class BinaryArrayHalfUniformCrossover(Crossover[BinaryArraySolution, BinaryArraySolution]):
    def __init__(self, probability: float):
        super(BinaryArrayHalfUniformCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[BinaryArraySolution]) -> List[BinaryArraySolution]:
        Check.that(type(parents[0]) is BinaryArraySolution, "Solution type invalid")
        Check.that(type(parents[1]) is BinaryArraySolution, "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            for i in range(0, parents[0].number_of_variables):
                bitset1 = copy.copy(parents[0].variables[i])
                bitset2 = copy.copy(parents[1].variables[i])
                for bit in range(0, len(bitset1)):
                    rand = random.random()
                    if rand <= 0.5:
                        offsprings[0].variables[i][bit] = bitset2[bit]
                        offsprings[1].variables[i][bit] = bitset1[bit]
        return offsprings

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "BinaryArrayHalfUniformCrossover"


class CompositeCrossover(Crossover[CompositeSolution, CompositeSolution]):
    __EPS = 1.0e-14

    def __init__(self, crossover_operator_list: [Crossover]):
        super(CompositeCrossover, self).__init__(probability=1.0)
        Check.is_not_none(crossover_operator_list)
        Check.collection_is_not_empty(crossover_operator_list)
        self.crossover_operators_list = []
        for operator in crossover_operator_list:
            Check.that(issubclass(operator.__class__, Crossover), "Object is not a subclass of Crossover")
            self.crossover_operators_list.append(operator)

    def execute(self, solutions: List[CompositeSolution]) -> List[CompositeSolution]:
        """Apply component-wise crossover to composite solutions.

        Each component solution is crossed using the corresponding crossover
        operator stored in ``crossover_operators_list``.

        Args:
            solutions (List[CompositeSolution]): Two composite parent solutions.

        Returns:
            List[CompositeSolution]: Two composite offspring solutions.
        """
        Check.is_not_none(solutions)
        Check.that(len(solutions) == 2, "The number of parents is not two: " + str(len(solutions)))
        offspring1 = []
        offspring2 = []
        number_of_solutions_in_composite_solution = solutions[0].number_of_variables
        for i in range(0, number_of_solutions_in_composite_solution):
            parents = [solutions[0].sub_solutions[i], solutions[1].sub_solutions[i]]
            children = self.crossover_operators_list[i].execute(parents)
            offspring1.append(children[0])
            offspring2.append(children[1])
        return [CompositeSolution(offspring1), CompositeSolution(offspring2)]

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return "CompositeCrossover"


class FloatBLXAlphaCrossover(Crossover[FloatSolution, FloatSolution]):
    """BLX-α crossover operator for float solutions.
    
    BLX-α (Blend Crossover with α) is a crossover operator for real-coded
    genetic algorithms. It generates offspring in an extended interval around
    the parent values.
    
    References:
        Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms
        and interval-schemata. Foundations of genetic algorithms, 2, 187-202.
    """
    
    def __init__(self, probability: float, alpha: float = 0.5):
        """Initialize BLX-α crossover operator.
        
        Args:
            probability (float): Probability of applying crossover (0.0 to 1.0).
            alpha (float): Expansion parameter. Higher values generate offspring
                further from parents. Typical value is 0.5. Defaults to 0.5.
        """
        super(FloatBLXAlphaCrossover, self).__init__(probability=probability)
        self.alpha = alpha
        if alpha < 0:
            from evolu.core.exceptions import InvalidParameterException
            raise InvalidParameterException(
                f"The alpha parameter is negative: {alpha}"
            )
    
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Apply BLX-α crossover to two float solutions.
        
        Args:
            parents (List[FloatSolution]): Two parent solutions.
            
        Returns:
            List[FloatSolution]: Two offspring generated by BLX-α crossover.
        """
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                x1, x2 = parents[0].variables[i], parents[1].variables[i]
                d = abs(x1 - x2)
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                # Extended interval
                lower = x_min - self.alpha * d
                upper = x_max + self.alpha * d
                # Clip to bounds
                lower = max(lower, parents[0].lower_bound[i])
                upper = min(upper, parents[0].upper_bound[i])
                # Generate two offspring
                offsprings[0].variables[i] = random.uniform(lower, upper)
                offsprings[1].variables[i] = random.uniform(lower, upper)
        return offsprings
    
    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 2
    
    def get_name(self) -> str:
        return "FloatBLXAlphaCrossover"


class FloatBLXAlphaBetaCrossover(Crossover[FloatSolution, FloatSolution]):
    """BLX-αβ crossover operator for float solutions.
    
    BLX-αβ is an extension of BLX-α that uses two parameters (α and β) to
    control the expansion on both sides of the parent interval independently.
    
    References:
        Deep, K., & Thakur, M. (2007). A new crossover operator for real coded
        genetic algorithms. Applied Mathematics and Computation, 188(1), 895-911.
    """
    
    def __init__(self, probability: float, alpha: float = 0.5, beta: float = 0.5):
        """Initialize BLX-αβ crossover operator.
        
        Args:
            probability (float): Probability of applying crossover (0.0 to 1.0).
            alpha (float): Expansion parameter for the lower side. Defaults to 0.5.
            beta (float): Expansion parameter for the upper side. Defaults to 0.5.
        """
        super(FloatBLXAlphaBetaCrossover, self).__init__(probability=probability)
        self.alpha = alpha
        self.beta = beta
        if alpha < 0 or beta < 0:
            from evolu.core.exceptions import InvalidParameterException
            raise InvalidParameterException(
                f"The alpha or beta parameter is negative: alpha={alpha}, beta={beta}"
            )
    
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Apply BLX-αβ crossover to two float solutions.
        
        Args:
            parents (List[FloatSolution]): Two parent solutions.
            
        Returns:
            List[FloatSolution]: Two offspring generated by BLX-αβ crossover.
        """
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            for i in range(parents[0].number_of_variables):
                x1, x2 = parents[0].variables[i], parents[1].variables[i]
                d = abs(x1 - x2)
                x_min = min(x1, x2)
                x_max = max(x1, x2)
                # Extended interval with independent parameters
                lower = x_min - self.alpha * d
                upper = x_max + self.beta * d
                # Clip to bounds
                lower = max(lower, parents[0].lower_bound[i])
                upper = min(upper, parents[0].upper_bound[i])
                # Generate two offspring
                offsprings[0].variables[i] = random.uniform(lower, upper)
                offsprings[1].variables[i] = random.uniform(lower, upper)
        return offsprings
    
    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 2
    
    def get_name(self) -> str:
        return "FloatBLXAlphaBetaCrossover"


class FloatNPointCrossover(Crossover[FloatSolution, FloatSolution]):
    """N-point crossover operator for float solutions.
    
    This is a generalized version of one-point and two-point crossover that
    allows any number of crossover points. It divides the variables into
    segments and alternates between parents for each segment.
    
    Args:
        probability (float): Probability of applying crossover.
        number_of_points (int): Number of crossover points. Must be at least 1
            and less than the number of variables.
    """
    
    def __init__(self, probability: float, number_of_points: int = 2):
        super(FloatNPointCrossover, self).__init__(probability=probability)
        self.number_of_points = number_of_points
        if number_of_points < 1:
            from evolu.core.exceptions import InvalidParameterException
            raise InvalidParameterException(
                f"The number of points must be at least 1: {number_of_points}"
            )
    
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Apply N-point crossover to two float solutions.
        
        Args:
            parents (List[FloatSolution]): Two parent solutions.
            
        Returns:
            List[FloatSolution]: Two offspring generated by N-point crossover.
        """
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            n_vars = parents[0].number_of_variables
            if n_vars > self.number_of_points:
                # Select crossover points
                points = sorted(random.sample(range(1, n_vars), min(self.number_of_points, n_vars - 1)))
                points = [0] + points + [n_vars]
                # Alternate between parents for each segment
                for seg in range(len(points) - 1):
                    start = points[seg]
                    end = points[seg + 1]
                    if seg % 2 == 0:
                        # First parent's segment
                        for i in range(start, end):
                            offsprings[0].variables[i] = parents[0].variables[i]
                            offsprings[1].variables[i] = parents[1].variables[i]
                    else:
                        # Second parent's segment
                        for i in range(start, end):
                            offsprings[0].variables[i] = parents[1].variables[i]
                            offsprings[1].variables[i] = parents[0].variables[i]
        return offsprings
    
    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 2
    
    def get_name(self) -> str:
        return "FloatNPointCrossover"


class FloatSinglePointPolynomialCrossover(Crossover[FloatSolution, FloatSolution]):
    """Single-Point Polynomial (SPX) crossover operator for float solutions.
    
    SPX is a variant of SBX (Simulated Binary Crossover) that uses a single crossover point and applies
    polynomial distribution only at that point, while other variables are
    directly copied from parents.
    
    References:
        Deb, K., & Agrawal, R. B. (1995). Simulated binary crossover for
        continuous search space. Complex systems, 9(2), 115-148.
    """
    __EPS = 1.0e-14
    
    def __init__(self, probability: float, distribution_index: float = 20.0):
        """Initialize Single-Point Polynomial crossover operator.
        
        Args:
            probability (float): Probability of applying crossover.
            distribution_index (float): Distribution index controlling the spread.
                Higher values produce offspring closer to parents. Defaults to 20.0.
        """
        super(FloatSinglePointPolynomialCrossover, self).__init__(probability=probability)
        self.distribution_index = distribution_index
        if distribution_index < 0:
            from evolu.core.exceptions import InvalidParameterException
            raise InvalidParameterException(
                f"The distribution index is negative: {distribution_index}"
            )
    
    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """Apply Single-Point Polynomial crossover to two float solutions.
        
        Args:
            parents (List[FloatSolution]): Two parent solutions.
            
        Returns:
            List[FloatSolution]: Two offspring generated by Single-Point Polynomial crossover.
        """
        Check.that(issubclass(type(parents[0]), FloatSolution), "Solution type invalid")
        Check.that(issubclass(type(parents[1]), FloatSolution), "Solution type invalid")
        Check.that(len(parents) == 2, "The number of parents is not two: {}".format(len(parents)))
        offsprings = copy.deepcopy(parents)
        rand = random.random()
        if rand <= self.probability:
            n_vars = parents[0].number_of_variables
            if n_vars > 1:
                # Select single crossover point
                crossover_point = random.randint(0, n_vars - 1)
                # Apply polynomial crossover at the crossover point
                i = crossover_point
                value_x1, value_x2 = parents[0].variables[i], parents[1].variables[i]
                if abs(value_x1 - value_x2) > self.__EPS:
                    if value_x1 < value_x2:
                        y1, y2 = value_x1, value_x2
                    else:
                        y1, y2 = value_x2, value_x1
                    lower_bound = parents[0].lower_bound[i]
                    upper_bound = parents[0].upper_bound[i]
                    beta = 1.0 + (2.0 * (y1 - lower_bound) / (y2 - y1))
                    alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))
                    rand = random.random()
                    if rand <= (1.0 / alpha):
                        betaq = pow(rand * alpha, (1.0 / (self.distribution_index + 1.0)))
                    else:
                        betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                    c1 = 0.5 * (y1 + y2 - betaq * (y2 - y1))
                    beta = 1.0 + (2.0 * (upper_bound - y2) / (y2 - y1))
                    alpha = 2.0 - pow(beta, -(self.distribution_index + 1.0))
                    if rand <= (1.0 / alpha):
                        betaq = pow((rand * alpha), (1.0 / (self.distribution_index + 1.0)))
                    else:
                        betaq = pow(1.0 / (2.0 - rand * alpha), 1.0 / (self.distribution_index + 1.0))
                    c2 = 0.5 * (y1 + y2 + betaq * (y2 - y1))
                    if c1 < lower_bound:
                        c1 = lower_bound
                    if c2 < lower_bound:
                        c2 = lower_bound
                    if c1 > upper_bound:
                        c1 = upper_bound
                    if c2 > upper_bound:
                        c2 = upper_bound
                    if random.random() <= 0.5:
                        offsprings[0].variables[i] = c2
                        offsprings[1].variables[i] = c1
                    else:
                        offsprings[0].variables[i] = c1
                        offsprings[1].variables[i] = c2
                # Copy other variables from parents
                for j in range(n_vars):
                    if j != crossover_point:
                        if random.random() <= 0.5:
                            offsprings[0].variables[j] = parents[0].variables[j]
                            offsprings[1].variables[j] = parents[1].variables[j]
                        else:
                            offsprings[0].variables[j] = parents[1].variables[j]
                            offsprings[1].variables[j] = parents[0].variables[j]
        return offsprings
    
    def get_number_of_parents(self) -> int:
        return 2
    
    def get_number_of_children(self) -> int:
        return 2
    
    def get_name(self) -> str:
        return "FloatSinglePointPolynomialCrossover"
