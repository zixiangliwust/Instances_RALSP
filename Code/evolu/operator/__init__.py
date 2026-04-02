"""Genetic operators module.

This module provides implementations of genetic operators used in evolutionary
algorithms:

- Crossover operators: Combine parent solutions to create offspring
- Mutation operators: Introduce diversity by modifying solutions
- Selection operators: Choose solutions for reproduction
- Replacement operators: Determine which solutions survive to next generation

All operators follow a common interface defined in the base Operator class,
allowing for flexible composition and customization of evolutionary algorithms.
"""
from .crossover import (
    NullCrossover,
    PermutationCycleCrossover,
    FloatDifferentialEvolutionCrossover,
    PermutationPartiallyMatchedCrossover,
    FloatSimulatedBinaryCrossover,
    BinaryArraySinglePointCrossover,
    FloatBLXAlphaCrossover,
    FloatBLXAlphaBetaCrossover,
    FloatNPointCrossover,
    FloatSinglePointPolynomialCrossover,
)
from .mutation import (
    BitFlipMutation,
    IntegerPolynomialMutation,
    NullMutation,
    FloatPermutationIntegerSwapMutation,
    FloatPolynomialMutation,
    FloatPermutationIntegerScrambleMutation,
    FloatSimpleRandomMutation,
    FloatUniformMutation,
)
from .replacement import (
    Replacement,
    GreedyPopulationReplacement,
    JoinPopulationSelectionReplacement,
    GreedyPopulationRankingAndDensityEstimatorReplacement,
    JoinPopulationRankingAndDensityEstimatorReplacement,
    JoinPopulationRankingAndDensityEstimatorReplacementRemoveDuplicatedSolution,
    RemovalPolicyType
)
from .selection import (
    BestSolutionSelection,
    BinaryTournament2Selection,
    BinaryTournamentSelection,
    NaryRandomSolutionSelection,
    RandomSolutionSelection,
    RankingAndCrowdingDistanceSelection,
    RouletteWheelSelection,
    DifferentialEvolutionSelection
)

__all__ = [
    "NullCrossover",
    "FloatSimulatedBinaryCrossover",
    "BinaryArraySinglePointCrossover",
    "FloatDifferentialEvolutionCrossover",
    "PermutationPartiallyMatchedCrossover",
    "PermutationCycleCrossover",
    "FloatBLXAlphaCrossover",
    "FloatBLXAlphaBetaCrossover",
    "FloatNPointCrossover",
    "FloatSinglePointPolynomialCrossover",
    "NullMutation",
    "BitFlipMutation",
    "FloatPolynomialMutation",
    "IntegerPolynomialMutation",
    "FloatUniformMutation",
    "FloatSimpleRandomMutation",
    "FloatPermutationIntegerScrambleMutation",
    "FloatPermutationIntegerSwapMutation",
    "RouletteWheelSelection",
    "BestSolutionSelection",
    "BinaryTournamentSelection",
    "BinaryTournament2Selection",
    "RandomSolutionSelection",
    "NaryRandomSolutionSelection",
    "RankingAndCrowdingDistanceSelection",
    "RouletteWheelSelection",
    "DifferentialEvolutionSelection",
    "Replacement",
    "GreedyPopulationReplacement",
    "JoinPopulationSelectionReplacement",
    "GreedyPopulationRankingAndDensityEstimatorReplacement",
    "JoinPopulationRankingAndDensityEstimatorReplacement",
    "JoinPopulationRankingAndDensityEstimatorReplacementRemoveDuplicatedSolution",
    "RemovalPolicyType",
]
