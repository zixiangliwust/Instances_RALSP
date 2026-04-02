# -*- coding: utf-8 -*-
from enum import Enum
from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar
from evolu.core.solution import Solution
from evolu.util.comparator import (Comparator, EqualSolutionsComparator, MultiComparator,
                                         DominanceWithConstraintsComparator)
from evolu.util.ranking import Ranking, FastNonDominatedRanking
from evolu.util.density_estimator import DensityEstimator, CrowdingDistance

S = TypeVar("S", bound=Solution)


class Replacement(Generic[S], ABC):
    """Base class for replacement operators.
    
    Replacement operators determine how the population is updated after generating
    offspring. They decide which solutions (parents or offspring) should survive
    to the next generation, implementing various survival strategies used in
    evolutionary algorithms.
    
    Note:
        Subclasses must implement the replace() method to define the replacement strategy.
    """
    
    def __init__(self) -> None:
        """Initialize replacement operator."""
        pass

    @abstractmethod
    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions in the population with offspring.
        
        Args:
            solution_list (List[S]): Current population (parent solutions).
            offspring_list (List[S]): Newly generated offspring solutions.
        
        Returns:
            List[S]: Updated population after replacement. Typically has the same
                size as solution_list.
        
        Note:
            Different replacement strategies may use different criteria:
            - Greedy: Compare each parent-offspring pair
            - Ranking-based: Use dominance ranking and density estimation
            - Join-and-select: Combine populations and select best solutions
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `replace`.")

    def get_name(self) -> str:
        """Get the name of the replacement operator.
        
        Returns:
            str: Name of the replacement operator.
        """
        return "Replacement"


class GreedyPopulationReplacement(Replacement[S]):
    """Greedy population replacement operator.
    
    This replacement operator performs pairwise comparison between each parent
    and its corresponding offspring. The better solution (according to the comparator)
    is kept in the population.
    
    For each position i:
    - If parent[i] is better than offspring[i]: keep parent[i]
    - Otherwise: replace with offspring[i]
    
    This is a simple and efficient replacement strategy that maintains population
    size by one-to-one replacement.
    
    Attributes:
        comparator (Comparator[S]): Comparator used to determine which solution
            is better. Defaults to DominanceWithConstraintsComparator().
    
    Note:
        Requires solution_list and offspring_list to have the same length.
    """
    
    def __init__(self, comparator: Comparator[S] = DominanceWithConstraintsComparator()) -> None:
        """Initialize greedy population replacement operator.
        
        Args:
            comparator (Comparator[S], optional): Comparator for comparing solutions.
                Defaults to DominanceWithConstraintsComparator().
        """
        super(GreedyPopulationReplacement, self).__init__()
        self.comparator = comparator

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions using greedy pairwise comparison.
        
        Args:
            solution_list (List[S]): Current population.
            offspring_list (List[S]): Offspring solutions (must have same length
                as solution_list).
        
        Returns:
            List[S]: Updated population with better solutions kept at each position.
        """
        result_list = []
        for i in range(0, len(solution_list)):
            if self.comparator.compare(solution_list[i], offspring_list[i]) == -1:
                result_list.append(solution_list[i])
            else:
                result_list.append(offspring_list[i])
        return result_list

    def get_name(self) -> str:
        return "Population greedy replacement"


class JoinPopulationSelectionReplacement(Replacement[S]):
    """Join population selection replacement operator.
    
    This replacement operator combines the parent and offspring populations,
    then sorts the combined population using a simple bubble-sort-like algorithm
    based on the comparator. The best solutions from the sorted combined
    population are selected to form the new population.
    
    The sorting is done in-place using a comparison-based approach where
    solutions are swapped to bring better solutions to the front of the list.
    The first N solutions (where N is the original population size) are
    then selected as the new population.
    
    Attributes:
        comparator (Comparator[S]): Comparator used to determine solution quality
            and order. Defaults to DominanceWithConstraintsComparator().
    
    Note:
        This is a simple but potentially inefficient replacement strategy for
        large populations due to the O(n²) sorting algorithm. For better
        performance with large populations, consider using ranking-based
        replacement operators.
    """
    
    def __init__(self, comparator: Comparator[S] = DominanceWithConstraintsComparator()) -> None:
        """Initialize join population selection replacement operator.
        
        Args:
            comparator (Comparator[S], optional): Comparator for comparing solutions.
                Defaults to DominanceWithConstraintsComparator().
        """
        super(JoinPopulationSelectionReplacement, self).__init__()
        self.comparator = comparator

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions by joining populations and selecting the best.
        
        Combines parent and offspring populations, sorts them using the comparator,
        and returns the best solutions up to the original population size.
        
        Args:
            solution_list (List[S]): Current population (parent solutions).
            offspring_list (List[S]): Newly generated offspring solutions.
        
        Returns:
            List[S]: New population consisting of the best solutions from the
                combined parent and offspring populations.
        """
        join_population = solution_list + offspring_list
        for i in range(0, len(join_population) - 1):
            for j in range(i + 1, len(join_population)):
                if self.comparator.compare(join_population[i], join_population[j]) == 1:
                    join_population[i], join_population[j] = join_population[j], join_population[i]
        result_list = join_population[0:len(solution_list)]
        return result_list

    def get_name(self) -> str:
        """Get name of the replacement operator.
        
        Returns:
            str: "Join population selection replacement"
        """
        return "Join population selection replacement"


class GreedyPopulationRankingAndDensityEstimatorReplacement(Replacement[S]):
    """Greedy population replacement using ranking and density estimator.
    
    This replacement operator combines ranking-based evaluation with greedy
    pairwise comparison. It first computes the ranking and density estimation
    on the combined population, but then uses a greedy pairwise comparison
    strategy similar to GreedyPopulationReplacement to select which solutions
    to keep.
    
    The operator uses a MultiComparator that considers both ranking and density
    information when comparing solutions. However, the final selection is done
    through pairwise comparison between parents and offspring at corresponding
    positions.
    
    Attributes:
        ranking (Ranking[S]): Ranking algorithm used to assign ranks to solutions.
            Defaults to FastNonDominatedRanking.
        density_estimator (DensityEstimator): Density estimator used to measure
            solution diversity. Defaults to CrowdingDistance.
        comparator (MultiComparator): Multi-level comparator that uses both ranking
            and density information to compare solutions.
    
    Note:
        This operator appears to compute ranking and density but then uses a
        simpler greedy strategy for the actual replacement. This may not fully
        utilize the ranking and density information computed.
    """
    
    def __init__(self,
                 ranking: Ranking[S] = FastNonDominatedRanking(DominanceWithConstraintsComparator()),
                 density_estimator: DensityEstimator = CrowdingDistance(),
                 ) -> None:
        """Initialize greedy population ranking and density estimator replacement operator.
        
        Args:
            ranking (Ranking[S], optional): Ranking algorithm for solution ranking.
                Defaults to FastNonDominatedRanking with DominanceWithConstraintsComparator.
            density_estimator (DensityEstimator, optional): Density estimator for diversity
                measurement. Defaults to CrowdingDistance.
        """
        super(GreedyPopulationRankingAndDensityEstimatorReplacement, self).__init__()
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.comparator = MultiComparator([self.ranking.get_comparator(), self.density_estimator.get_comparator()])

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions using greedy pairwise comparison with ranking context.
        
        Computes ranking and density estimation on the combined population,
        then performs greedy pairwise comparison between parents and offspring.
        
        Args:
            solution_list (List[S]): Current population (parent solutions).
            offspring_list (List[S]): Newly generated offspring solutions.
        
        Returns:
            List[S]: Updated population after replacement, maintaining original size.
        """
        join_population = solution_list + offspring_list
        size_of_the_result_list = len(solution_list) + len(offspring_list)
        self.ranking.compute_ranking(join_population)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) <= (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
        result_list = []
        for i in range(0, len(solution_list)):
            if self.comparator.compare(solution_list[i], offspring_list[i]) == -1:
                result_list.append(solution_list[i])
            else:
                result_list.append(offspring_list[i])
        return result_list

    def get_name(self) -> str:
        """Get name of the replacement operator.
        
        Returns:
            str: "Greedy population ranking and density estimator replacement"
        """
        return "Greedy population ranking and density estimator replacement"


class RemovalPolicyType(Enum):
    """Enumeration of removal policies for population truncation.
    
    Defines different strategies for removing solutions when a population
    exceeds the desired size during truncation operations.
    
    Attributes:
        SEQUENTIAL (int): Sequential removal policy. Solutions are removed one
            at a time, and density estimation is recomputed after each removal.
            This provides more accurate density information but is computationally
            more expensive.
        ONE_SHOT (int): One-shot removal policy. Solutions are sorted by density
            and the worst ones are removed all at once. This is faster but may
            be less accurate in maintaining diversity.
    
    Note:
        - SEQUENTIAL: Better diversity preservation but slower
        - ONE_SHOT: Faster but potentially less accurate diversity management
    """
    SEQUENTIAL = 1
    ONE_SHOT = 2


class JoinPopulationRankingAndDensityEstimatorReplacement(Replacement[S]):
    """Join population replacement using ranking and density estimator.
    
    This replacement operator implements the standard multi-objective replacement
    strategy used in algorithms like NSGA-II. It combines parent and offspring
    populations, ranks them by dominance, and selects the best solutions while
    maintaining diversity through density estimation.
    
    The operator supports two removal policies:
    - SEQUENTIAL: Removes solutions one at a time, recomputing density after each
      removal (more accurate but slower)
    - ONE_SHOT: Sorts solutions by density and removes the worst ones all at once
      (faster but potentially less accurate)
    
    Attributes:
        ranking (Ranking[S]): Ranking algorithm used to assign dominance ranks.
            Defaults to FastNonDominatedRanking.
        density_estimator (DensityEstimator): Density estimator for maintaining
            diversity. Defaults to CrowdingDistance.
        removal_policy (RemovalPolicyType): Policy for removing solutions when
            truncating to desired population size. Defaults to ONE_SHOT.
    
    References:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002).
        A fast and elitist multiobjective genetic algorithm: NSGA-II.
        IEEE transactions on evolutionary computation, 6(2), 182-197.
    """
    
    def __init__(self,
                 ranking: Ranking[S] = FastNonDominatedRanking(DominanceWithConstraintsComparator()),
                 density_estimator: DensityEstimator = CrowdingDistance(),
                 removal_policy: RemovalPolicyType = RemovalPolicyType.ONE_SHOT
                 ) -> None:
        """Initialize join population ranking and density estimator replacement operator.
        
        Args:
            ranking (Ranking[S], optional): Ranking algorithm for dominance ranking.
                Defaults to FastNonDominatedRanking with DominanceWithConstraintsComparator.
            density_estimator (DensityEstimator, optional): Density estimator for diversity.
                Defaults to CrowdingDistance.
            removal_policy (RemovalPolicyType, optional): Removal policy for truncation.
                Defaults to ONE_SHOT.
        """
        super(JoinPopulationRankingAndDensityEstimatorReplacement, self).__init__()
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.removal_policy = removal_policy

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions using ranking and density-based selection.
        
        Combines parent and offspring populations, computes dominance ranking,
        and selects the best solutions while maintaining diversity. The selection
        continues until the desired population size is reached, using the
        specified removal policy for truncation.
        
        Args:
            solution_list (List[S]): Current population (parent solutions).
            offspring_list (List[S]): Newly generated offspring solutions.
        
        Returns:
            List[S]: New population of the same size as solution_list, selected
                from the combined parent and offspring populations based on
                ranking and density.
        """
        join_population = solution_list + offspring_list
        if self.removal_policy is RemovalPolicyType.SEQUENTIAL:
            result_list = self.sequential_truncation_ranks(join_population, len(solution_list))
            # self.ranking.compute_ranking(join_population)
            # result_list = self.sequential_truncation(0, len(solution_list))
        else:
            result_list = self.one_shot_truncation_ranks(join_population, len(solution_list))
            # self.ranking.compute_ranking(join_population)
            # result_list = self.one_shot_truncation(0, len(solution_list))
        return result_list

    def sequential_truncation_ranks(self, front: List[S], size_of_the_result_list: int) -> List[S]:
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) < (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                last_ranked_solutions = []
                for solution in current_ranked_solutions:
                    last_ranked_solutions.append(solution)
                while len(last_ranked_solutions) > (size_of_the_result_list - len(result_list)):
                    self.density_estimator.sort(last_ranked_solutions)
                    del last_ranked_solutions[-1]
                    self.density_estimator.compute_density_estimator(last_ranked_solutions)
                result_list = result_list + last_ranked_solutions
        return result_list

    def one_shot_truncation_ranks(self, front: List[S], size_of_the_result_list: int) -> List[S]:
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) < (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                self.density_estimator.sort(current_ranked_solutions)
                for i in range(size_of_the_result_list - len(result_list)):
                    result_list.append(current_ranked_solutions[i])
        return result_list

    def sequential_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)
        result_list: List[S] = []
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_sub_front(ranking_id))
            result_list.extend(
                self.sequential_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            for solution in current_ranked_solutions:
                result_list.append(solution)
            while len(result_list) > size_of_the_result_list:
                self.density_estimator.sort(result_list)
                del result_list[-1]
                self.density_estimator.compute_density_estimator(result_list)
        return result_list

    def one_shot_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        """Recursively truncate population using one-shot removal (legacy method).
        
        This is a recursive implementation of one-shot truncation that processes
        ranks one at a time. It may be less efficient than one_shot_truncation_ranks
        due to recursive overhead.
        
        Args:
            ranking_id (int): Current rank being processed.
            size_of_the_result_list (int): Desired population size after truncation.
        
        Returns:
            List[S]: Truncated population of the specified size.
        
        Note:
            This method appears to be a legacy implementation. Consider using
            one_shot_truncation_ranks instead for better performance.
        """
        current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)
        result_list: List[S] = []
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_sub_front(ranking_id))
            result_list.extend(
                self.one_shot_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            self.density_estimator.sort(current_ranked_solutions)
            i = 0
            while len(result_list) < size_of_the_result_list:
                result_list.append(current_ranked_solutions[i])
                i += 1
        return result_list

    def get_name(self) -> str:
        """Get name of the replacement operator.
        
        Returns:
            str: "Join population ranking and density estimator replacement"
        """
        return "Join population ranking and density estimator replacement"


class JoinPopulationRankingAndDensityEstimatorReplacementRemoveDuplicatedSolution(Replacement[S]):
    """Join population replacement with duplicate solution removal.
    
    This replacement operator extends JoinPopulationRankingAndDensityEstimatorReplacement
    by adding duplicate solution detection and removal before ranking and selection.
    
    Duplicate solutions are identified using EqualSolutionsComparator, and when found,
    one of the duplicates is penalized (by setting all objectives to a large value)
    to effectively remove it from consideration during ranking and selection.
    
    This is useful in algorithms where duplicate solutions can accumulate and reduce
    population diversity.
    
    Attributes:
        ranking (Ranking[S]): Ranking algorithm used to assign dominance ranks.
            Defaults to FastNonDominatedRanking.
        density_estimator (DensityEstimator): Density estimator for maintaining
            diversity. Defaults to CrowdingDistance.
        removal_policy (RemovalPolicyType): Policy for removing solutions when
            truncating to desired population size. Defaults to ONE_SHOT.
    
    Note:
        Duplicate detection is performed before ranking, which means duplicates
        are removed from the combined population before selection begins.
    """
    
    def __init__(self,
                 ranking: Ranking[S] = FastNonDominatedRanking(DominanceWithConstraintsComparator()),
                 density_estimator: DensityEstimator = CrowdingDistance(),
                 removal_policy: RemovalPolicyType = RemovalPolicyType.ONE_SHOT
                 ) -> None:
        """Initialize join population ranking and density estimator replacement operator with duplicate removal.
        
        Args:
            ranking (Ranking[S], optional): Ranking algorithm for dominance ranking.
                Defaults to FastNonDominatedRanking with DominanceWithConstraintsComparator.
            density_estimator (DensityEstimator, optional): Density estimator for diversity.
                Defaults to CrowdingDistance.
            removal_policy (RemovalPolicyType, optional): Removal policy for truncation.
                Defaults to ONE_SHOT.
        """
        super(JoinPopulationRankingAndDensityEstimatorReplacementRemoveDuplicatedSolution, self).__init__()
        self.ranking = ranking
        self.density_estimator = density_estimator
        self.removal_policy = removal_policy

    def replace(self, solution_list: List[S], offspring_list: List[S]) -> List[S]:
        """Replace solutions with duplicate removal and ranking-based selection.
        
        First removes duplicate solutions from the combined population by penalizing
        them, then applies ranking and density-based selection using the specified
        removal policy.
        
        Args:
            solution_list (List[S]): Current population (parent solutions).
            offspring_list (List[S]): Newly generated offspring solutions.
        
        Returns:
            List[S]: New population of the same size as solution_list, selected
                from the combined parent and offspring populations after duplicate
                removal, based on ranking and density.
        """
        join_population = solution_list + offspring_list
        equal_solutions_comparator = EqualSolutionsComparator()
        for j in range(0, len(join_population) - 1):
            for k in range(j + 1, len(join_population)):
                if equal_solutions_comparator.compare(join_population[j], join_population[k]) == 0:
                    for i in range(0, join_population[k].number_of_objectives):
                        join_population[k].objectives[i] = 1.0e30
        if self.removal_policy is RemovalPolicyType.SEQUENTIAL:
            result_list = self.sequential_truncation_ranks(join_population, len(solution_list))
        else:
            result_list = self.one_shot_truncation_ranks(join_population, len(solution_list))
        return result_list

    def sequential_truncation_ranks(self, front: List[S], size_of_the_result_list: int) -> List[S]:
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) < (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                last_ranked_solutions = []
                for solution in current_ranked_solutions:
                    last_ranked_solutions.append(solution)
                while len(last_ranked_solutions) > (size_of_the_result_list - len(result_list)):
                    self.density_estimator.sort(last_ranked_solutions)
                    del last_ranked_solutions[-1]
                    self.density_estimator.compute_density_estimator(last_ranked_solutions)
                result_list = result_list + last_ranked_solutions
        return result_list

    def one_shot_truncation_ranks(self, front: List[S], size_of_the_result_list: int) -> List[S]:
        self.ranking.compute_ranking(front)
        ranking_id = 0
        result_list = []
        while len(result_list) < size_of_the_result_list:
            current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
            self.density_estimator.compute_density_estimator(current_ranked_solutions)
            if len(current_ranked_solutions) < (size_of_the_result_list - len(result_list)):
                result_list = result_list + current_ranked_solutions
                ranking_id += 1
            else:
                self.density_estimator.sort(current_ranked_solutions)
                for i in range(size_of_the_result_list - len(result_list)):
                    result_list.append(current_ranked_solutions[i])
        return result_list

    def sequential_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)
        result_list: List[S] = []
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_sub_front(ranking_id))
            result_list.extend(
                self.sequential_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            for solution in current_ranked_solutions:
                result_list.append(solution)
            while len(result_list) > size_of_the_result_list:
                self.density_estimator.sort(result_list)
                del result_list[-1]
                self.density_estimator.compute_density_estimator(result_list)
        return result_list

    def one_shot_truncation(self, ranking_id: int, size_of_the_result_list: int) -> List[S]:
        current_ranked_solutions = self.ranking.get_sub_front(ranking_id)
        self.density_estimator.compute_density_estimator(current_ranked_solutions)
        result_list: List[S] = []
        if len(current_ranked_solutions) < size_of_the_result_list:
            result_list.extend(self.ranking.get_sub_front(ranking_id))
            result_list.extend(
                self.one_shot_truncation(ranking_id + 1, size_of_the_result_list - len(current_ranked_solutions))
            )
        else:
            self.density_estimator.sort(current_ranked_solutions)
            i = 0
            while len(result_list) < size_of_the_result_list:
                result_list.append(current_ranked_solutions[i])
                i += 1
        return result_list

    def get_name(self) -> str:
        return "Join population ranking and density estimator replacement"
