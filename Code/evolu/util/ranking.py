# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from evolu.core.solution import Solution
from evolu.util.comparator import (Comparator, DominanceComparator, SolutionAttributeComparator, )

S = TypeVar("S", bound=Solution)


class Ranking(Generic[S], ABC):
    """Base class for ranking solutions.
    
    Ranking algorithms assign solutions to different fronts or ranks based on
    their relative quality. This is commonly used in multi-objective optimization
    to identify non-dominated fronts.
    
    Attributes:
        number_of_comparisons (int): Counter for the number of comparisons performed.
        ranked_sub_lists (List[List[S]]): List of fronts, where each front contains
            solutions at that rank level.
        comparator (Comparator[S]): Comparator used for ranking solutions.
    
    Note:
        Subclasses must implement compute_ranking() and get_comparator() methods.
    """
    
    def __init__(self, comparator: Comparator[S] = DominanceComparator()) -> None:
        """Initialize ranking with a comparator.
        
        Args:
            comparator (Comparator[S], optional): Comparator to use for ranking.
                Defaults to DominanceComparator().
        """
        super(Ranking, self).__init__()
        self.number_of_comparisons = 0
        self.ranked_sub_lists: List[List[S]] = []
        self.comparator = comparator

    @abstractmethod
    def compute_ranking(self, solutions: List[S], k: Optional[int] = None) -> None:
        """Compute ranking of solutions.
        
        Args:
            solutions (List[S]): List of solutions to rank.
            k (Optional[int], optional): Number of fronts to compute. If None,
                computes all fronts. Defaults to None.
        
        Note:
            After calling this method, ranked_sub_lists will contain the ranked
            fronts, with rank 0 being the best (non-dominated) front.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `compute_ranking`.")

    def get_nondominated(self) -> List[S]:
        """Get the non-dominated solutions (first front).
        
        Returns:
            List[S]: List of solutions in the first (best) front.
        """
        return self.ranked_sub_lists[0]

    def get_sub_front(self, rank: int) -> List[S]:
        """Get solutions at the specified rank.
        
        Args:
            rank (int): The rank (front index) to retrieve. Rank 0 is the best front.
        
        Returns:
            List[S]: List of solutions at the specified rank.
        
        Raises:
            InvalidRankException: If rank is out of bounds.
        """
        if rank >= len(self.ranked_sub_lists):
            from evolu.core.exceptions import InvalidRankException
            raise InvalidRankException(
                f"Invalid rank: {rank}. Max rank: {len(self.ranked_sub_lists) - 1}"
            )
        return self.ranked_sub_lists[rank]

    def get_number_of_sub_fronts(self) -> int:
        """Get the number of sub-fronts.
        
        Returns:
            int: Number of ranked fronts computed.
        """
        return len(self.ranked_sub_lists)

    @classmethod
    @abstractmethod
    def get_comparator(cls) -> Comparator:
        """Get the comparator used for ranking.
        
        Returns:
            Comparator: The default comparator for this ranking method.
        """
        raise NotImplementedError(f"{cls.__name__} must implement `get_comparator`.")


class FastNonDominatedRanking(Ranking[S]):
    """Fast non-dominated ranking algorithm from NSGA-II.
    
    This class implements the efficient non-dominated sorting algorithm proposed
    by Deb et al. in NSGA-II. It assigns solutions to different Pareto fronts
    based on dominance relationships.
    
    Reference:
        Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and
        elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on
        evolutionary computation, 6(2), 182-197.
    """

    def __init__(self, comparator: Comparator[S] = DominanceComparator()) -> None:
        """Initialize fast non-dominated ranking.
        
        Args:
            comparator (Comparator[S], optional): Comparator to use for dominance
                checking. Defaults to DominanceComparator().
        """
        super(FastNonDominatedRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: Optional[int] = None) -> None:
        """Compute fast non-dominated ranking of solutions.
        
        This method efficiently assigns solutions to non-dominated fronts using
        the NSGA-II algorithm. Solutions in the same front are mutually non-dominated.
        
        Args:
            solutions (List[S]): List of solutions to rank.
            k (Optional[int], optional): Maximum number of fronts to compute.
                If None, computes all fronts. Defaults to None.
        
        Note:
            After execution, each solution will have a 'dominance_ranking' attribute
            set in its attributes dictionary, and ranked_sub_lists will contain
            the sorted fronts.
        """
        # number of solutions dominating solution ith
        dominating_ith = [0 for _ in range(len(solutions))]
        # list of solutions dominated by solution ith
        ith_dominated = [[] for _ in range(len(solutions))]
        # front[i] contains the list of solutions belonging to front i
        front = [[] for _ in range(len(solutions) + 1)]
        for p in range(len(solutions) - 1):
            for q in range(p + 1, len(solutions)):
                dominance_test_result = self.comparator.compare(solutions[p], solutions[q])
                self.number_of_comparisons += 1
                if dominance_test_result == -1:
                    ith_dominated[p].append(q)
                    dominating_ith[q] += 1
                elif dominance_test_result == 1:
                    ith_dominated[q].append(p)
                    dominating_ith[p] += 1
        for i in range(len(solutions)):
            if dominating_ith[i] == 0:
                front[0].append(i)
                solutions[i].attributes["dominance_ranking"] = 0
        rank = 0
        while len(front[rank]) != 0:
            rank += 1
            for p in front[rank - 1]:
                for q in ith_dominated[p]:
                    dominating_ith[q] -= 1
                    if dominating_ith[q] == 0:
                        front[rank].append(q)
                        solutions[q].attributes["dominance_ranking"] = rank
        self.ranked_sub_lists = [[]] * rank
        for j in range(rank):
            q = [0] * len(front[j])
            for m in range(len(front[j])):
                q[m] = solutions[front[j][m]]
            self.ranked_sub_lists[j] = q
        if k:
            count = 0
            for i, front in enumerate(self.ranked_sub_lists):
                count += len(front)
                if count >= k:
                    self.ranked_sub_lists = self.ranked_sub_lists[: i + 1]
                    break

    @classmethod
    def get_comparator(cls) -> Comparator:
        """Get comparator for fast non-dominated ranking.
        
        Returns:
            Comparator: SolutionAttributeComparator that compares solutions
                based on "dominance_ranking" attribute (lower rank values
                indicate better fronts, rank 0 is the non-dominated front).
        """
        return SolutionAttributeComparator("dominance_ranking")


class StrengthRanking(Ranking[S]):
    """Ranking scheme based on strength value, used in SPEA2 algorithm.
    
    Strength ranking assigns ranks to solutions based on how many solutions
    they dominate (strength value) and how many solutions dominate them
    (raw fitness). Solutions with lower raw fitness values get better ranks.
    
    The strength of a solution is the number of solutions it dominates.
    The raw fitness of a solution is the sum of strengths of all solutions
    that dominate it. Solutions are ranked by raw fitness (lower is better).
    
    Reference:
        Zitzler, E., Laumanns, M., & Thiele, L. (2001). SPEA2: Improving the
        strength Pareto evolutionary algorithm. TIK-report, 103.
    
    Attributes:
        comparator (Comparator[S]): Comparator used for dominance checking.
            Defaults to DominanceComparator.
    """

    def __init__(self, comparator: Comparator[S] = DominanceComparator()) -> None:
        """Initialize strength ranking.
        
        Args:
            comparator (Comparator[S], optional): Comparator for dominance checking.
                Defaults to DominanceComparator.
        """
        super(StrengthRanking, self).__init__(comparator)

    def compute_ranking(self, solutions: List[S], k: Optional[int] = None) -> None:
        """Compute strength-based ranking of solutions.
        
        Calculates strength and raw fitness for each solution, then assigns
        ranks. Solutions with the same raw fitness are in the same rank.
        Stores ranking information in solution attributes.
        
        Args:
            solutions (List[S]): List of solutions to rank.
            k (Optional[int]): Number of individuals (not used in this implementation).
                Defaults to None.
        
        Note:
            - Strength: Number of solutions dominated by this solution
            - Raw fitness: Sum of strengths of solutions dominating this solution
            - Lower raw fitness = better rank
            - Ranking is stored in solution.attributes["strength_ranking"]
            - Ranked sublists are stored in self.ranked_sub_lists
        """
        strength: List[int] = [0 for _ in range(len(solutions))]
        raw_fitness: List[int] = [0 for _ in range(len(solutions))]
        # strength(i) = | {j | j < - SolutionSet and i dominate j} |
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) < 0:
                    strength[i] += 1
        # Calculate the raw fitness:
        # rawFitness(i) = |{sum strength(j) | j <- SolutionSet and j dominate i}|
        for i in range(len(solutions)):
            for j in range(len(solutions)):
                if self.comparator.compare(solutions[i], solutions[j]) == 1:
                    raw_fitness[i] += strength[j]
        max_fitness_value: int = 0
        for i in range(len(solutions)):
            solutions[i].attributes["strength_ranking"] = raw_fitness[i]
            if raw_fitness[i] > max_fitness_value:
                max_fitness_value = raw_fitness[i]
        # Initialize the ranked sublists. In the worst case will be max_fitness_value + 1 different sublists
        self.ranked_sub_lists = [[] for _ in range(max_fitness_value + 1)]
        # Assign each solution to its corresponding front
        for solution in solutions:
            self.ranked_sub_lists[int(solution.attributes["strength_ranking"])].append(solution)
        # Remove empty fronts
        counter = 0
        while counter < len(self.ranked_sub_lists):
            if len(self.ranked_sub_lists[counter]) == 0:
                del self.ranked_sub_lists[counter]
            else:
                counter += 1

    @classmethod
    def get_comparator(cls) -> Comparator:
        """Get comparator for strength ranking.
        
        Returns:
            Comparator: SolutionAttributeComparator that compares solutions
                based on "strength_ranking" attribute (lower raw fitness
                values indicate better ranks).
        """
        return SolutionAttributeComparator("strength_ranking")
