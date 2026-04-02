import copy
from typing import List, TypeVar
from evolu.core.operator import Selection
from evolu.core.solution import Solution
from evolu.core.exceptions import EmptyFrontException
from evolu.util.comparator import Comparator, DominanceWithConstraintsComparator

S = TypeVar("S", bound=Solution)
"""
module:: Sort population
synopsis: Module implementing selection operators.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class SortPopulation(Selection[List[S], List[S]]):
    """Population sorting operator.
    
    This operator sorts a population of solutions based on a comparator. It
    implements a simple bubble sort algorithm to order solutions according to
    their quality as determined by the comparator.
    
    The sorted population will have better solutions at the beginning (lower indices)
    and worse solutions at the end (higher indices), based on the comparator's
    ordering.
    
    Attributes:
        comparator (Comparator[S]): Comparator used to determine solution ordering.
            Defaults to DominanceWithConstraintsComparator().
    
    Note:
        This uses O(n²) bubble sort. For large populations, consider using
        Python's built-in sorted() with a key function for better performance.
    """
    
    def __init__(self, comparator: Comparator[S] = DominanceWithConstraintsComparator()) -> None:
        """Initialize sort population operator with a comparator.
        
        Args:
            comparator (Comparator[S], optional): Comparator for ordering solutions.
                Defaults to DominanceWithConstraintsComparator().
        """
        super(SortPopulation, self).__init__()
        self.comparator = comparator

    def execute(self, front: List[S]) -> List[S]:
        """Sort the population of solutions.
        
        Args:
            front (List[S]): List of solutions to sort.
        
        Returns:
            List[S]: Sorted copy of the input list. Better solutions (according
                to the comparator) appear first in the list.
        
        Raises:
            EmptyFrontException: If front is None or empty.
        
        Note:
            The input list is not modified. A sorted copy is returned.
        """
        if front is None:
            raise EmptyFrontException("The front is null")
        elif len(front) == 0:
            raise EmptyFrontException("The front is empty")
        sorted_population = copy.deepcopy(front)
        for i in range(0, len(front) - 1):
            for j in range(i + 1, len(front)):
                if self.comparator.compare(sorted_population[i], sorted_population[j]) == 1:
                    solution = sorted_population[i]
                    sorted_population[i] = sorted_population[j]
                    sorted_population[j] = solution
        return sorted_population

    def get_name(self) -> str:
        """Get the name of the operator.
        
        Returns:
            str: "Sort population"
        """
        return "Sort population"
