# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import List, Union
import numpy
from scipy.spatial import distance

"""
module:: distance
synopsis: implementation of distances between entities
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Distance(ABC):
    """Base class for distance metrics.
    
    Distance metrics measure the similarity or dissimilarity between two points
    or vectors in a multi-dimensional space. Different distance metrics are used
    for various purposes in evolutionary computation, such as diversity measures,
    neighborhood definitions, and similarity calculations.
    
    Note:
        Subclasses must implement get_distance() method.
    """
    
    @abstractmethod
    def get_distance(self, element1: Union[List[float], List[int]], element2: Union[List[float], List[int]]) -> float:
        """Calculate distance between two elements.
        
        Args:
            element1 (Union[List[float], List[int]]): First element (point/vector).
            element2 (Union[List[float], List[int]]): Second element (point/vector).
        
        Returns:
            float: Distance value between the two elements. Always non-negative.
                Lower values indicate closer/similar elements.
        
        Note:
            Both elements must have the same dimensionality.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_distance`.")


class EuclideanDistance(Distance):
    """Euclidean (L2) distance metric.
    
    Euclidean distance is the straight-line distance between two points in
    Euclidean space. It is computed as:
        d = sqrt(Σ(x_i - y_i)²)
    where x_i and y_i are the i-th components of the two vectors.
    
    This is the most commonly used distance metric in evolutionary computation
    for measuring solution diversity and similarity.
    """
    
    def get_distance(self, list1: Union[List[float], List[int]], list2: Union[List[float], List[int]]) -> float:
        """Calculate Euclidean distance between two lists.
        
        Args:
            list1 (Union[List[float], List[int]]): First vector.
            list2 (Union[List[float], List[int]]): Second vector.
        
        Returns:
            float: Euclidean distance between the two vectors.
        
        Note:
            Both lists must have the same length.
        """
        return distance.euclidean(list1, list2)


class CosineDistance(Distance):
    """Cosine distance metric relative to a reference point.
    
    Cosine distance measures the angular similarity between two vectors when
    both are translated relative to a reference point. It's based on the cosine
    of the angle between the vectors, providing a scale-invariant measure of
    direction similarity.
    
    Attributes:
        reference_point (Union[List[float], List[int]]): Reference point used
            as the origin for computing cosine distances.
    
    Note:
        This implementation computes a modified cosine distance that accounts
        for the reference point, useful in multi-objective optimization where
        solutions are compared relative to an ideal or nadir point.
    """
    
    def __init__(self, reference_point: Union[List[float], List[int]]) -> None:
        """Initialize with a reference point.
        
        Args:
            reference_point (Union[List[float], List[int]]): Reference point
                to use as the origin for distance calculations.
        """
        self.reference_point = reference_point

    def get_distance(self, list1: Union[List[float], List[int]], list2: Union[List[float], List[int]]) -> float:
        """Calculate cosine distance between two lists relative to reference point.
        
        The distance is computed by:
        1. Translating both vectors relative to the reference point
        2. Computing the cosine similarity between the translated vectors
        3. Returning a normalized distance measure
        
        Args:
            list1 (Union[List[float], List[int]]): First vector.
            list2 (Union[List[float], List[int]]): Second vector.
        
        Returns:
            float: Cosine distance value between the two vectors relative to
                the reference point. Lower values indicate more similar directions.
        """
        total = sum(
            numpy.multiply(
                [(x - r) for x, r in zip(list1, self.reference_point)],
                [(y - r) for y, r in zip(list2, self.reference_point)],
            )
        )
        a = distance.cosine(
            [x - y for x, y in zip(list1, self.reference_point)], [x - y for x, y in zip(list2, self.reference_point)]
        )
        b = total / (
                self.__sum_of_distances_to_reference_point(list1) * self.__sum_of_distances_to_reference_point(list2)
        )
        return b

    def __sum_of_distances_to_reference_point(self, l: Union[List[float], List[int]]) -> float:
        """Calculate sum of squared distances to reference point."""
        return sum([pow(x - y, 2.0) for x, y in zip(l, self.reference_point)])
