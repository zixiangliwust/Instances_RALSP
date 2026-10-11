# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Iterable, List, Optional

import numpy as np
from scipy import spatial

from evolu.core.exceptions import ReferenceFrontNotSetException


class QualityIndicator(ABC):
    """Base class for quality indicators.
    
    Quality indicators measure the quality of a set of solutions, typically used
    to evaluate the performance of multi-objective optimization algorithms.
    Common indicators include generational distance, inverted generational distance,
    hypervolume, and spread metrics.
    
    Attributes:
        is_minimization (bool): Whether lower values indicate better quality.
            True for indicators where smaller is better (e.g., GD, IGD),
            False for indicators where larger is better (e.g., Hypervolume).
    
    Note:
        Subclasses must implement compute(), get_name(), and get_short_name() methods.
    """
    
    def __init__(self, is_minimization: bool) -> None:
        """Initialize quality indicator.
        
        Args:
            is_minimization (bool): Whether lower values indicate better quality.
        """
        self.is_minimization = is_minimization

    @abstractmethod
    def compute(self, solutions: np.ndarray) -> float:
        """Compute quality indicator value for a set of solutions.
        
        Args:
            solutions (np.ndarray): Bi-dimensional numpy array of shape [m, n],
                where m is the number of solutions and n is the number of objectives.
                Each row represents one solution's objective values.
        
        Returns:
            float: The computed quality indicator value.
        
        Note:
            The interpretation of the value depends on is_minimization:
            - If is_minimization=True, lower values indicate better quality.
            - If is_minimization=False, higher values indicate better quality.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `compute`.")

    @abstractmethod
    def get_name(self) -> str:
        """Get the full name of the quality indicator.
        
        Returns:
            str: Full descriptive name of the indicator (e.g., "Generational Distance").
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_name`.")

    @abstractmethod
    def get_short_name(self) -> str:
        """Get the short name/abbreviation of the quality indicator.
        
        Returns:
            str: Short abbreviation of the indicator (e.g., "GD", "IGD", "HV").
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_short_name`.")


class FitnessValue(QualityIndicator):
    """Simple fitness value quality indicator.
    
    Computes the mean of objective values across all solutions. For minimization,
    returns the mean directly. For maximization, returns the negative mean.
    
    This is a simple indicator that can be used when a basic quality measure
    is needed. It does not account for diversity or convergence to a reference
    front, just the average objective value.
    
    Note:
        - For minimization: returns mean of all objectives
        - For maximization: returns negative mean of all objectives
        - This is a basic indicator and may not be suitable for multi-objective
          optimization evaluation without additional metrics.
    """
    
    def __init__(self, is_minimization: bool = True) -> None:
        """Initialize fitness value quality indicator.
        
        Args:
            is_minimization (bool, optional): Whether lower values are better.
                Defaults to True.
        """
        super(FitnessValue, self).__init__(is_minimization=is_minimization)

    def compute(self, solutions: np.ndarray) -> float:
        """Compute mean fitness value across all solutions.
        
        Args:
            solutions (np.ndarray): Array of solutions. Each solution should have
                an 'objectives' attribute containing objective values.
        
        Returns:
            float: Mean objective value (or negative mean for maximization).
        """
        if self.is_minimization:
            mean = np.mean([s.objectives[0] for s in solutions])
        else:
            mean = -np.mean([s.objectives[0] for s in solutions])
        return mean

    def get_name(self) -> str:
        """Get full name of the quality indicator.
        
        Returns:
            str: "Fitness"
        """
        return "Fitness"

    def get_short_name(self) -> str:
        """Get short name/acronym of the quality indicator.
        
        Returns:
            str: "FIT"
        """
        return "FIT"


class GenerationalDistance(QualityIndicator):
    """Generational Distance (GD) quality indicator.
    
    GD measures the average distance from solutions in the obtained front to
    their nearest solutions in the reference (true) Pareto front. Lower GD
    values indicate better convergence to the true Pareto front.
    
    The GD is computed as:
        GD = (1/n) * Σ min(d(s_i, r_j))
    where n is the number of solutions in the obtained front, d(s_i, r_j) is
    the Euclidean distance from solution s_i to reference point r_j, and
    min(d(s_i, r_j)) is the minimum distance from s_i to any reference point.
    
    Attributes:
        reference_front (Optional[np.ndarray]): Reference Pareto front for comparison.
            Shape: [m, n] where m is number of solutions and n is number of objectives.
            Must be set before calling compute().
    
    References:
        [1] Valenzuela-Rendón, Manuel, and Eduardo Uresti-Charre. 1997.
            A Non-Generational Genetic Algorithm for Multiobjective Optimization.
            Paper presented at the ICGA.
        [2] Van Veldhuizen, David A, and Gary B Lamont. 1998.
            "Multiobjective evolutionary algorithm research: A history and analysis."
            In.: Citeseer.
    """
    
    def __init__(self, reference_front: Optional[np.ndarray] = None) -> None:
        """Initialize Generational Distance indicator.
        
        Args:
            reference_front (Optional[np.ndarray], optional): Reference Pareto front.
                If None, must be set later before computing. Defaults to None.
        """
        super(GenerationalDistance, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, front: np.ndarray) -> float:
        """Compute Generational Distance for the given front.
        
        Args:
            front (np.ndarray): Obtained Pareto front. Shape: [m, n] where m is
                number of solutions and n is number of objectives.
        
        Returns:
            float: The Generational Distance value. Lower is better.
        
        Raises:
            ReferenceFrontNotSetException: If reference_front is None.
        """
        if self.reference_front is None:
            raise ReferenceFrontNotSetException("Reference front is not set")
        
        # Compute distances from each solution in front to closest reference point
        distances = spatial.distance.cdist(front, self.reference_front)
        min_distances = np.min(distances, axis=1)
        
        # Apply power transformation (default is 2.0 for squared Euclidean distance)
        pow = 2.0  # Standard for GD calculation
        sum_pow_distances = np.sum(np.power(min_distances, pow))
        
        # Take the root and divide by number of solutions
        result = np.power(sum_pow_distances, 1.0 / pow) / len(front)
        
        return result

    def get_name(self) -> str:
        return "Generational Distance"

    def get_short_name(self) -> str:
        return "GD"


class InvertedGenerationalDistance(QualityIndicator):
    """Inverted Generational Distance (IGD) quality indicator.
    
    IGD measures the average distance from solutions in the reference (true)
    Pareto front to their nearest solutions in the obtained front. Unlike GD,
    IGD considers coverage of the reference front, making it sensitive to both
    convergence and diversity. Lower IGD values indicate better performance.
    
    The IGD is computed as:
        IGD = (1/m) * Σ min(d(r_i, s_j))
    where m is the number of solutions in the reference front, d(r_i, s_j) is
    the Euclidean distance from reference point r_i to solution s_j, and
    min(d(r_i, s_j)) is the minimum distance from r_i to any solution.
    
    Attributes:
        reference_front (Optional[np.ndarray]): Reference Pareto front for comparison.
            Shape: [m, n] where m is number of solutions and n is number of objectives.
            Must be set before calling compute().
    
    References:
        [1] Valenzuela-Rendón, Manuel, and Eduardo Uresti-Charre. 1997.
            A Non-Generational Genetic Algorithm for Multiobjective Optimization.
            Paper presented at the ICGA.
        [2] Van Veldhuizen, David A, and Gary B Lamont. 1998.
            "Multiobjective evolutionary algorithm research: A history and analysis."
            In.: Citeseer.
    """

    def __init__(self, reference_front: Optional[np.ndarray] = None) -> None:
        """Initialize Inverted Generational Distance indicator.
        
        Args:
            reference_front (Optional[np.ndarray], optional): Reference Pareto front.
                If None, must be set later before computing. Defaults to None.
        """
        super(InvertedGenerationalDistance, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, front: np.ndarray) -> float:
        """Compute Inverted Generational Distance for the given front.
        
        Args:
            front (np.ndarray): Obtained Pareto front. Shape: [m, n] where m is
                number of solutions and n is number of objectives.
        
        Returns:
            float: The Inverted Generational Distance value. Lower is better.
        
        Raises:
            ReferenceFrontNotSetException: If reference_front is None.
        """
        if self.reference_front is None:
            raise ReferenceFrontNotSetException("Reference front is not set")
        
        # Compute distances from each reference point to closest solution in front
        distances = spatial.distance.cdist(self.reference_front, front)
        min_distances = np.min(distances, axis=1)
        
        # Apply power transformation (default is 2.0 for squared Euclidean distance)
        pow = 2.0  # Standard for IGD calculation
        sum_pow_distances = np.sum(np.power(min_distances, pow))
        
        # Take the root and divide by number of reference points
        result = np.power(sum_pow_distances, 1.0 / pow) / len(self.reference_front)
        
        return result

    def get_name(self) -> str:
        return "Inverted Generational Distance"

    def get_short_name(self) -> str:
        return "IGD"


class SpreadIndicator(QualityIndicator):
    """Spread quality indicator for measuring diversity of solutions.
    
    The spread indicator measures the diversity of solutions in a front
    compared to a reference front. It calculates how evenly distributed
    the solutions are.
    
    Lower values indicate better diversity (more evenly distributed solutions).
    
    Attributes:
        reference_front (np.ndarray): Reference Pareto front for comparison.
    
    References:
        Deb, Kalyanmoy, Amrit Pratap, Sameer Agarwal, and TAMT Meyarivan. 2002.
        "A fast and elitist multiobjective genetic algorithm: NSGA-II."
        IEEE transactions on evolutionary computation 6 (2):182-97.
    """
    
    def __init__(self, reference_front: Optional[np.ndarray] = None) -> None:
        """Initialize spread indicator.
        
        Args:
            reference_front (Optional[np.ndarray], optional): Reference Pareto front.
                If None, must be set later before computing. Defaults to None.
        """
        super(SpreadIndicator, self).__init__(is_minimization=True)
        self.reference_front = reference_front
    
    def compute(self, front: np.ndarray) -> float:
        """Compute spread indicator value.
        
        Args:
            front (np.ndarray): Obtained Pareto front. Shape: [m, n] where m is
                number of solutions and n is number of objectives.
        
        Returns:
            float: Spread indicator value. Lower is better.
        """
        if self.reference_front is None:
            raise ReferenceFrontNotSetException("Reference front is not set")
        
        # Convert to numpy arrays if not already
        front = np.array(front)
        reference_front = np.array(self.reference_front)
        
        # Get number of objectives
        num_objectives = front.shape[1]
        
        # Step 1. Obtain the maximum and minimum values of the reference front
        max_vals = np.max(reference_front, axis=0)
        min_vals = np.min(reference_front, axis=0)
        
        # Step 2. Normalize front and reference front
        def normalize(data, min_vals, max_vals):
            # Avoid division by zero
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1  # If min equals max, set range to 1
            return (data - min_vals) / ranges
        
        normalized_front = normalize(front, min_vals, max_vals)
        normalized_reference_front = normalize(reference_front, min_vals, max_vals)
        
        # Step 3. Sort normalized fronts
        # Sort by lexicographic order (first objective, then second, etc.)
        sorted_indices = np.lexsort(tuple(normalized_front[:, i] for i in reversed(range(num_objectives))))
        sorted_normalized_front = normalized_front[sorted_indices]
        
        sorted_ref_indices = np.lexsort(tuple(normalized_reference_front[:, i] for i in reversed(range(num_objectives))))
        sorted_normalized_reference_front = normalized_reference_front[sorted_ref_indices]
        
        num_points = len(sorted_normalized_front)
        
        # Step 4. Compute df and dl (distances between extremes)
        if num_points == 0:
            return 1.0
        elif num_points == 1:
            return 1.0
        else:
            # Distance between first points
            df = np.linalg.norm(sorted_normalized_front[0] - sorted_normalized_reference_front[0])
            # Distance between last points
            dl = np.linalg.norm(sorted_normalized_front[-1] - sorted_normalized_reference_front[-1])
            
            # Step 5. Calculate mean distance between consecutive points
            distances = []
            for i in range(len(sorted_normalized_front) - 1):
                dist = np.linalg.norm(sorted_normalized_front[i] - sorted_normalized_front[i + 1])
                distances.append(dist)
            
            if len(distances) == 0:
                return 1.0
            
            mean_dist = np.mean(distances)
            
            # Step 6. Calculate spread
            diversity_sum = df + dl
            for i in range(len(distances)):
                diversity_sum += abs(distances[i] - mean_dist)
            
            denominator = df + dl + (num_points - 1) * mean_dist
            
            if denominator == 0:
                return 0.0  # Perfect spread if denominator is 0
            
            return diversity_sum / denominator
    
    def get_name(self) -> str:
        """Get full name of the quality indicator.
        
        Returns:
            str: "Spread quality indicator"
        """
        return "Spread quality indicator"
    
    def get_short_name(self) -> str:
        """Get short name/acronym of the quality indicator.
        
        Returns:
            str: "SP"
        """
        return "SP"


class EpsilonIndicator(QualityIndicator):
    """Unary epsilon additive indicator (ε-indicator).
    
    The epsilon indicator measures the smallest value ε such that for every
    solution in the reference front, there exists at least one solution in the
    obtained front that is better by at most ε in all objectives (additive form).
    
    Lower epsilon values indicate better quality. This indicator is particularly
    useful for comparing algorithm performance as it provides a single scalar
    value that captures both convergence and diversity aspects.
    
    The epsilon indicator is computed as:
        ε = max_{r∈R} min_{s∈S} max_{k} (s[k] - r[k])
    where R is the reference front, S is the obtained front, and k indexes objectives.
    
    Attributes:
        reference_front (Optional[np.ndarray]): Reference Pareto front for comparison.
            Shape: [m, n] where m is number of solutions and n is number of objectives.
            Must be set before calling compute().
    
    References:
        Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., & da Fonseca, V. G. (2003).
        Performance assessment of multiobjective optimizers: an analysis and review.
        IEEE transactions on evolutionary computation, 7(2), 117-132.
        doi: 10.1109/TEVC.2003.810758.
    """
    
    def __init__(self, reference_front: Optional[np.ndarray] = None, method: str = 'multiplicative') -> None:
        """Initialize epsilon indicator.
        
        Args:
            reference_front (Optional[np.ndarray], optional): Reference Pareto front.
                If None, must be set later before computing. Defaults to None.
            method (str, optional): Method to use ('additive' or 'multiplicative'). Defaults to 'multiplicative'.
        """
        super(EpsilonIndicator, self).__init__(is_minimization=True)
        self.reference_front = reference_front
        self.method = method

    def compute(self, front: np.ndarray) -> float:
        """Compute epsilon indicator value.
        
        Args:
            front (np.ndarray): Obtained Pareto front. Shape: [m, n] where m is
                number of solutions and n is number of objectives.
        
        Returns:
            float: Epsilon indicator value. Lower is better.
        
        Note:
            The computation finds the minimum epsilon such that each reference
            point is epsilon-dominated by at least one solution in the front.
        """
        if self.method == 'additive':
            # Additive epsilon indicator: max_{r∈R} min_{s∈S} max_{k} (s[k] - r[k])
            return max([min([max([s2[k] - s1[k] for k in range(len(s2))]) for s2 in front]) for s1 in self.reference_front])
        elif self.method == 'multiplicative':
            # Multiplicative epsilon indicator: max_{r∈R} min_{s∈S} max_{k} (s[k] / r[k])
            # Need to handle zero values to avoid division by zero
            result = float('-inf')
            for s1 in self.reference_front:  # Iterate over reference points
                min_eps_j = float('inf')
                for s2 in front:  # Iterate over front points
                    max_eps_k = float('-inf')
                    for k in range(len(s2)):  # Iterate over objectives
                        if s1[k] == 0:
                            # If reference value is 0, use a large penalty value
                            eps_temp = float('inf') if s2[k] > 0 else 0.0
                        else:
                            # Calculate ratio
                            eps_temp = s2[k] / s1[k]
                        max_eps_k = max(max_eps_k, eps_temp)
                    min_eps_j = min(min_eps_j, max_eps_k)
                result = max(result, min_eps_j)
            return result
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'additive' or 'multiplicative'.")

    def get_name(self) -> str:
        """Get full name of the quality indicator.
        
        Returns:
            str: "Additive Epsilon" or "Multiplicative Epsilon"
        """
        if self.method == 'multiplicative':
            return "Multiplicative Epsilon"
        else:
            return "Additive Epsilon"

    def get_short_name(self) -> str:
        """Get short name/acronym of the quality indicator.
        
        Returns:
            str: "EP"
        """
        return "EP"


class HyperVolume(QualityIndicator):
    """Hypervolume (HV) quality indicator using dimension-sweep algorithm.
    
    The hypervolume indicator measures the volume of the objective space dominated
    by a set of solutions and bounded by a reference point. It is one of the most
    widely used quality indicators in multi-objective optimization as it measures
    both convergence and diversity in a single metric.
    
    Higher hypervolume values indicate better quality. The hypervolume is the
    Lebesgue measure of the union of hyper-rectangles, each defined by a solution
    and the reference point.
    
    This implementation uses variant 3 of the dimension-sweep algorithm proposed
    by Fonseca et al. for efficient hypervolume computation.
    
    Attributes:
        referencePoint (Optional[List[float]]): Reference point for hypervolume
            computation (typically nadir point or worse than worst solutions).
        list (MultiList): Internal data structure for efficient computation.
    
    References:
        Fonseca, C. M., Paquete, L., & Lopez-Ibanez, M. (2006). An improved
        dimension-sweep algorithm for the hypervolume indicator. In IEEE Congress
        on Evolutionary Computation (pp. 1157-1163). Vancouver, Canada.
    
    Note:
        - Assumes minimization of all objectives.
        - The reference point should be worse than all solutions in all objectives.
    """
    
    def __init__(self, reference_point: Optional[List[float]] = None) -> None:
        """Initialize hypervolume indicator.
        
        Args:
            reference_point (Optional[List[float]], optional): Reference point for
                hypervolume computation. Should be worse than all solutions in all
                objectives. If None, must be set later. Defaults to None.
        """
        super(HyperVolume, self).__init__(is_minimization=False)
        self.referencePoint = reference_point
        self.list: MultiList = []

    def compute(self, front: np.ndarray) -> float:
        """Compute hypervolume for the given front.
        
        Before computation, the front and reference point are translated so that
        the reference point becomes [0, ..., 0], simplifying the calculation.
        
        Args:
            front (np.ndarray): Non-dominated front. Shape: [m, n] where m is
                number of solutions and n is number of objectives.
        
        Returns:
            float: Hypervolume value. Higher is better.
        
        Note:
            Only solutions that dominate the reference point are considered.
            The computation uses an efficient recursive dimension-sweep algorithm.
        """

        def weakly_dominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevant_points = []
        reference_point = self.referencePoint
        dimensions = len(reference_point)
        for point in front:
            # only consider points that dominate the reference point
            if weakly_dominates(point, reference_point):
                relevant_points.append(point)
        if any(reference_point):
            # shift points so that reference_point == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            for j in range(len(relevant_points)):
                relevant_points[j] = [relevant_points[j][i] - reference_point[i] for i in range(dimensions)]
        self._pre_process(relevant_points)
        bounds = [-1.0e308] * dimensions
        return self._hv_recursive(dimensions - 1, len(relevant_points), bounds)

    def _hv_recursive(self, dim_index: int, length: int, bounds: list):
        """Recursive hypervolume calculation using dimension-sweep.
        
        Implements the recursive dimension-sweep algorithm for hypervolume computation.
        The algorithm works by processing dimensions from highest to lowest, using
        the MultiList structure to maintain sorted points.
        
        Args:
            dim_index (int): Current dimension being processed (decrements from
                highest to lowest dimension).
            length (int): Number of points currently in the list.
            bounds (List[float]): Bounds array for tracking dominated regions.
        
        Returns:
            float: Hypervolume contribution in the current dimension slice.
        
        Note:
            This is an internal method called recursively during hypervolume computation.
            Assumes reference point is [0, ..., 0] after translation, which simplifies
            calculations compared to the original paper.
        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dim_index == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dim_index == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                p_cargo = p.cargo
                hvol += h * (q.cargo[1] - p_cargo[1])
                if p_cargo[0] < h:
                    h = p_cargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hv_recursive = self._hv_recursive
            p = sentinel
            q = p.prev[dim_index]
            while q.cargo is not None:
                if q.ignore < dim_index:
                    q.ignore = 0
                q = q.prev[dim_index]
            q = p.prev[dim_index]
            while length > 1 and (
                    q.cargo[dim_index] > bounds[dim_index] or q.prev[dim_index].cargo[dim_index] >= bounds[dim_index]
            ):
                p = q
                remove(p, dim_index, bounds)
                q = p.prev[dim_index]
                length -= 1
            q_area = q.area
            q_cargo = q.cargo
            q_prev_dim_index = q.prev[dim_index]
            if length > 1:
                hvol = q_prev_dim_index.volume[dim_index] + q_prev_dim_index.area[dim_index] * (
                        q_cargo[dim_index] - q_prev_dim_index.cargo[dim_index]
                )
            else:
                q_area[0] = 1
                q_area[1: dim_index + 1] = [q_area[i] * -q_cargo[i] for i in range(dim_index)]
            q.volume[dim_index] = hvol
            if q.ignore >= dim_index:
                q_area[dim_index] = q_prev_dim_index.area[dim_index]
            else:
                q_area[dim_index] = hv_recursive(dim_index - 1, length, bounds)
                if q_area[dim_index] <= q_prev_dim_index.area[dim_index]:
                    q.ignore = dim_index
            while p is not sentinel:
                p_cargo_dim_index = p.cargo[dim_index]
                hvol += q.area[dim_index] * (p_cargo_dim_index - q.cargo[dim_index])
                bounds[dim_index] = p_cargo_dim_index
                reinsert(p, dim_index, bounds)
                length += 1
                q = p
                p = p.next[dim_index]
                q.volume[dim_index] = hvol
                if q.ignore >= dim_index:
                    q.area[dim_index] = q.prev[dim_index].area[dim_index]
                else:
                    q.area[dim_index] = hv_recursive(dim_index - 1, length, bounds)
                    if q.area[dim_index] <= q.prev[dim_index].area[dim_index]:
                        q.ignore = dim_index
            hvol -= q.area[dim_index] * q.cargo[dim_index]
            return hvol

    def _pre_process(self, front):
        """Preprocess front by setting up MultiList structure for hypervolume calculation.
        
        Creates a MultiList structure with nodes sorted by each dimension, which
        is required for the efficient dimension-sweep algorithm.
        
        Args:
            front (List[List[float]]): List of points (objective vectors) to preprocess.
        
        Note:
            This method is called internally before hypervolume computation.
            It sets up self.list with nodes sorted by each dimension.
        """
        dimensions = len(self.referencePoint)
        node_list = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self._sort_by_dimension(nodes, i)
            node_list.extend(nodes, i)
        self.list = node_list

    def _sort_by_dimension(self, nodes, i):
        """Sort nodes by their i-th dimension value.
        
        Args:
            nodes (List[MultiList.Node]): List of nodes to sort (modified in-place).
            i (int): Dimension index to sort by (0-based).
        
        Note:
            This method sorts nodes in-place based on the i-th objective value.
        """
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        decorated.sort(key=lambda n: n[0])
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]

    def get_name(self) -> str:
        """Get full name of the quality indicator.
        
        Returns:
            str: "Hypervolume (Fonseca et al. implementation)"
        """
        return "Hypervolume (Fonseca et al. implementation)"

    def get_short_name(self) -> str:
        """Get short name/acronym of the quality indicator.
        
        Returns:
            str: "HV"
        """
        return "HV"


class MultiList:
    """Special multi-list structure used by FonsecaHyperVolume algorithm.
    
    A data structure consisting of several doubly linked lists that share
    common nodes. Each node can belong to multiple lists simultaneously,
    having multiple predecessors and successors (one for each list it belongs to).
    This structure is used internally by the hypervolume computation algorithm.
    
    Attributes:
        number_lists (int): Number of doubly linked lists in this structure.
        sentinel (Node): Sentinel node used to mark the end of each list.
    
    Note:
        This is an internal data structure used specifically for efficient
        hypervolume computation in the FonsecaHyperVolume implementation.
    """

    class Node:
        """Node in a multi-list structure.
        
        A node that can belong to multiple doubly linked lists simultaneously.
        Maintains separate next/prev pointers for each list, along with
        area and volume information.
        
        Attributes:
            cargo: Data stored in the node (typically objective values).
            next (List[Node]): Next node pointers for each list.
            prev (List[Node]): Previous node pointers for each list.
            ignore (int): Ignore flag for algorithm use.
            area (List[float]): Area information for each list.
            volume (List[float]): Volume information for each list.
        """
        
        def __init__(self, number_lists, cargo=None):
            """Initialize multi-list node.
            
            Args:
                number_lists (int): Number of lists this node can belong to.
                cargo: Optional data to store in the node.
            """
            self.cargo = cargo
            self.next = [None] * number_lists
            self.prev = [None] * number_lists
            self.ignore = 0
            self.area = [0.0] * number_lists
            self.volume = [0.0] * number_lists

        def __str__(self):
            """String representation of the node.
            
            Returns:
                str: String representation of the cargo data.
            """
            return str(self.cargo)

    def __init__(self, number_lists):
        """Initialize multi-list structure.
        
        Args:
            number_lists (int): Number of doubly linked lists to create.
        
        Note:
            Creates a sentinel node that serves as the boundary for all lists.
        """
        self.number_lists = number_lists
        self.sentinel = MultiList.Node(number_lists)
        self.sentinel.next = [self.sentinel] * number_lists
        self.sentinel.prev = [self.sentinel] * number_lists

    def __str__(self):
        """String representation of all lists in the multi-list.
        
        Returns:
            str: Multi-line string showing the contents of each list.
        """
        strings = []
        for i in range(self.number_lists):
            current_list = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                current_list.append(str(node))
                node = node.next[i]
            strings.append(str(current_list))
        string_repr = ""
        for string in strings:
            string_repr += string + "\n"
        return string_repr

    def __len__(self):
        """Get the number of lists in this multi-list.
        
        Returns:
            int: Number of doubly linked lists.
        """
        return self.number_lists

    def get_length(self, i):
        """Get the length of a specific list.
        
        Args:
            i (int): Index of the list (0-based).
        
        Returns:
            int: Number of nodes in the i-th list (excluding sentinel).
        """
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length

    def append(self, node, index):
        """Append a node to the end of a specific list.
        
        Args:
            node (Node): Node to append.
            index (int): Index of the list to append to.
        
        Note:
            The node is added at the end of the specified list, before the sentinel.
        """
        last_but_one = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = last_but_one
        # set the last element as the new one
        self.sentinel.prev[index] = node
        last_but_one.next[index] = node

    def extend(self, nodes, index):
        """Extend a list with multiple nodes.
        
        Args:
            nodes (List[Node]): List of nodes to append.
            index (int): Index of the list to extend.
        
        Note:
            All nodes are appended to the end of the specified list.
        """
        sentinel = self.sentinel
        for node in nodes:
            last_but_one = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = last_but_one
            # set the last element as the new one
            sentinel.prev[index] = node
            last_but_one.next[index] = node

    def remove(self, node, index, bounds):
        """Remove a node from all lists up to a given index.
        
        Removes the node from all lists in the range [0, index) and updates
        bounds based on the node's cargo values.
        
        Args:
            node (Node): Node to remove.
            index (int): Remove from all lists with index < index.
            bounds (List[float]): Bounds array to update based on node cargo.
        
        Returns:
            Node: The removed node.
        
        Note:
            Bounds are updated to track the minimum cargo values encountered.
        """
        for i in range(index):
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """Reinsert a node into lists up to a given index.
        
        Reinserts a previously removed node back into all lists in the range
        [0, index). Assumes the node's next and prev pointers are still valid.
        
        Args:
            node (Node): Node to reinsert.
            index (int): Reinsert into all lists with index < index.
            bounds (List[float]): Bounds array to update based on node cargo.
        
        Note:
            This method should be used to restore a node that was previously
            removed using remove(). The node's connections must still be valid.
        """
        for i in range(index):
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


class NormalizedHyperVolume(QualityIndicator):
    """Normalized hypervolume quality indicator.
    
    Computes the normalized hypervolume relative to a reference front.
    The normalized hypervolume is calculated as:
    
    normalized_HV = 1 - (HV of the front / HV of the reference front)
    
    This provides a value between 0 and 1, where 0 indicates the front
    has the same hypervolume as the reference front, and values closer
    to 1 indicate worse quality.
    
    Attributes:
        __EPS (float): Small epsilon value to avoid division by zero.
        reference_point (Iterable[float]): Reference point for hypervolume computation.
        _hv (HyperVolume): Internal HyperVolume indicator instance.
        _reference_hypervolume (float): Hypervolume of the reference front.
    
    Note:
        - Assumes minimization of all objectives.
        - Requires reference front to have non-zero hypervolume.
        - Lower values indicate better quality (minimization indicator).
    
    Reference:
        Based on the normalized hypervolume metric used in multi-objective
        optimization to measure the relative quality of a front compared
        to a known reference front.
    """
    
    __EPS = 1.0e-14

    def __init__(self, reference_point: Iterable[float], reference_front: np.ndarray) -> None:
        """Initialize normalized hypervolume indicator.
        
        Args:
            reference_point (Iterable[float]): Reference point for hypervolume
                computation (typically nadir point or worse than worst).
            reference_front (np.ndarray): Reference Pareto front for normalization.
                Used to compute the reference hypervolume.
        
        Raises:
            AssertionError: If the hypervolume of the reference front is zero.
        
        Note:
            The reference front hypervolume is computed once during initialization
            and used for all subsequent normalization calculations.
        """
        self.reference_point = reference_point
        self._hv = HyperVolume(reference_point=reference_point)
        self._reference_hypervolume = self._hv.compute(reference_front) + self.__EPS
        assert self._reference_hypervolume != 0, "Hypervolume of reference front is zero"

    def compute(self, solutions: np.ndarray) -> float:
        """Compute normalized hypervolume for given solutions.
        
        Args:
            solutions (np.ndarray): Array of solutions (Pareto front) to evaluate.
                Shape should be (n_solutions, n_objectives).
        
        Returns:
            float: Normalized hypervolume value between 0 and 1.
                - 0: Front has same hypervolume as reference
                - 1: Front has zero hypervolume (worst case)
        
        Note:
            Lower values indicate better quality (minimization indicator).
        """
        hv = self._hv.compute(front=solutions)
        return 1 - (hv / self._reference_hypervolume)

    def get_name(self) -> str:
        """Get full name of the quality indicator.
        
        Returns:
            str: "Normalized Hypervolume"
        """
        return "Normalized Hypervolume"

    def get_short_name(self) -> str:
        """Get short name/acronym of the quality indicator.
        
        Returns:
            str: "NHV"
        """
        return "NHV"
