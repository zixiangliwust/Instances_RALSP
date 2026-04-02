# -*- coding: utf-8 -*-
import numpy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, List, Optional, TypeVar
from evolu.core.solution import Solution
from evolu.util.checking import Check

"""
module:: neighborhood
synopsis: implementation of neighborhoods in the context of list of solutions. The goal is,
given the index of an element of the list, to find its neighbor solutions according to a criterion.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""
S = TypeVar("S", bound=Solution)


class Neighborhood(Generic[S], ABC):
    """Base class for neighborhood structures.
    
    Neighborhood structures define which solutions are considered "neighbors"
    of a given solution. They are commonly used in decomposition-based algorithms
    (e.g., MOEA/D) where solutions are associated with weight vectors, and
    neighborhoods are defined based on similarity between weight vectors.
    
    The goal is to find neighbor solutions according to some criterion (e.g.,
    distance in weight space, spatial proximity in solution space).
    
    Note:
        Subclasses must implement get_neighbors() method.
    """
    
    @abstractmethod
    def get_neighbors(self, index: int, solution_list: List[S]) -> List[S]:
        """Get neighbor solutions for a solution at the given index.
        
        Args:
            index (int): Index of the solution in solution_list for which to
                find neighbors.
            solution_list (List[S]): List of all solutions.
        
        Returns:
            List[S]: List of neighbor solutions.
        
        Note:
            The neighborhood structure determines which solutions are considered
            neighbors based on the implementation-specific criterion.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `get_neighbors`.")


class WeightNeighborhood(Neighborhood[S], ABC):
    """Base class for weight-based neighborhood structures.
    
    This class provides the foundation for neighborhoods based on weight vectors,
    commonly used in MOEA/D and similar decomposition-based algorithms. Neighbors
    are determined by the distance between weight vectors in weight space.
    
    Attributes:
        number_of_weight_vectors (int): Number of weight vectors (population size).
        neighborhood_size (int): Number of neighbors for each weight vector.
        weight_vector_size (int): Number of objectives (dimension of weight vectors).
        weights_path (Optional[str]): Path to directory containing precomputed
            weight vector files (for dimensions > 2).
        neighborhood (numpy.ndarray): Matrix of shape (number_of_weight_vectors, neighborhood_size)
            storing neighbor indices for each weight vector.
        weight_vectors (numpy.ndarray): Matrix of shape (number_of_weight_vectors, weight_vector_size)
            storing all weight vectors.
    
    Note:
        Subclasses must initialize weight_vectors and compute neighborhood relationships.
    """
    
    def __init__(self,
                 number_of_weight_vectors: int,
                 neighborhood_size: int,
                 weight_vector_size: int = 2,
                 weights_path: Optional[str] = None,
                 ) -> None:
        """Initialize weight-based neighborhood structure.
        
        Args:
            number_of_weight_vectors (int): Number of weight vectors (population size).
            neighborhood_size (int): Number of neighbors for each weight vector.
            weight_vector_size (int, optional): Number of objectives. Defaults to 2.
            weights_path (Optional[str], optional): Path to directory containing
                precomputed weight files. Required for weight_vector_size > 2.
                Defaults to None.
        """
        self.number_of_weight_vectors = number_of_weight_vectors
        self.neighborhood_size = neighborhood_size
        self.weight_vector_size = weight_vector_size
        self.weights_path = weights_path
        self.neighborhood = numpy.zeros((number_of_weight_vectors, neighborhood_size), dtype=int)
        self.weight_vectors = numpy.zeros((number_of_weight_vectors, weight_vector_size))


class WeightVectorNeighborhood(WeightNeighborhood[S]):
    """Weight vector-based neighborhood for decomposition algorithms.
    
    This neighborhood structure is used in MOEA/D and similar algorithms. It
    creates uniform weight vectors (for 2 objectives) or loads precomputed ones
    (for higher dimensions), then defines neighborhoods based on Euclidean distance
    between weight vectors in weight space.
    
    For each weight vector, its neighbors are the closest weight vectors in terms
    of Euclidean distance. This ensures that solutions associated with similar
    weight vectors (similar search directions) can collaborate during evolution.
    
    Reference:
        Zhang, Q., & Li, H. (2007). MOEA/D: A multiobjective evolutionary
        algorithm based on decomposition. IEEE Transactions on evolutionary
        computation, 11(6), 712-731.
    
    Precomputed weights source:
        Zhang, Multiobjective Optimization Problems With Complicated Pareto Sets,
        MOEA/D and NSGA-II. Available at:
        http://dces.essex.ac.uk/staff/qzhang/MOEAcompetition/CEC09final/code/ZhangMOEADcode/moead030510.rar
    """
    
    def __init__(self,
                 number_of_weight_vectors: int,
                 neighborhood_size: int,
                 weight_vector_size: int = 2,
                 weights_path: Optional[str] = None,
                 ) -> None:
        """Initialize weight vector neighborhood.
        
        Automatically initializes weight vectors and computes neighborhood
        relationships based on distance between weight vectors.
        
        Args:
            number_of_weight_vectors (int): Number of weight vectors (population size).
            neighborhood_size (int): Number of neighbors for each weight vector.
            weight_vector_size (int, optional): Number of objectives. Defaults to 2.
            weights_path (Optional[str], optional): Path to directory containing
                precomputed weight files (W{D}_{N}.dat format). Required for
                weight_vector_size > 2. Defaults to None.
        """
        super(WeightVectorNeighborhood, self).__init__(
            number_of_weight_vectors, neighborhood_size, weight_vector_size, weights_path
        )
        self.initialize_uniform_weight(weight_vector_size, number_of_weight_vectors)
        self.initialize_neighborhood()

    def initialize_uniform_weight(self, weight_vector_size: int, number_of_weight_vectors: int) -> None:
        """Initialize uniform weight vectors.
        
        For 2 objectives, creates uniform weight vectors where each weight vector
        [w1, w2] satisfies w1 + w2 = 1.0. For higher dimensions, loads precomputed
        weights from files.
        
        Args:
            weight_vector_size (int): Number of objectives.
            number_of_weight_vectors (int): Number of weight vectors to create.
        
        Raises:
            FileNotFoundError: If weight_vector_size > 2 and weight file is not found.
        
        Note:
            For 2 objectives: Creates uniform distribution [0, 1], [1/(N-1), (N-2)/(N-1)], ..., [1, 0]
            For >2 objectives: Requires precomputed weight files in weights_path directory.
        """
        if weight_vector_size == 2:
            for i in range(0, number_of_weight_vectors):
                v = 1.0 * i / (number_of_weight_vectors - 1)
                self.weight_vectors[i, 0] = v
                self.weight_vectors[i, 1] = 1 - v
        else:
            file_name = "W{}D_{}.dat".format(weight_vector_size, number_of_weight_vectors)
            file_path = self.weights_path + "/" + file_name
            if Path(file_path).is_file():
                with open(file_path) as file:
                    for index, line in enumerate(file):
                        vector = [float(x) for x in line.split()]
                        self.weight_vectors[index][:] = vector
            else:
                raise FileNotFoundError("Failed to initialize weights: {} not found".format(file_path))

    def initialize_neighborhood(self) -> None:
        """Compute neighborhood relationships based on weight vector distances.
        
        For each weight vector, finds its neighborhood_size closest neighbors
        by computing Euclidean distances between all weight vectors and selecting
        the nearest ones.
        
        The neighborhood matrix is populated with indices of neighbor weight vectors,
        sorted by distance (closest first).
        """
        distance = numpy.zeros((len(self.weight_vectors), len(self.weight_vectors)))
        for i in range(len(self.weight_vectors)):
            for j in range(len(self.weight_vectors)):
                distance[i][j] = numpy.linalg.norm(self.weight_vectors[i] - self.weight_vectors[j])
            indexes = numpy.argsort(distance[i, :])
            self.neighborhood[i, :] = indexes[0: self.neighborhood_size]

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        """Get neighbor solutions for the solution at the given index.
        
        Args:
            index (int): Index of the solution in solution_list.
            solution_list (List[Solution]): List of all solutions.
        
        Returns:
            List[Solution]: List of neighbor solutions corresponding to the
                neighbor weight vectors of the weight vector at index.
        
        Raises:
            IndexError: If neighbor indices are out of range for solution_list.
        """
        neighbors_indexes = self.neighborhood[index]
        if any(i > len(solution_list) for i in neighbors_indexes):
            raise IndexError("Neighbor index out of range")
        return [solution_list[i] for i in neighbors_indexes]

    def get_neighborhood(self) -> numpy.ndarray:
        """Get the neighborhood matrix.
        
        Returns:
            numpy.ndarray: Matrix of shape (number_of_weight_vectors, neighborhood_size)
                where each row contains indices of neighbor weight vectors for
                that weight vector, sorted by distance (closest first).
        """
        return self.neighborhood


class TwoDimensionalMesh(Neighborhood[S]):
    """Two-dimensional mesh neighborhood structure.
    
    This neighborhood defines neighbors based on a 2D grid structure (mesh topology).
    Solutions are arranged in a rows × columns grid, and neighbors are defined
    based on spatial adjacency in the grid (e.g., 4-connected or 8-connected neighborhoods).
    
    This topology is commonly used in cellular genetic algorithms where solutions
    interact only with their immediate neighbors in the spatial grid.
    
    Attributes:
        rows (int): Number of rows in the 2D mesh.
        columns (int): Number of columns in the 2D mesh.
        neighborhood (List[List[int]]): Neighborhood structure defining which
            positions are neighbors. Each inner list contains relative position
            offsets [row_offset, column_offset] for neighbors.
    """

    def __init__(self, rows: int, columns: int, neighborhood: List[List[int]]) -> None:
        self.rows = rows
        self.columns = columns
        self.neighborhood = neighborhood
        self.mesh = None
        self.__create_mesh()

    def __create_mesh(self) -> None:
        """Example:
        if rows = 5, and columns=3, we need to fill the mesh as follows
        |00-01-02|
        |03-04-05|
        |06-07-08|
        |09-10-11|
        |12-13-14|
        """
        self.mesh = numpy.zeros((self.rows, self.columns), dtype=int)
        next_value = 0
        for i in range(self.rows):
            for j in range(self.columns):
                self.mesh[i][j] = next_value
                next_value += 1

    def __get_row(self, index: int) -> int:
        """
        Returns the row in the mesh where the index is local
        param index:
        return:
        """
        return index // self.columns

    def __get_column(self, index: int) -> int:
        """
        Returns the column in the mesh where the index is local
        param index:
        return:
        """
        return index % self.columns

    def __get_neighbor(self, index: int, neighbor: List[int]) -> int:
        """
        Returns the neighbor of the index
        param index:
        param neighbor:
        return:
        """
        row = self.__get_row(index)
        r = (row + neighbor[0]) % self.rows
        if r < 0:
            r = self.rows - 1
        column = self.__get_column(index)
        c = (column + neighbor[1]) % self.columns
        if c < 0:
            c = self.columns - 1
        return self.mesh[r][c]

    def __find_neighbors(self, solution_list: List[S], solution_index: int, neighborhood: List[List[int]]) -> List[S]:
        """
        Returns a list containing the neighbors of a given solution belonging to a solution list
        param solution_list:
        param solution_index:
        param neighborhood:
        return:
        """
        neighbors = []
        for neighbor in neighborhood:
            index = self.__get_neighbor(solution_index, neighbor=neighbor)
            neighbors.append(solution_list[index])
        return neighbors

    def get_neighbors(self, index: int, solution_list: List[Solution]) -> List[Solution]:
        Check.is_not_none(solution_list)
        Check.that(len(solution_list) != 0, "The list of solutions is empty")
        return self.__find_neighbors(solution_list, index, self.neighborhood)


class C9(TwoDimensionalMesh[S]):
    """C9 neighborhood structure (9-connected neighborhood).
    
    Defines a 9-connected neighborhood in a two-dimensional mesh topology.
    Each solution has 8 neighbors in a 3x3 grid centered on itself, including
    all adjacent and diagonal neighbors. Used in cellular algorithms like
    MOCell.
    
    Neighborhood shape::
    
        * * *
        * o *
        * * *
    
    Where 'o' is the center solution and '*' are neighbors.
    
    Topology:
        - north: (-1, 0)
        - south: (1, 0)
        - east: (0, 1)
        - west: (0, -1)
        - north_east: (-1, 1)
        - north_west: (-1, -1)
        - south_east: (1, 1)
        - south_west: (1, -1)
    
    Args:
        rows (int): Number of rows in the mesh.
        columns (int): Number of columns in the mesh.
    
    Note:
        This neighborhood structure is commonly used in cellular evolutionary
        algorithms where solutions are arranged in a grid and interact with
        their immediate neighbors.
    """

    def __init__(self, rows: int, columns: int) -> None:
        """Initialize C9 neighborhood structure.
        
        Args:
            rows (int): Number of rows in the two-dimensional mesh.
            columns (int): Number of columns in the two-dimensional mesh.
        """
        super(C9, self).__init__(rows, columns, [[-1, 0], [1, 0], [0, 1], [0, -1], [-1, 1], [-1, -1], [1, 1], [1, -1]])


class L5(TwoDimensionalMesh[S]):
    """L5 neighborhood structure (5-connected, cross-shaped neighborhood).
    
    Defines a 5-connected neighborhood in a two-dimensional mesh topology.
    Each solution has 4 neighbors forming a cross pattern (north, south, east, west),
    excluding diagonal neighbors. This is a more sparse neighborhood than C9.
    
    Neighborhood shape::
    
          *
        * o *
          *
    
    Where 'o' is the center solution and '*' are neighbors.
    
    Topology:
        - north: (-1, 0)
        - south: (1, 0)
        - east: (0, 1)
        - west: (0, -1)
    
    Args:
        rows (int): Number of rows in the mesh.
        columns (int): Number of columns in the mesh.
    
    Note:
        The L5 neighborhood is more sparse than C9, providing less connectivity
        between solutions in cellular algorithms.
    """

    def __init__(self, rows: int, columns: int) -> None:
        """Initialize L5 neighborhood structure.
        
        Args:
            rows (int): Number of rows in the two-dimensional mesh.
            columns (int): Number of columns in the two-dimensional mesh.
        """
        super(L5, self).__init__(rows, columns, [[-1, 0], [1, 0], [0, 1], [0, -1]])
