"""Multi-Objective Reconfigurable Assembly Line Balancing Problem implementations."""

import copy
import random
import re
from typing import List, Optional, Tuple
import numpy as np

from evolu.core.solution import FloatSolution, IntegerSolution
from evolusalbp1.problem.singleobjective.SALBP1_problems import ALBP1Base


class MOReconfigurableALSPBase(ALBP1Base):
    """Base class for Multi-Objective Reconfigurable Assembly Line Balancing Problem."""
    
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize MOReconfigurableALSP base problem."""
        super(MOReconfigurableALSPBase, self).__init__()
        self.number_of_variables: Optional[int] = None
        self.number_of_objectives: int = 3
        self.number_of_constraints: int = 0
        self.obj_labels: List[str] = ["$ f_{} $".format(i) for i in range(self.number_of_objectives)]
        
        # Problem parameters from paper
        self.number_of_product_types: Optional[int] = None
        self.number_of_workstations: Optional[int] = None
        self.number_of_parts: Optional[int] = None
        self.product_cycle: List[int] = []  # d_i in paper
        self.station_length: List[int] = []
        self.working_hours: List[List[int]] = []  # PT_{i,k} in paper
        self.product_switching_cost: List[List[int]] = []  # C_{i,j} in paper
        self.interval_time: int = 5
        self.velocity: int = 1
        self.part_requirement_matrix: List[List[int]] = []  # z_{i,l} in paper        
        # Part frequency constraints (A_l, B_l) from paper
        self.part_frequency_constraints: List[Tuple[int, int]] = []      
        
        print("Input problem and process the problem data")
        self.__read_instance_from_file(file_path, file_name)

        self.product_sequence: List[int] = []
        self.objectives: List[float] = []

    def __read_instance_from_file(self, file_path: str, file_name: str) -> None:
        """Read problem instance data from file with paper's data format."""
        self.file_path = file_path
        self.file_name = file_name
        file_path_and_name = file_path + file_name
        print("The selected problem is: " + str(file_name))
        file = open(file_path_and_name, "r")
        lines = file.readlines()
        file.close()
        # Clean and parse lines
        lines = [line.strip() for line in lines if line.strip() and not line.startswith('<')]
        # print(f"lines are {lines}")
        line_index = 0
        self.number_of_product_types = int(lines[line_index])
        # print(f"product category is {self.number_of_product_types}")
        line_index = line_index + 1
        for i in range(0, self.number_of_product_types):
            line = lines[line_index].split(' ')
            line = [int(j) for j in line]
            self.product_cycle.append(line[-1])
            line_index = line_index + 1
        # print(f"product cycle is {self.product_cycle}")
        self.number_of_workstations = int(lines[line_index])
        # print(f"number of workstations is {self.number_of_workstations}")
        line_index = line_index + 1
        line = lines[line_index].split(' ')
        self.station_length = [int(j) for j in line]
        # print(f"station length is {self.station_length}")
        line_index = line_index + 1
        # Create a 2D array filled with zeros
        self.working_hours = [[0 for _ in range(self.number_of_workstations)]
                              for _ in range(self.number_of_product_types)]
        for i in range(0, self.number_of_product_types):
            line = lines[line_index].split(' ')
            product = [int(x) for x in line]
            product.pop(0)  # Remove the product number indicator
            for j in range(0, self.number_of_workstations):
                self.working_hours[i][j] = product[j]
            line_index = line_index + 1
        self.interval_time = int(lines[line_index])
        line_index = line_index + 1
        # print(f"working hours is {self.working_hours}")
        self.product_switching_cost = [[0 for _ in range(self.number_of_product_types)]
                                       for _ in range(self.number_of_product_types)]
        for i in range(0, self.number_of_product_types * self.number_of_product_types):
            line = lines[line_index]
            line = re.split(" |,", line)
            line = [int(x) for x in line]
            self.product_switching_cost[line[0] - 1][line[1] - 1] = line[2]
            line_index = line_index + 1
        # print(f"product switching cost is {self.product_switching_cost}")
        self.number_of_parts = int(lines[line_index])
        # print(f"number of parts is {self.number_of_parts}")
        line_index = line_index + 1
        for i in range(self.number_of_parts):
            line = lines[line_index].split(' ')
            line = [int(x) for x in line]
            self.part_frequency_constraints.append(line)
            line_index = line_index + 1
        # print(f"optional frequency is{self.part_frequency_constraints}")
        for i in range(self.number_of_product_types):
            line = lines[line_index].split(' ')
            line = [int(x) for x in line]
            self.part_requirement_matrix.append(line)
            line_index = line_index + 1
        # print(f"product assembly parts is{self.part_requirement_matrix}")

    def _calculate_switching_cost(self) -> float:
        """Calculate reconfiguration cost (F1 in paper)."""
        
        reconfiguration_cost = 0.0
        
        # Calculate cost for consecutive pairs (including cyclic wrap-around)
        for i in range(len(self.product_sequence) - 1):
            current_product = self.product_sequence[i]
            next_product = self.product_sequence[i + 1]
            reconfiguration_cost += self.product_switching_cost[current_product][next_product]
        
        # Add cost from last to first (cyclic completion)
        reconfiguration_cost += self.product_switching_cost[self.product_sequence[-1]][self.product_sequence[0]]
        
        return reconfiguration_cost

    def _calculate_workload_equalization(self) -> float:
        """Calculate production workload equalization violations (F2 in paper)."""
        
        part_frequency_violations = 0.0
        
        # Check each part's frequency constraint
        for l in range(self.number_of_parts):
            A_l, B_l = self.part_frequency_constraints[l]
            
            # Check sliding window of size B_l
            for start_idx in range(len(self.product_sequence)):
                # Count how many products in this window require part l
                count = 0
                for offset in range(B_l):
                    idx = (start_idx + offset) % len(self.product_sequence)
                    product_idx = self.product_sequence[idx]
                    if self.part_requirement_matrix[product_idx][l] >= 1:
                        count += 1
                
                # If count exceeds A_l, it's a violation
                if count > A_l:
                    part_frequency_violations += 1
        
        return part_frequency_violations

    def _calculate_logistics_leveling(self) -> float:
        """Calculate logistics logistics_leveling (F3 in paper)."""
        
        # Initialize arrays to track workstation completion times and cumulative production
        completion_times = [[0.0] * self.number_of_workstations for _ in range(len(self.product_sequence))]
        
        # Calculate completion times for the first product
        first_product = self.product_sequence[0]
        completion_times[0][0] = self.working_hours[first_product][0]
        for workstation_idx in range(1, self.number_of_workstations):
            completion_times[0][workstation_idx] = completion_times[0][workstation_idx-1] + self.working_hours[first_product][workstation_idx]
        
        # Calculate completion times for remaining products using max logic
        for p in range(1, len(self.product_sequence)):
            product = self.product_sequence[p]
            
            # First workstation: add previous workstation time, interval time, and processing time
            completion_times[p][0] = completion_times[p-1][0] + self.interval_time + self.working_hours[product][0]
            
            # Other workstations: use max logic to determine earliest possible start time
            for workstation_idx in range(1, self.number_of_workstations):
                completion_times[p][workstation_idx] = max(
                    completion_times[p][workstation_idx-1] + self.working_hours[product][workstation_idx],
                    completion_times[p-1][workstation_idx] + self.working_hours[product][workstation_idx]
                )
        
        # Calculate total time for normalization
        total_time = completion_times[-1][-1]  # T in paper
        
        # Calculate cumulative production (u_{p,i} in paper)
        cumulative_production = [[0] * self.number_of_product_types for _ in range(len(self.product_sequence))]
        for p in range(len(self.product_sequence)):
            current_product = self.product_sequence[p]
            if p == 0:
                cumulative_production[0][current_product] = 1
            else:
                for i in range(self.number_of_product_types):
                    cumulative_production[p][i] = cumulative_production[p-1][i]
                cumulative_production[p][current_product] += 1
        # Calculate ideal_cumulative_production cumulative production (v_{p,i} in paper)
        total_time = completion_times[-1][-1]  # T in paper
        logistics_leveling = 0.0        
        for p in range(len(self.product_sequence)):
            for i in range(self.number_of_product_types):
                # if self.product_sequence[p] != i:
                #     continue
                # Calculate ideal_cumulative_production cumulative count
                ideal_cumulative_production = (self.product_cycle[i] * completion_times[p][-1]) / total_time
                actual_cumulative_count = cumulative_production[p][i]
                # Calculate absolute deviation
                logistics_leveling += abs(actual_cumulative_count - ideal_cumulative_production)
        
        return logistics_leveling

    def obtain_scores(self) -> None:
        """Calculate all three objective values according to paper model."""
        # 1. Reconfiguration cost (F1)
        reconfiguration_cost = self._calculate_switching_cost()
        
        # 2. Production workload equalization violations (F2)
        part_frequency_violations = self._calculate_workload_equalization()
        
        # 3. Logistics leveling (F3)
        logistics_leveling = self._calculate_logistics_leveling()
        
        self.objectives = [reconfiguration_cost, part_frequency_violations, logistics_leveling]
    
    def get_name(self) -> str:
        """Get the name of the problem."""
        return "MOReconfigurableALBP"


class FloatMOReconfigurableALSP(MOReconfigurableALSPBase):
    """Multi-Objective Reconfigurable ALBP with float solution representation.
    
    This class implements MOReconfigurableALBP using continuous (float) variables
    that are decoded to determine the product sequence. The float variables are
    used to sort and determine the sequence order.
    
    Attributes:
        solution_type (str): Type of solution representation, set to "Float".
        number_of_variables (int): Number of decision variables (sum of product cycles).
        lower_bound (List[float]): Lower bounds for variables (all 0.0).
        upper_bound (List[float]): Upper bounds for variables (all 1.0).
    """
    
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize FloatMOReconfigurableALBP problem.
        
        Args:
            file_path (str): Path to the directory containing the problem instance file.
            file_name (str): Name of the problem instance file to load.
        """
        super(FloatMOReconfigurableALSP, self).__init__(file_path=file_path, file_name=file_name)
        self.solution_type: str = "Float"
        self.number_of_variables: int = sum(self.product_cycle)
        self.lower_bound: List[float] = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound: List[float] = [1.0 for _ in range(self.number_of_variables)]

    def create_variables(self, lower_bound: Optional[List[float]] = None, 
                         upper_bound: Optional[List[float]] = None) -> List[float]:
        """Create random float variables for solution encoding.
        
        Generates random float values within the specified bounds. These values
        are used to determine the product sequence through sorting.
        
        Args:
            lower_bound (Optional[List[float]]): Lower bounds for variables.
                If None, uses self.lower_bound.
            upper_bound (Optional[List[float]]): Upper bounds for variables.
                If None, uses self.upper_bound.
        
        Returns:
            List[float]: A list of random float values within the bounds.
        """
        if lower_bound is None or upper_bound is None:
            variables = [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0)
                         for i in range(self.number_of_variables)]
        else:
            variables = [random.uniform(lower_bound[i] * 1.0, upper_bound[i] * 1.0)
                         for i in range(self.number_of_variables)]
        return variables

    def create_solution(self) -> FloatSolution:
        """Create a new float solution instance.
        
        Generates a new FloatSolution with random variables initialized
        within the problem bounds.
        
        Returns:
            FloatSolution: A new solution with initialized variables.
        """
        new_solution = FloatSolution(
            self.lower_bound, self.upper_bound, 
            self.number_of_variables, 
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        """Evaluate a float solution by decoding and computing objectives.
        
        The evaluation process:
        1. Initializes product sequence based on product cycles
        2. Sorts the sequence based on float variable values (descending order)
        3. Computes objective values using obtain_scores()
        4. Sets solution objectives
        
        Args:
            solution (FloatSolution): The solution to evaluate.
        
        Returns:
            FloatSolution: The evaluated solution with updated objectives.
        """
        solution = self.remedy_solution(solution)
        self.product_sequence = []
        for i in range(0, self.number_of_product_types):
            for j in range(0, self.product_cycle[i]):
                self.product_sequence.append(i)
        task_variables = solution.variables
        for i in range(0, self.number_of_variables - 1):
            for j in range(i + 1, self.number_of_variables):
                if task_variables[i] < task_variables[j]:
                    task_variables[i], task_variables[j] = task_variables[j], task_variables[i]
                    self.product_sequence[i], self.product_sequence[j] = self.product_sequence[j], \
                        self.product_sequence[i]
        self.obtain_scores()
        solution.objectives = copy.deepcopy(self.objectives)
        return solution

    def remedy_solution(self, solution: FloatSolution) -> FloatSolution:
        """Repair solution to ensure variables are within bounds.
        
        Clips any variables that are outside the valid range [lower_bound, upper_bound]
        to the nearest bound.
        
        Args:
            solution (FloatSolution): The solution to repair.
        
        Returns:
            FloatSolution: The repaired solution with all variables in valid range.
        """
        variables = solution.variables
        for i in range(0, len(variables)):
            if variables[i] < self.lower_bound[i]:
                variables[i] = self.lower_bound[i]
            if variables[i] > self.upper_bound[i]:
                variables[i] = self.upper_bound[i]
        solution.variables = variables
        return solution


class IntegerMOReconfigurableALSP(MOReconfigurableALSPBase):
    """Multi-Objective Reconfigurable ALBP with integer solution representation.
    
    This class implements MOReconfigurableALBP using integer variables that directly
    represent product indices in the sequence. Each variable value corresponds to
    a product type.
    
    Attributes:
        solution_type (str): Type of solution representation, set to "Integer".
        number_of_variables (int): Number of decision variables (sum of product cycles).
        lower_bound (List[int]): Lower bounds for variables (all 0).
        upper_bound (List[int]): Upper bounds for variables (product index range).
    """
    
    def __init__(self, file_path: str, file_name: str) -> None:
        """Initialize IntegerMOReconfigurableALBP problem.
        
        Args:
            file_path (str): Path to the directory containing the problem instance file.
            file_name (str): Name of the problem instance file to load.
        """
        super(IntegerMOReconfigurableALSP, self).__init__(file_path=file_path, file_name=file_name)
        self.solution_type: str = "Integer"
        self.number_of_variables: int = sum(self.product_cycle)
        self.lower_bound: List[int] = self.number_of_variables * [0]
        self.upper_bound: List[int] = self.number_of_variables * [self.number_of_product_types - 1]

    def create_variables(self, lower_bound: Optional[List[int]] = None, 
                         upper_bound: Optional[List[int]] = None) -> List[int]:
        """Create random integer variables representing a valid product sequence.
        
        Generates a random permutation of products that satisfies the cycle
        requirements (each product appears according to its cycle count).
        
        Args:
            lower_bound (Optional[List[int]]): Lower bounds (unused, kept for interface compatibility).
            upper_bound (Optional[List[int]]): Upper bounds (unused, kept for interface compatibility).
        
        Returns:
            List[int]: A random permutation of product indices satisfying cycle constraints.
        """
        variables = []
        for i in range(0, self.number_of_product_types):
            for j in range(0, self.product_cycle[i]):
                variables.append(i)
        variables = random.sample(variables, len(variables))
        return variables

    def create_solution(self) -> IntegerSolution:
        """Create a new integer solution instance.
        
        Generates a new IntegerSolution with random variables initialized
        as a valid product sequence permutation.
        
        Returns:
            IntegerSolution: A new solution with initialized variables.
        """
        new_solution = IntegerSolution(
            self.lower_bound, self.upper_bound, 
            self.number_of_variables, 
            self.number_of_objectives,
            self.number_of_constraints
        )
        new_solution.variables = self.create_variables()
        return new_solution

    def evaluate_solution(self, solution: IntegerSolution) -> IntegerSolution:
        """Evaluate an integer solution by computing objectives.
        
        The evaluation process:
        1. Uses the integer variables directly as the product sequence
        2. Computes objective values using obtain_scores()
        3. Sets solution objectives
        
        Args:
            solution (IntegerSolution): The solution to evaluate.
        
        Returns:
            IntegerSolution: The evaluated solution with updated objectives.
        """
        self.product_sequence = solution.variables
        self.obtain_scores()
        solution.objectives = copy.deepcopy(self.objectives)
        return solution
