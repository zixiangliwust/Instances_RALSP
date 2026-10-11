"""
Mixed-integer programming models for Multi-Objective Reconfigurable Assembly Line Balancing Problem.

This module provides mathematical programming models for solving the 
Multi-Objective Reconfigurable Assembly Line Balancing Problem using exact optimization methods.

The models use CPLEX (via docplex) to solve the MILP formulations for three objectives:
1. Minimizing reconfiguration cost (Model 1)
2. Minimizing part frequency constraint violations (Model 2)  
3. Minimizing production logistics leveling (Model 3)

Module structure:
    ReconfigurableALBPModel: Class containing three MILP models
"""

import sys
import os

# Add CPLEX path to sys.path only if it's not already there
cplex_path = r'D:\Cplex\cplex\python\3.10\x64_win64'
if cplex_path not in sys.path:
    sys.path.insert(0, cplex_path)

import time
import math
from typing import Any, Optional, Tuple, List, Dict
from docplex.mp.model import Model


class ReconfigurableALSPModel:
    """
    MILP model for Multi-Objective Reconfigurable Assembly Line Balancing Problem.
    
    This class contains three separate MILP models for solving the three objectives
    of the reconfigurable assembly line balancing problem.
    
    Attributes:
        problem: An instance of MOReconfigurableALSPBase containing problem data
        max_seconds (int): Maximum time limit for each model
    """
    
    def __init__(self, problem, max_seconds: int = 300):
        """Initialize the model with problem instance.
        
        Args:
            problem: An instance of MOReconfigurableALSPBase containing problem data
            max_seconds (int): Maximum time limit for each model (default: 300 seconds)
        """
        self.problem = problem
        self.max_seconds = max_seconds
        
        # Extract parameters from problem instance
        self.number_of_product_types = problem.number_of_product_types
        self.number_of_workstations = problem.number_of_workstations
        self.number_of_parts = problem.number_of_parts
        self.product_cycle = problem.product_cycle
        self.total_number_of_products = sum(self.product_cycle)
        self.product_switching_cost = problem.product_switching_cost
        self.part_requirement_matrix = problem.part_requirement_matrix
        
        # Part frequency constraints
        self.part_frequency_max_usage = []
        self.part_frequency_window = []
        for part_constraint in problem.part_frequency_constraints:
            self.part_frequency_max_usage.append(part_constraint[0])
            self.part_frequency_window.append(part_constraint[1])
        
        self.working_hours = problem.working_hours
        self.interval_time = getattr(problem, 'interval_time', 5)
        self.big_M_parameter = 10000
    
    def solve_model1(self) -> Dict[str, Any]:
        """Solve Model 1: Minimizing reconfiguration cost.
        
        Returns:
            Dict containing solution details: objective, status, time, gap, solution
        """
        print("=" * 60)
        print("Solving Model 1: Reconfiguration Cost Minimization")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            model = Model(name="Reconfiguration_Cost_Minimization")
            
            # Decision variables (0-based indexing)
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x')
            y = model.binary_var_cube(range(self.total_number_of_products), range(self.number_of_product_types), range(self.number_of_product_types), name='y')
            
            # Objective: Minimize reconfiguration cost
            objective_expr = 0
            for position in range(self.total_number_of_products):
                for product_i in range(self.number_of_product_types):
                    for product_j in range(self.number_of_product_types):
                        objective_expr += y[position, product_i, product_j] * self.product_switching_cost[product_i][product_j]
            
            model.minimize(objective_expr)
            
            # Constraints
            # Each position has exactly one product
            for position in range(self.total_number_of_products):
                model.add(model.sum(x[position, product] for product in range(self.number_of_product_types)) == 1, 
                         name=f"one_product_per_position_{position}")
            
            # Meet product demand requirements
            for product in range(self.number_of_product_types):
                model.add(model.sum(x[position, product] for position in range(self.total_number_of_products)) == self.product_cycle[product],
                         name=f"demand_product_{product}")
            
            # Linearize x and y relationship for consecutive positions
            for position in range(self.total_number_of_products - 1):
                for product_i in range(self.number_of_product_types):
                    for product_j in range(self.number_of_product_types):
                        model.add(y[position, product_i, product_j] <= x[position, product_i], 
                                 name=f"y_leq_x1_{position}_{product_i}_{product_j}")
                        model.add(y[position, product_i, product_j] <= x[position + 1, product_j], 
                                 name=f"y_leq_x2_{position}_{product_i}_{product_j}")
                        model.add(x[position, product_i] + x[position + 1, product_j] - 1 <= y[position, product_i, product_j], 
                                 name=f"y_geq_sum_{position}_{product_i}_{product_j}")
            
            # Cycle constraints (last to first)
            for product_i in range(self.number_of_product_types):
                for product_j in range(self.number_of_product_types):
                    model.add(y[self.total_number_of_products - 1, product_i, product_j] <= x[self.total_number_of_products - 1, product_i], 
                             name=f"cycle_y_leq_x1_{product_i}_{product_j}")
                    model.add(y[self.total_number_of_products - 1, product_i, product_j] <= x[0, product_j], 
                             name=f"cycle_y_leq_x2_{product_i}_{product_j}")
                    model.add(x[self.total_number_of_products - 1, product_i] + x[0, product_j] - 1 <= y[self.total_number_of_products - 1, product_i, product_j], 
                             name=f"cycle_y_geq_sum_{product_i}_{product_j}")
            
            # Solve
            model.set_time_limit(self.max_seconds)
            model.parameters.mip.tolerances.mipgap = 0.00
            
            solution = model.solve(log_output=True)
            solve_details = model.get_solve_details()
            solve_time = time.time() - start_time
            
            result = {
                'objective': solution.objective_value if solution else math.pow(10, 8),
                'status': solve_details.status,
                'time': solve_time,
                'gap': getattr(solve_details, 'gap', None),
                'solution': solution
            }
            
            print(f"Model 1 - Optimal reconfiguration cost: {result['objective']}")
            print(f"Model 1 - Solve status: {solve_details.status}")
            print(f"Model 1 - Solve time: {solve_time:.2f} seconds")
            if result['gap'] is not None:
                print(f"Model 1 - Optimality gap: {result['gap']:.4f}")
            
            if solution:
                reconf_cost, violations, leveling, product_sequence = self.decode_all_objectives(solution)
                print(f"Model 1 - Decoded reconfiguration cost: {reconf_cost}")
                print(f"Model 1 - Decoded violations: {violations}")
                print(f"Model 1 - Decoded logistics leveling: {leveling}")
                print(f"Model 1 - Difference: {abs(result['objective'] - reconf_cost)}")
                result['product_sequence'] = product_sequence
            
            return result
            
        except Exception as e:
            print(f"Error solving Model 1: {e}")
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'time': time.time() - start_time,
                'gap': None,
                'solution': None,
                'error': str(e)
            }
    
    def solve_model2(self) -> Dict[str, Any]:
        """
        Solve Model 2: Minimizing part frequency constraint violations.
        
        This model formulates and solves the part frequency constraint violation
        minimization problem as a mixed-integer linear program.
        """
        print("\n" + "=" * 60)
        print("Solving Model 2: Part Frequency Violation Minimization")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Create CPLEX model
            model = Model(name="Part_Frequency_Violation_Minimization")
            
            # Decision variables
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x')
            q = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_parts), name='q')
            
            # Objective: Minimize total constraint violations
            objective_expr = model.sum(q[position, part] for position in range(self.total_number_of_products) for part in range(self.number_of_parts))
            model.minimize(objective_expr)
            
            # Constraints
            
            # Constraint (12): Each position has exactly one product
            for position in range(self.total_number_of_products):
                model.add(model.sum(x[position, product] for product in range(self.number_of_product_types)) == 1,
                        name=f"one_product_per_position_{position}")
            
            # Constraint (13): Meet product demand requirements
            for product in range(self.number_of_product_types):
                model.add(model.sum(x[position, product] for position in range(self.total_number_of_products)) == self.product_cycle[product],
                        name=f"demand_product_{product}")
            
            # Constraints for part frequency checking
            
            for position in range(self.total_number_of_products):
                for part in range(self.number_of_parts):
                    B_l = self.part_frequency_window[part]
                    
                    # Calculate part usage in the sliding window of size B_l
                    usage_expr = 0
                    for window_offset in range(B_l):
                        pos_idx = (position + window_offset) % self.total_number_of_products
                        for product in range(self.number_of_product_types):
                            if self.part_requirement_matrix[product][part] >= 1:
                                usage_expr += x[pos_idx, product]
                    
                    # Constraint 1: If usage_expr > A[part], then q must be 1
                    model.add(usage_expr - self.part_frequency_max_usage[part] <= self.big_M_parameter * q[position, part],
                            name=f"violation_force_q1_{position}_{part}")
                    
                    # Constraint 2: If usage_expr <= A[part], then q must be 0
                    # Note: Add a small epsilon to handle integer equality cases
                    model.add(self.part_frequency_max_usage[part] - usage_expr <= self.big_M_parameter * (1 - q[position, part]),
                            name=f"violation_force_q0_{position}_{part}")
            
            # Solve the model
            model.set_time_limit(self.max_seconds)
            model.parameters.mip.tolerances.mipgap = 0.00
            
            solution = model.solve(log_output=True)
            solve_details = model.get_solve_details()
            
            solve_time = time.time() - start_time
            
            # Prepare result dictionary
            result = {
                'objective': solution.objective_value if solution else math.pow(10, 8),
                'status': solve_details.status,
                'time': solve_time,
                'gap': getattr(solve_details, 'gap', None),
                'solution': solution
            }
            
            print(f"Model 2 - Optimal constraint violations: {result['objective']}")
            print(f"Model 2 - Solve status: {solve_details.status}")
            print(f"Model 2 - Solve time: {solve_time:.2f} seconds")
            if result['gap'] is not None:
                print(f"Model 2 - Optimality gap: {result['gap']:.4f}")
            
            # Decode the solution using algorithm logic to get all objective values
            if solution:
                reconf_cost, violations, leveling, product_sequence = self.decode_all_objectives(solution)
                print(f"Model 2 - Decoded reconfiguration cost using algorithm logic: {reconf_cost}")
                print(f"Model 2 - Decoded violations using algorithm logic: {violations}")
                print(f"Model 2 - Decoded logistics leveling using algorithm logic: {leveling}")
                print(f"Model 2 - Difference between model and algorithm: {abs(result['objective'] - violations)}")
                
                # Add product sequence to the result
                result['product_sequence'] = product_sequence
            
            return result
            
        except Exception as e:
            print(f"Error solving Model 2: {e}")
            import traceback
            traceback.print_exc()
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'time': time.time() - start_time,
                'gap': None,
                'solution': None,
                'error': str(e)
            }

    def solve_model3(self) -> Dict[str, Any]:
        """Solve Model 3: Minimizing logistics leveling.
                
        Uses epsilon-constraint method: first minimize total completion time, then fix it to solve logistics leveling.
        """
        print("\n" + "=" * 60)
        print("Solving Model 3: Logistics Leveling Minimization (Epsilon-Constraint Method)")
        print("=" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Minimize total completion time
            print("\nStep 1: Minimizing total completion time...")
            makespan_opt, solution_step1 = self._solve_minimize_total_completion_time()
            
            if makespan_opt is None or makespan_opt <= 0:
                print("Error: Could not find feasible total completion time")
                return {
                    'objective': math.pow(10, 8),
                    'status': 'error',
                    'time': time.time() - start_time,
                    'gap': None,
                    'solution': None,
                    'error': 'No feasible total completion time found'
                }
            
            print(f"Optimal total completion time makespan* = {makespan_opt}")
            
            # Step 2: Fix makespan and minimize logistics leveling
            print("\nStep 2: Minimizing logistics leveling with fixed makespan...")
            result = self._solve_logistics_leveling_fixed_makespan(makespan_opt)
            
            # Add step 1 solve time
            result['time'] = time.time() - start_time
            
            # Decode and verify
            if result['solution']:
                reconf_cost, violations, leveling, product_sequence = self.decode_all_objectives(result['solution'])
                print(f"Model 3 - Decoded reconfiguration cost: {reconf_cost}")
                print(f"Model 3 - Decoded violations: {violations}")
                print(f"Model 3 - Decoded logistics leveling: {leveling}")
                print(f"Model 3 - Difference: {abs(result['objective'] - leveling)}")
                
                result['product_sequence'] = product_sequence
            
            return result
            
        except Exception as e:
            print(f"Error solving Model 3: {e}")
            import traceback
            traceback.print_exc()
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'time': time.time() - start_time,
                'gap': None,
                'solution': None,
                'error': str(e)
            }

    def _solve_minimize_total_completion_time(self):
        """Solve subproblem: minimize total completion time."""
        try:
            model = Model(name="Minimize_Total_Completion_Time")
            
            # Variables
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x_T')
            t = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_workstations), lb=0, name='t_T')
            makespan = model.continuous_var(lb=0, name='makespan')
            
            # Constraints: Each position has exactly one product
            for position in range(self.total_number_of_products):
                model.add(model.sum(x[position, product] for product in range(self.number_of_product_types)) == 1,
                        name=f"T_one_product_per_position_{position}")
            
            # Constraints: Meet product demand requirements
            for product in range(self.number_of_product_types):
                model.add(model.sum(x[position, product] for position in range(self.total_number_of_products)) == self.product_cycle[product],
                        name=f"T_demand_product_{product}")
            
            # Completion time constraints
            # First product (position 0) in first station (station 0)
            model.add(t[0, 0] == model.sum(x[0, product] * self.working_hours[product][0] for product in range(self.number_of_product_types)),
                    name="T_first_product_first_station")
            
            # Other products (positions 1 to nd-1) in first station (station 0)
            for position in range(1, self.total_number_of_products):
                model.add(t[position, 0] == t[position-1, 0] + self.interval_time + 
                        model.sum(x[position, product] * self.working_hours[product][0] for product in range(self.number_of_product_types)),
                        name=f"T_product_{position}_first_station")
            
            # All products and stations precedence constraints
            for position in range(self.total_number_of_products):
                for station in range(1, self.number_of_workstations):
                    # From previous station for same product
                    model.add(t[position, station] >= t[position, station-1] + 
                            model.sum(x[position, product] * self.working_hours[product][station] for product in range(self.number_of_product_types)),
                            name=f"T_station_precedence_{position}_{station}")
            
            # From previous product for same station
            for position in range(1, self.total_number_of_products):
                for station in range(self.number_of_workstations):
                    model.add(t[position, station] >= t[position-1, station] + 
                            model.sum(x[position, product] * self.working_hours[product][station] for product in range(self.number_of_product_types)),
                            name=f"T_product_precedence_{position}_{station}")
            
            # Total completion time
            model.add(makespan == t[self.total_number_of_products - 1, self.number_of_workstations - 1], name="T_total_completion_time")
            
            # Objective: Minimize total completion time
            model.minimize(makespan)
            
            model.set_time_limit(self.max_seconds / 2)  # Allocate half time for this step
            model.parameters.mip.tolerances.mipgap = 0.00
            
            solution = model.solve(log_output=False)
            
            if solution:
                return solution.objective_value, solution
            else:
                return None, None
                
        except Exception as e:
            print(f"Error in _solve_minimize_total_completion_time: {e}")
            return None, None

    def _solve_logistics_leveling_fixed_makespan(self, makespan_fixed):
        """Solve logistics leveling with fixed total completion time makespan_fixed."""
        try:
            model = Model(name="Logistics_Leveling_Fixed_T")
                
            # Variables
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x')
            t = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_workstations), lb=0, name='t')
            u = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='u')
            v = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='v')
            delta_plus = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='delta_plus')
            delta_minus = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='delta_minus')
            
            # Constraints: Each position has exactly one product
            for position in range(self.total_number_of_products):
                model.add(model.sum(x[position, product] for product in range(self.number_of_product_types)) == 1,
                        name=f"one_product_per_position_{position}")
            
            # Constraints: Meet product demand requirements
            for product in range(self.number_of_product_types):
                model.add(model.sum(x[position, product] for position in range(self.total_number_of_products)) == self.product_cycle[product],
                        name=f"demand_product_{product}")
            
            # Completion time constraints
            # First product (position 0) in first station (station 0)
            model.add(t[0, 0] == model.sum(x[0, product] * self.working_hours[product][0] for product in range(self.number_of_product_types)),
                    name="first_product_first_station")
            
            # Other products (positions 1 to nd-1) in first station (station 0)
            for position in range(1, self.total_number_of_products):
                model.add(t[position, 0] == t[position-1, 0] + self.interval_time + 
                        model.sum(x[position, product] * self.working_hours[product][0] for product in range(self.number_of_product_types)),
                        name=f"product_{position}_first_station")
            
            # All products and stations precedence constraints
            for position in range(self.total_number_of_products):
                for station in range(1, self.number_of_workstations):
                    # From previous station for same product
                    model.add(t[position, station] >= t[position, station-1] + 
                            model.sum(x[position, product] * self.working_hours[product][station] for product in range(self.number_of_product_types)),
                            name=f"station_precedence_{position}_{station}")
            
            # From previous product for same station
            for position in range(1, self.total_number_of_products):
                for station in range(self.number_of_workstations):
                    model.add(t[position, station] >= t[position-1, station] + 
                            model.sum(x[position, product] * self.working_hours[product][station] for product in range(self.number_of_product_types)),
                            name=f"product_precedence_{position}_{station}")
            
            # Cumulative production
            for product in range(self.number_of_product_types):
                model.add(u[0, product] == x[0, product], name=f"cumulative_first_{product}")
                            
            for position in range(1, self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    model.add(u[position, product] == u[position-1, product] + x[position, product],
                            name=f"cumulative_{position}_{product}")
            
            # Fixed total completion time constraint
            model.add(t[self.total_number_of_products - 1, self.number_of_workstations - 1] == makespan_fixed, name="fixed_total_completion_time")
            
            # Ideal cumulative production: v[p,i] = d_i * t[p,nk] / makespan_fixed
            # CORRECTED: use last workstation index (number_of_workstations - 1) instead of number_of_workstations
            last_station_idx = self.number_of_workstations - 1
            for position in range(self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    model.add(v[position, product] == self.product_cycle[product] * t[position, last_station_idx] / makespan_fixed,
                            name=f"ideal_cumulative_{position}_{product}")
            
            # Deviation calculation
            for position in range(self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    model.add(u[position, product] - v[position, product] == delta_plus[position, product] - delta_minus[position, product],
                            name=f"deviation_{position}_{product}")
            
            # Objective: Minimize total deviation
            objective_expr = model.sum(delta_plus[position, product] + delta_minus[position, product] 
                                    for position in range(self.total_number_of_products) for product in range(self.number_of_product_types))
            model.minimize(objective_expr)
            
            # Solve the model
            model.set_time_limit(self.max_seconds / 2)
            model.parameters.mip.tolerances.mipgap = 0.00
            
            solution = model.solve(log_output=True)
            solve_details = model.get_solve_details()
            
            result = {
                'objective': solution.objective_value if solution else math.pow(10, 8),
                'status': solve_details.status,
                'time': 0,
                'gap': getattr(solve_details, 'gap', None),
                'solution': solution,
                'makespan_fixed': makespan_fixed
            }
            
            print(f"Model 3 - Optimal logistics leveling with makespan={makespan_fixed}: {result['objective']}")
            print(f"Model 3 - Solve status: {solve_details.status}")
            
            return result
            
        except Exception as e:
            print(f"Error in _solve_logistics_leveling_fixed_makespan: {e}")
            import traceback
            traceback.print_exc()
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'time': 0,
                'gap': None,
                'solution': None,
                'makespan_fixed': makespan_fixed,
                'error': str(e)
            }
        
    def decode_all_objectives(self, model_solution):
        """
        Decode the model solution to calculate all three objectives using algorithm logic.
        
        Args:
            model_solution: Solution object from CPLEX model
            
        Returns:
            Tuple of (reconfiguration_cost, part_frequency_violations, logistics_leveling, product_sequence)
        """
        if not model_solution:
            return float('inf'), float('inf'), float('inf'), []
        
        try:
            # Reconstruct the product sequence from the x variables (used for all objectives)
            product_sequence = []
            for position in range(self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    try:
                        x_val = model_solution.get_value(f'x_{position}_{product}')
                    except:
                        # Try alternative naming
                        try:
                            x_val = model_solution.get_value(f'x[{position},{product}]')
                        except:
                            x_val = None
                    
                    if x_val is not None and x_val > 0.5:
                        product_sequence.append(product)
                        break
                else:
                    product_sequence.append(0)  # Default if no product found
            
            # Calculate reconfiguration cost (Model 1 objective)
            reconfiguration_cost = 0.0
            for position in range(len(product_sequence) - 1):
                current_product = product_sequence[position]
                next_product = product_sequence[position + 1]
                reconfiguration_cost += self.product_switching_cost[current_product][next_product]
            
            # Calculate cost from last to first position (cycle)
            if len(product_sequence) > 0:
                last_product = product_sequence[-1]
                first_product = product_sequence[0]
                reconfiguration_cost += self.product_switching_cost[last_product][first_product]
            
            # Calculate part frequency violations (Model 2 objective)
            part_frequency_violations = 0.0
            
            # Check each part's frequency constraint
            for l in range(self.number_of_parts):
                A_l, B_l = self.part_frequency_max_usage[l], self.part_frequency_window[l]
                
                # Check sliding window of size B_l
                for start_idx in range(len(product_sequence)):
                    # Count how many products in this window require part l
                    count = 0
                    for offset in range(B_l):
                        # Calculate position index with cyclic wrap-around
                        pos_idx = start_idx + offset
                        if pos_idx >= len(product_sequence):
                            pos_idx = pos_idx - len(product_sequence)
                        product_idx = product_sequence[pos_idx]
                        if self.part_requirement_matrix[product_idx][l] >= 1:  # Using >=1 as in original
                            count += 1
                    
                    # If count exceeds A_l, it's a violation
                    if count > A_l:
                        part_frequency_violations += 1
            
            # Calculate logistics leveling (Model 3 objective)
            # Initialize arrays to track workstation completion times and cumulative production
            completion_times = [[0.0] * self.number_of_workstations for _ in range(len(product_sequence))]
            
            # Calculate completion times for the first product
            first_product = product_sequence[0]
            completion_times[0][0] = self.working_hours[first_product][0]
            for workstation_idx in range(1, self.number_of_workstations):
                completion_times[0][workstation_idx] = completion_times[0][workstation_idx-1] + self.working_hours[first_product][workstation_idx]
            
            # Calculate completion times for remaining products using max logic
            for p in range(1, len(product_sequence)):
                product = product_sequence[p]
                
                # First workstation: add previous workstation time, interval time, and processing time
                completion_times[p][0] = completion_times[p-1][0] + self.interval_time + self.working_hours[product][0]
                
                # Other workstations: use max logic to determine earliest possible start time
                for workstation_idx in range(1, self.number_of_workstations):
                    completion_times[p][workstation_idx] = max(
                        completion_times[p][workstation_idx-1] + self.working_hours[product][workstation_idx],
                        completion_times[p-1][workstation_idx] + self.working_hours[product][workstation_idx]
                    )
            
            # Calculate total time for normalization
            total_time = completion_times[-1][-1]  # makespan in paper
            
            # Calculate cumulative production (u_{p,i} in paper)
            cumulative_production = [[0] * self.number_of_product_types for _ in range(len(product_sequence))]
            for p in range(len(product_sequence)):
                current_product = product_sequence[p]
                if p == 0:
                    cumulative_production[0][current_product] = 1
                else:
                    for i in range(self.number_of_product_types):
                        cumulative_production[p][i] = cumulative_production[p-1][i]
                    cumulative_production[p][current_product] += 1
            # Calculate ideal cumulative production (v_{p,i} in paper)
            logistics_leveling = 0.0        
            for p in range(len(product_sequence)):
                for i in range(self.number_of_product_types):
                    # Calculate ideal cumulative production count
                    ideal_cumulative_production = (self.product_cycle[i] * completion_times[p][-1]) / total_time
                    actual_cumulative_count = cumulative_production[p][i]
                    # Calculate absolute deviation
                    logistics_leveling += abs(actual_cumulative_count - ideal_cumulative_production)
            
            return reconfiguration_cost, part_frequency_violations, logistics_leveling, product_sequence
        except Exception as e:
            print(f"Error in decode_all_objectives: {e}")
            import traceback
            traceback.print_exc()
            return float('inf'), float('inf'), float('inf'), []

    def solve_epsilon_constraint_method(self, primary_objective: str = 'model1', epsilon_values: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Solve the multi-objective problem using the epsilon-constraint method.
        
        The epsilon-constraint method optimizes one primary objective while constraining others:
        min F_1 subject to F_2 ≤ ϵ_2 and F_3 ≤ ϵ_3, generating Pareto-optimal solutions through parameter variation.
        
        For primary_objective == 'model3', this method uses a two-stage approach:
        1. First minimize total completion time to obtain T_min.
        2. Then fix T = T_min and solve logistics leveling minimization with additional epsilon constraints.
        
        Args:
            primary_objective (str): The primary objective to optimize ('model1', 'model2', or 'model3')
            epsilon_values (Dict[str, float]): Dictionary specifying epsilon constraints for non-primary objectives
                                              e.g., {'model2': 5.0, 'model3': 10.0}
        
        Returns:
            Dict containing solution details with the primary objective minimized subject to epsilon constraints
        """
        print("=" * 60)
        print(f"Solving using Epsilon-Constraint Method")
        print(f"Primary objective: {primary_objective}")
        if epsilon_values:
            print(f"Epsilon constraints: {epsilon_values}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Special handling for primary_objective == 'model3' (two-stage approach)
        if primary_objective == 'model3':
            # Step 1: Minimize total completion time
            makespan_opt, _ = self._solve_minimize_total_completion_time()
            if makespan_opt is None or makespan_opt <= 0:
                print("Error: Could not find feasible total completion time for epsilon-constraint method")
                return {
                    'objective': math.pow(10, 8),
                    'status': 'error',
                    'time': time.time() - start_time,
                    'gap': None,
                    'solution': None,
                    'primary_objective': primary_objective,
                    'epsilon_values': epsilon_values,
                    'error': 'No feasible total completion time found'
                }
            
            # Step 2: Solve logistics leveling with fixed makespan and epsilon constraints
            result = self._solve_logistics_leveling_with_epsilon(makespan_opt, epsilon_values)
            result['time'] = time.time() - start_time
            result['primary_objective'] = primary_objective
            result['epsilon_values'] = epsilon_values
            if result.get('solution'):
                reconf_cost, violations, leveling, product_sequence = self.decode_all_objectives(result['solution'])
                result['decoded_objectives'] = {
                    'reconfiguration_cost': reconf_cost,
                    'part_frequency_violations': violations,
                    'logistics_leveling': leveling
                }
                result['product_sequence'] = product_sequence
            return result
        
        # For primary_objective in ['model1','model2'] with epsilon containing 'model3' (two-stage approach)
        if epsilon_values and 'model3' in epsilon_values and epsilon_values['model3'] != float('inf'):
            makespan_opt, _ = self._solve_minimize_total_completion_time()
            if makespan_opt is None or makespan_opt <= 0:
                print("Error: Could not find feasible total completion time for epsilon-constraint method")
                return {
                    'objective': math.pow(10, 8),
                    'status': 'error',
                    'time': time.time() - start_time,
                    'gap': None,
                    'solution': None,
                    'primary_objective': primary_objective,
                    'epsilon_values': epsilon_values,
                    'error': 'No feasible total completion time found'
                }
            result = self._solve_primary_with_epsilon_constraints(primary_objective, makespan_opt, epsilon_values)
            result['time'] = time.time() - start_time
            result['primary_objective'] = primary_objective
            result['epsilon_values'] = epsilon_values
            if result.get('solution'):
                reconf_cost, violations, leveling, product_sequence = self.decode_all_objectives(result['solution'])
                result['decoded_objectives'] = {
                    'reconfiguration_cost': reconf_cost,
                    'part_frequency_violations': violations,
                    'logistics_leveling': leveling
                }
                result['product_sequence'] = product_sequence
            return result
        
        # For primary_objective == 'model1' or 'model2' without model3 constraint
        try:
            # Set default epsilon values if not provided
            if epsilon_values is None:
                epsilon_values = {}
            
            # Create CPLEX model based on the primary objective
            model = Model(name=f"Epsilon_Constraint_{primary_objective}")
            
            # Decision variables
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x')
            if primary_objective == 'model1' or 'model1' in epsilon_values:
                y = model.binary_var_cube(range(self.total_number_of_products), range(self.number_of_product_types), range(self.number_of_product_types), name='y')
            if primary_objective == 'model2' or 'model2' in epsilon_values:
                q = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_parts), name='q')
            
            # Add constraints that are common to all models
            for position in range(self.total_number_of_products):
                model.add(model.sum(x[position, product] for product in range(self.number_of_product_types)) == 1, 
                         name=f"one_product_per_position_{position}")
            for product in range(self.number_of_product_types):
                model.add(model.sum(x[position, product] for position in range(self.total_number_of_products)) == self.product_cycle[product],
                         name=f"demand_product_{product}")
            
            # Add y linearization if needed
            if primary_objective == 'model1' or 'model1' in epsilon_values:
                for position in range(self.total_number_of_products - 1):
                    for i in range(self.number_of_product_types):
                        for j in range(self.number_of_product_types):
                            model.add(y[position, i, j] <= x[position, i], 
                                     name=f"y_leq_x1_{position}_{i}_{j}")
                            model.add(y[position, i, j] <= x[position+1, j], 
                                     name=f"y_leq_x2_{position}_{i}_{j}")
                            model.add(x[position, i] + x[position+1, j] - 1 <= y[position, i, j], 
                                     name=f"y_geq_sum_{position}_{i}_{j}")
                for i in range(self.number_of_product_types):
                    for j in range(self.number_of_product_types):
                        model.add(y[self.total_number_of_products-1, i, j] <= x[self.total_number_of_products-1, i], 
                                 name=f"cycle_y_leq_x1_{i}_{j}")
                        model.add(y[self.total_number_of_products-1, i, j] <= x[0, j], 
                                 name=f"cycle_y_leq_x2_{i}_{j}")
                        model.add(x[self.total_number_of_products-1, i] + x[0, j] - 1 <= y[self.total_number_of_products-1, i, j], 
                                 name=f"cycle_y_geq_sum_{i}_{j}")
            
            # Add q constraints if needed
            if primary_objective == 'model2' or 'model2' in epsilon_values:
                for position in range(self.total_number_of_products):
                    for part in range(self.number_of_parts):
                        B_l = self.part_frequency_window[part]
                        usage_expr = 0
                        for offset in range(B_l):
                            pos_idx = (position + offset) % self.total_number_of_products
                            for product in range(self.number_of_product_types):
                                if self.part_requirement_matrix[product][part] >= 1:
                                    usage_expr += x[pos_idx, product]
                        model.add(usage_expr - self.part_frequency_max_usage[part] <= self.big_M_parameter * q[position, part],
                                 name=f"violation_force_q1_{position}_{part}")
                        model.add(self.part_frequency_max_usage[part] - usage_expr <= self.big_M_parameter * (1 - q[position, part]),
                                 name=f"violation_force_q0_{position}_{part}")
            
            # Define objective
            if primary_objective == 'model1':
                cost_expr = 0
                for pos in range(self.total_number_of_products):
                    for i in range(self.number_of_product_types):
                        for j in range(self.number_of_product_types):
                            cost_expr += y[pos, i, j] * self.product_switching_cost[i][j]
                model.minimize(cost_expr)
            elif primary_objective == 'model2':
                violation_expr = model.sum(q[pos, part] for pos in range(self.total_number_of_products) for part in range(self.number_of_parts))
                model.minimize(violation_expr)
            else:
                raise ValueError(f"Unsupported primary objective: {primary_objective}")
            
            # Add epsilon constraints for non-primary objectives
            for obj_name, eps_val in epsilon_values.items():
                if obj_name == primary_objective or eps_val == float('inf'):
                    continue
                if obj_name == 'model1':
                    cost_expr = model.sum(y[pos, i, j] * self.product_switching_cost[i][j] 
                                        for pos in range(self.total_number_of_products)
                                        for i in range(self.number_of_product_types)
                                        for j in range(self.number_of_product_types))
                    model.add(cost_expr <= eps_val, name=f"epsilon_constraint_model1")
                elif obj_name == 'model2':
                    violation_expr = model.sum(q[pos, part] for pos in range(self.total_number_of_products) for part in range(self.number_of_parts))
                    model.add(violation_expr <= eps_val, name=f"epsilon_constraint_model2")
                # model3 is handled earlier via two-stage method
            
            # Solve
            model.set_time_limit(self.max_seconds)
            model.parameters.mip.tolerances.mipgap = 0.00
            solution = model.solve(log_output=True)
            solve_details = model.get_solve_details()
            solve_time = time.time() - start_time
            
            result = {
                'objective': solution.objective_value if solution else math.pow(10, 8),
                'status': solve_details.status,
                'time': solve_time,
                'gap': getattr(solve_details, 'gap', None),
                'solution': solution,
                'primary_objective': primary_objective,
                'epsilon_values': epsilon_values
            }
            
            print(f"Epsilon-Constraint Method - Optimal {primary_objective} value: {result['objective']}")
            print(f"Epsilon-Constraint Method - Solve status: {solve_details.status}")
            print(f"Epsilon-Constraint Method - Solve time: {solve_time:.2f} seconds")
            if result['gap'] is not None:
                print(f"Epsilon-Constraint Method - Optimality gap: {result['gap']:.4f}")
            
            if solution:
                reconf_cost, violations, leveling, product_sequence = self.decode_all_objectives(solution)
                result['decoded_objectives'] = {
                    'reconfiguration_cost': reconf_cost,
                    'part_frequency_violations': violations,
                    'logistics_leveling': leveling
                }
                result['product_sequence'] = product_sequence
            
            return result
            
        except Exception as e:
            print(f"Error solving Epsilon-Constraint Method: {e}")
            import traceback
            traceback.print_exc()
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'time': time.time() - start_time,
                'gap': None,
                'solution': None,
                'primary_objective': primary_objective,
                'epsilon_values': epsilon_values,
                'error': str(e)
            }

    def _solve_logistics_leveling_with_epsilon(self, makespan_fixed, epsilon_values):
        """
        Solve logistics leveling with fixed makespan and additional epsilon constraints.
        
        This is a helper method for epsilon-constraint when primary_objective == 'model3'.
        """
        try:
            model = Model(name="Logistics_Leveling_With_Epsilon")
                
            # Variables
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x')
            t = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_workstations), lb=0, name='t')
            u = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='u')
            v = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='v')
            delta_plus = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='delta_plus')
            delta_minus = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='delta_minus')
            
            # For epsilon constraints on model1 and model2, we need additional variables
            if epsilon_values and ('model1' in epsilon_values or 'model2' in epsilon_values):
                y = model.binary_var_cube(range(self.total_number_of_products), range(self.number_of_product_types), range(self.number_of_product_types), name='y')
                q = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_parts), name='q')
            
            # Common constraints: each position has exactly one product
            for position in range(self.total_number_of_products):
                model.add(model.sum(x[position, product] for product in range(self.number_of_product_types)) == 1,
                        name=f"one_product_per_position_{position}")
            
            # Meet product demand requirements
            for product in range(self.number_of_product_types):
                model.add(model.sum(x[position, product] for position in range(self.total_number_of_products)) == self.product_cycle[product],
                        name=f"demand_product_{product}")
            
            # Completion time constraints (same as in _solve_logistics_leveling_fixed_makespan)
            model.add(t[0, 0] == model.sum(x[0, product] * self.working_hours[product][0] for product in range(self.number_of_product_types)),
                    name="first_product_first_station")
            for position in range(1, self.total_number_of_products):
                model.add(t[position, 0] == t[position-1, 0] + self.interval_time + 
                        model.sum(x[position, product] * self.working_hours[product][0] for product in range(self.number_of_product_types)),
                        name=f"product_{position}_first_station")
            for position in range(self.total_number_of_products):
                for station in range(1, self.number_of_workstations):
                    model.add(t[position, station] >= t[position, station-1] + 
                            model.sum(x[position, product] * self.working_hours[product][station] for product in range(self.number_of_product_types)),
                            name=f"station_precedence_{position}_{station}")
            for position in range(1, self.total_number_of_products):
                for station in range(self.number_of_workstations):
                    model.add(t[position, station] >= t[position-1, station] + 
                            model.sum(x[position, product] * self.working_hours[product][station] for product in range(self.number_of_product_types)),
                            name=f"product_precedence_{position}_{station}")
            
            # Cumulative production
            for product in range(self.number_of_product_types):
                model.add(u[0, product] == x[0, product], name=f"cumulative_first_{product}")
            for position in range(1, self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    model.add(u[position, product] == u[position-1, product] + x[position, product],
                            name=f"cumulative_{position}_{product}")
            
            # Fixed total completion time
            model.add(t[self.total_number_of_products - 1, self.number_of_workstations - 1] == makespan_fixed, name="fixed_total_completion_time")
            
            # Ideal cumulative production (linear form)
            for position in range(self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    model.add(v[position, product] * makespan_fixed == self.product_cycle[product] * t[position, self.number_of_workstations - 1],
                            name=f"ideal_cumulative_{position}_{product}")
            
            # Deviation calculation
            for position in range(self.total_number_of_products):
                for product in range(self.number_of_product_types):
                    model.add(u[position, product] - v[position, product] == delta_plus[position, product] - delta_minus[position, product],
                            name=f"deviation_{position}_{product}")
            
            # Objective: Minimize total deviation (logistics leveling)
            objective_expr = model.sum(delta_plus[position, product] + delta_minus[position, product] 
                                    for position in range(self.total_number_of_products) for product in range(self.number_of_product_types))
            model.minimize(objective_expr)
            
            # Add epsilon constraints if provided
            if epsilon_values:
                # For model1 (reconfiguration cost)
                if 'model1' in epsilon_values and epsilon_values['model1'] != float('inf'):
                    # Add linearization constraints for y
                    for position in range(self.total_number_of_products - 1):
                        for i in range(self.number_of_product_types):
                            for j in range(self.number_of_product_types):
                                model.add(y[position, i, j] <= x[position, i],
                                        name=f"eps_y_leq_x1_{position}_{i}_{j}")
                                model.add(y[position, i, j] <= x[position+1, j],
                                        name=f"eps_y_leq_x2_{position}_{i}_{j}")
                                model.add(x[position, i] + x[position+1, j] - 1 <= y[position, i, j],
                                        name=f"eps_y_geq_sum_{position}_{i}_{j}")
                    for i in range(self.number_of_product_types):
                        for j in range(self.number_of_product_types):
                            model.add(y[self.total_number_of_products-1, i, j] <= x[self.total_number_of_products-1, i],
                                    name=f"eps_cycle_y_leq_x1_{i}_{j}")
                            model.add(y[self.total_number_of_products-1, i, j] <= x[0, j],
                                    name=f"eps_cycle_y_leq_x2_{i}_{j}")
                            model.add(x[self.total_number_of_products-1, i] + x[0, j] - 1 <= y[self.total_number_of_products-1, i, j],
                                    name=f"eps_cycle_y_geq_sum_{i}_{j}")
                    
                    cost_expr = 0
                    for position in range(self.total_number_of_products):
                        for i in range(self.number_of_product_types):
                            for j in range(self.number_of_product_types):
                                cost_expr += y[position, i, j] * self.product_switching_cost[i][j]
                    model.add(cost_expr <= epsilon_values['model1'], name="epsilon_constraint_model1")
                
                # For model2 (part frequency violations)
                if 'model2' in epsilon_values and epsilon_values['model2'] != float('inf'):
                    epsilon_param = 0.001
                    for position in range(self.total_number_of_products):
                        for part in range(self.number_of_parts):
                            B_l = self.part_frequency_window[part]
                            usage_expr = 0
                            for offset in range(B_l):
                                pos_idx = (position + offset) % self.total_number_of_products
                                for product in range(self.number_of_product_types):
                                    if self.part_requirement_matrix[product][part] >= 1:
                                        usage_expr += x[pos_idx, product]
                            model.add(usage_expr - self.part_frequency_max_usage[part] <= self.big_M_parameter * q[position, part],
                                    name=f"eps_violation_force_q1_{position}_{part}")
                            model.add(self.part_frequency_max_usage[part] - usage_expr + epsilon_param <= self.big_M_parameter * (1 - q[position, part]),
                                    name=f"eps_violation_force_q0_{position}_{part}")
                    violation_expr = model.sum(q[position, part] for position in range(self.total_number_of_products) for part in range(self.number_of_parts))
                    model.add(violation_expr <= epsilon_values['model2'], name="epsilon_constraint_model2")
            
            # Solve
            model.set_time_limit(self.max_seconds / 2)
            model.parameters.mip.tolerances.mipgap = 0.00
            
            solution = model.solve(log_output=True)
            solve_details = model.get_solve_details()
            
            result = {
                'objective': solution.objective_value if solution else math.pow(10, 8),
                'status': solve_details.status,
                'time': 0,
                'gap': getattr(solve_details, 'gap', None),
                'solution': solution,
                'makespan_fixed': makespan_fixed
            }
            
            print(f"Logistics leveling with epsilon - Optimal logistics leveling: {result['objective']}")
            print(f"Solve status: {solve_details.status}")
            
            return result
            
        except Exception as e:
            print(f"Error in _solve_logistics_leveling_with_epsilon: {e}")
            import traceback
            traceback.print_exc()
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'time': 0,
                'gap': None,
                'solution': None,
                'makespan_fixed': makespan_fixed,
                'error': str(e)
            }

    def _solve_primary_with_epsilon_constraints(self, primary_objective: str, makespan_fixed: float, epsilon_values: Dict[str, float]) -> Dict[str, Any]:
        """Solve primary objective (model1 or model2) with fixed makespan and epsilon constraints (including model3)."""
        try:
            model = Model(name=f"Epsilon_Constraint_Primary_{primary_objective}_FixedT")
            
            # Variables
            x = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), name='x')
            # For primary objective model1: need y; for model2: need q
            if primary_objective == 'model1':
                y = model.binary_var_cube(range(self.total_number_of_products), range(self.number_of_product_types), range(self.number_of_product_types), name='y')
            if primary_objective == 'model2' or ('model2' in epsilon_values and epsilon_values['model2'] != float('inf')):
                q = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_parts), name='q')
            # For model3 epsilon constraint, need t, u, v, delta variables
            if 'model3' in epsilon_values and epsilon_values['model3'] != float('inf'):
                t = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_workstations), lb=0, name='t')
                u = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='u')
                v = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='v')
                delta_plus = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='delta_plus')
                delta_minus = model.continuous_var_matrix(range(self.total_number_of_products), range(self.number_of_product_types), lb=0, name='delta_minus')
            
            # Common constraints: product assignment and demand
            for pos in range(self.total_number_of_products):
                model.add(model.sum(x[pos, prod] for prod in range(self.number_of_product_types)) == 1)
            for prod in range(self.number_of_product_types):
                model.add(model.sum(x[pos, prod] for pos in range(self.total_number_of_products)) == self.product_cycle[prod])
            
            # If model3 constraint needed, add completion time, cumulative, ideal cumulative constraints (with fixed T)
            if 'model3' in epsilon_values and epsilon_values['model3'] != float('inf'):
                # Completion time constraints (same as in _solve_logistics_leveling_fixed_makespan)
                model.add(t[0, 0] == model.sum(x[0, prod] * self.working_hours[prod][0] for prod in range(self.number_of_product_types)))
                for pos in range(1, self.total_number_of_products):
                    model.add(t[pos, 0] == t[pos-1, 0] + self.interval_time + 
                             model.sum(x[pos, prod] * self.working_hours[prod][0] for prod in range(self.number_of_product_types)))
                for pos in range(self.total_number_of_products):
                    for sta in range(1, self.number_of_workstations):
                        model.add(t[pos, sta] >= t[pos, sta-1] + 
                                 model.sum(x[pos, prod] * self.working_hours[prod][sta] for prod in range(self.number_of_product_types)))
                for pos in range(1, self.total_number_of_products):
                    for sta in range(self.number_of_workstations):
                        model.add(t[pos, sta] >= t[pos-1, sta] + 
                                 model.sum(x[pos, prod] * self.working_hours[prod][sta] for prod in range(self.number_of_product_types)))
                # Cumulative production
                for prod in range(self.number_of_product_types):
                    model.add(u[0, prod] == x[0, prod])
                for pos in range(1, self.total_number_of_products):
                    for prod in range(self.number_of_product_types):
                        model.add(u[pos, prod] == u[pos-1, prod] + x[pos, prod])
                # Fixed total completion time
                model.add(t[self.total_number_of_products-1, self.number_of_workstations-1] == makespan_fixed)
                # Ideal cumulative: v * T_fixed = d_i * t[p, last]
                for pos in range(self.total_number_of_products):
                    for prod in range(self.number_of_product_types):
                        model.add(v[pos, prod] * makespan_fixed == self.product_cycle[prod] * t[pos, self.number_of_workstations-1])
                # Deviations
                for pos in range(self.total_number_of_products):
                    for prod in range(self.number_of_product_types):
                        model.add(u[pos, prod] - v[pos, prod] == delta_plus[pos, prod] - delta_minus[pos, prod])
                # Constraint on logistics leveling
                leveling_expr = model.sum(delta_plus[pos, prod] + delta_minus[pos, prod] for pos in range(self.total_number_of_products) for prod in range(self.number_of_product_types))
                model.add(leveling_expr <= epsilon_values['model3'])
            
            # If primary objective is model1, add y linearization and objective
            if primary_objective == 'model1':
                # y linearization (consecutive positions)
                for pos in range(self.total_number_of_products - 1):
                    for i in range(self.number_of_product_types):
                        for j in range(self.number_of_product_types):
                            model.add(y[pos, i, j] <= x[pos, i])
                            model.add(y[pos, i, j] <= x[pos+1, j])
                            model.add(x[pos, i] + x[pos+1, j] - 1 <= y[pos, i, j])
                # cycle
                for i in range(self.number_of_product_types):
                    for j in range(self.number_of_product_types):
                        model.add(y[self.total_number_of_products-1, i, j] <= x[self.total_number_of_products-1, i])
                        model.add(y[self.total_number_of_products-1, i, j] <= x[0, j])
                        model.add(x[self.total_number_of_products-1, i] + x[0, j] - 1 <= y[self.total_number_of_products-1, i, j])
                cost_expr = model.sum(y[pos, i, j] * self.product_switching_cost[i][j] for pos in range(self.total_number_of_products) for i in range(self.number_of_product_types) for j in range(self.number_of_product_types))
                model.minimize(cost_expr)
            
            # If primary objective is model2, add q constraints and objective
            if primary_objective == 'model2':
                for pos in range(self.total_number_of_products):
                    for part in range(self.number_of_parts):
                        B_l = self.part_frequency_window[part]
                        usage_expr = 0
                        for offset in range(B_l):
                            idx = (pos + offset) % self.total_number_of_products
                            for prod in range(self.number_of_product_types):
                                if self.part_requirement_matrix[prod][part] >= 1:
                                    usage_expr += x[idx, prod]
                        model.add(usage_expr - self.part_frequency_max_usage[part] <= self.big_M_parameter * q[pos, part])
                        model.add(self.part_frequency_max_usage[part] - usage_expr <= self.big_M_parameter * (1 - q[pos, part]))
                violation_expr = model.sum(q[pos, part] for pos in range(self.total_number_of_products) for part in range(self.number_of_parts))
                model.minimize(violation_expr)
            
            # Add epsilon constraints for other objectives (if not model3 already handled)
            if epsilon_values:
                for obj_name, eps_val in epsilon_values.items():
                    if obj_name == primary_objective or eps_val == float('inf'):
                        continue
                    if obj_name == 'model1' and primary_objective != 'model1':
                        # Need to add y constraints and cost <= eps_val
                        if 'y' not in locals():
                            y = model.binary_var_cube(range(self.total_number_of_products), range(self.number_of_product_types), range(self.number_of_product_types), name='y')
                            # Add y linearization constraints
                            for pos in range(self.total_number_of_products - 1):
                                for i in range(self.number_of_product_types):
                                    for j in range(self.number_of_product_types):
                                        model.add(y[pos, i, j] <= x[pos, i])
                                        model.add(y[pos, i, j] <= x[pos+1, j])
                                        model.add(x[pos, i] + x[pos+1, j] - 1 <= y[pos, i, j])
                            for i in range(self.number_of_product_types):
                                for j in range(self.number_of_product_types):
                                    model.add(y[self.total_number_of_products-1, i, j] <= x[self.total_number_of_products-1, i])
                                    model.add(y[self.total_number_of_products-1, i, j] <= x[0, j])
                                    model.add(x[self.total_number_of_products-1, i] + x[0, j] - 1 <= y[self.total_number_of_products-1, i, j])
                        cost_expr = model.sum(y[pos, i, j] * self.product_switching_cost[i][j] for pos in range(self.total_number_of_products) for i in range(self.number_of_product_types) for j in range(self.number_of_product_types))
                        model.add(cost_expr <= eps_val)
                    elif obj_name == 'model2' and primary_objective != 'model2':
                        # Add q constraints
                        if 'q' not in locals():
                            q = model.binary_var_matrix(range(self.total_number_of_products), range(self.number_of_parts), name='q')
                        for pos in range(self.total_number_of_products):
                            for part in range(self.number_of_parts):
                                B_l = self.part_frequency_window[part]
                                usage_expr = 0
                                for offset in range(B_l):
                                    idx = (pos + offset) % self.total_number_of_products
                                    for prod in range(self.number_of_product_types):
                                        if self.part_requirement_matrix[prod][part] >= 1:
                                            usage_expr += x[idx, prod]
                                model.add(usage_expr - self.part_frequency_max_usage[part] <= self.big_M_parameter * q[pos, part])
                                model.add(self.part_frequency_max_usage[part] - usage_expr <= self.big_M_parameter * (1 - q[pos, part]))
                        violation_expr = model.sum(q[pos, part] for pos in range(self.total_number_of_products) for part in range(self.number_of_parts))
                        model.add(violation_expr <= eps_val)
            
            # Solve
            model.set_time_limit(self.max_seconds / 2)
            model.parameters.mip.tolerances.mipgap = 0.00
            solution = model.solve(log_output=True)
            solve_details = model.get_solve_details()
            return {
                'objective': solution.objective_value if solution else math.pow(10, 8),
                'status': solve_details.status,
                'gap': getattr(solve_details, 'gap', None),
                'solution': solution,
                'makespan_fixed': makespan_fixed
            }
        except Exception as e:
            print(f"Error in _solve_primary_with_epsilon_constraints: {e}")
            import traceback
            traceback.print_exc()
            return {
                'objective': math.pow(10, 8),
                'status': 'error',
                'gap': None,
                'solution': None,
                'makespan_fixed': makespan_fixed,
                'error': str(e)
            }

    def solve_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Solve all three models sequentially.
        
        Returns:
            Dictionary containing solutions from all three models
        """
        results = {}
        
        # Solve Model 1
        results['model1'] = self.solve_model1()
        
        # Solve Model 2
        results['model2'] = self.solve_model2()
        
        # Solve Model 3
        results['model3'] = self.solve_model3()
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Print summary of all solutions."""
        print("\n" + "=" * 80)
        print("SUMMARY OF ALL MODELS")
        print("=" * 80)
        print(f"{'Model':<10} {'Objective':<20} {'Time (s)':<10} {'Status':<15}")
        print("-" * 80)
        for model_name, result in results.items():
            obj = result.get('objective', 'N/A')
            if isinstance(obj, float):
                obj = f"{obj:.2f}"
            time_val = result.get('time', 0)
            status = result.get('status', 'N/A')
            print(f"{model_name:<10} {obj:<20} {time_val:<10.2f} {status:<15}")
        print("=" * 80)