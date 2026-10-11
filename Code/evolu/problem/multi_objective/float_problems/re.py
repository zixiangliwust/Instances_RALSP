# -*- coding: utf-8 -*-
"""Real-World Engineering (RE) series problems.

This module provides real-world engineering optimization problems for
multi-objective optimization benchmarking.

References:
[1] Initial code built based on jMetalPy implementation
[2] Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
    and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
"""
from math import cos, exp, pi, sin, sqrt
from evolu.core.problem import FloatProblem
from evolu.core.solution import FloatSolution


class RE1(FloatProblem):
    """Problem RE1: Four bar truss design.
    
    This is a real-world engineering problem involving the design of a four-bar truss
    to minimize structural volume and displacement.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE1, self).__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Volume", "Displacement"]
        self.lower_bound = [1.0, sqrt(2.0), sqrt(2.0), 1.0]
        self.upper_bound = [3.0, 3.0, 3.0, 3.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Volume calculation
        volume = 100 * (2 * x[0] + sqrt(2.0) * x[1] + sqrt(x[2]) + x[3])
        
        # Displacement calculation
        displacement = (20 / (x[2] * x[3])) + (40 * sqrt(2.0) / (x[1] * x[3]))
        
        solution.objectives[0] = volume
        solution.objectives[1] = displacement
        return solution
    
    def get_name(self):
        return "RE1"


class RE2(FloatProblem):
    """Problem RE2: Reinforced concrete beam design.
    
    This problem involves designing a reinforced concrete beam to minimize
    cost and maximize reliability.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE2, self).__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Cost", "Reliability"]
        self.lower_bound = [0.2, 10.0, 10.0, 0.9]
        self.upper_bound = [15.0, 50.0, 50.0, 5.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Cost calculation (to minimize)
        cost = 29.4 * x[0] + 0.6 * x[1] * x[2]
        
        # Reliability calculation (to minimize, actually represents failure probability)
        # This is a simplified version - actual implementation may vary
        reliability = -1.0 * (x[0] * x[1] * x[2] * x[3] / 100.0)
        
        solution.objectives[0] = cost
        solution.objectives[1] = reliability
        return solution
    
    def get_name(self):
        return "RE2"


class RE3(FloatProblem):
    """Problem RE3: Pressure vessel design.
    
    This problem involves designing a pressure vessel to minimize cost
    while meeting safety requirements.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE3, self).__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Cost", "Weight"]
        self.lower_bound = [1.0, 1.0, 10.0, 10.0]
        self.upper_bound = [99.0, 99.0, 200.0, 200.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Cost calculation
        cost = 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2] * x[2] + \
               3.1661 * x[0] * x[0] * x[3] + 19.84 * x[0] * x[0] * x[2]
        
        # Weight calculation
        weight = 0.7854 * x[0] * x[2] * x[2] * (x[3] + 2.0 * x[1])
        
        solution.objectives[0] = cost
        solution.objectives[1] = weight
        return solution
    
    def get_name(self):
        return "RE3"


class RE4(FloatProblem):
    """Problem RE4: Welded beam design.
    
    This problem involves designing a welded beam to minimize cost
    and deflection.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE4, self).__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Cost", "Deflection"]
        self.lower_bound = [0.125, 0.1, 0.1, 0.125]
        self.upper_bound = [5.0, 10.0, 10.0, 5.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Cost calculation
        cost = 1.10471 * x[0] * x[0] * x[1] + 0.04811 * x[2] * x[3] * (14.0 + x[1])
        
        # Deflection calculation
        P = 6000.0
        L = 14.0
        E = 30e6
        G = 12e6
        
        delta = (4 * P * L * L * L) / (E * x[3] * x[2] * x[2] * x[2])
        
        solution.objectives[0] = cost
        solution.objectives[1] = delta
        return solution
    
    def get_name(self):
        return "RE4"


class RE5(FloatProblem):
    """Problem RE5: Disk brake design.
    
    This problem involves designing a disk brake to minimize mass
    and stopping time.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE5, self).__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Mass", "StoppingTime"]
        self.lower_bound = [55.0, 75.0, 1000.0, 11.0]
        self.upper_bound = [80.0, 110.0, 3000.0, 20.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Mass calculation
        mass = 4.9 * 1e-5 * (x[1] * x[1] - x[0] * x[0]) * (x[3] - 1.0)
        
        # Stopping time calculation (simplified)
        # Actual stopping time involves complex brake dynamics
        stopping_time = (9.82 * 1e6 * (x[1] * x[1] - x[0] * x[0])) / \
                       ((x[1] * x[1] * x[1] - x[0] * x[0] * x[0]) * x[2] * x[3])
        
        solution.objectives[0] = mass
        solution.objectives[1] = stopping_time
        return solution
    
    def get_name(self):
        return "RE5"


class RE6(FloatProblem):
    """Problem RE6: Gear train design.
    
    This problem involves designing a gear train to minimize
    gear ratio error and volume.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE6, self).__init__()
        self.number_of_variables = 4
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Error", "Volume"]
        self.lower_bound = [12.0, 12.0, 12.0, 12.0]
        self.upper_bound = [60.0, 60.0, 60.0, 60.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Gear ratio error (to minimize)
        target_ratio = 1.0 / 6.931
        actual_ratio = (x[0] * x[1]) / (x[2] * x[3])
        error = abs(actual_ratio - target_ratio)
        
        # Volume calculation (simplified)
        volume = x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]
        
        solution.objectives[0] = error
        solution.objectives[1] = volume
        return solution
    
    def get_name(self):
        return "RE6"


class RE7(FloatProblem):
    """Problem RE7: Two bar truss design.
    
    This problem involves designing a two-bar truss to minimize
    weight and deflection.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE7, self).__init__()
        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ["Weight", "Deflection"]
        self.lower_bound = [0.05, 0.2, 0.2]
        self.upper_bound = [0.5, 1.0, 1.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Weight calculation
        rho = 0.283  # Material density (lb/in^3)
        weight = rho * x[0] * x[1] * sqrt(1.0 + x[2] * x[2])
        
        # Deflection calculation (simplified)
        E = 30e6  # Young's modulus
        P = 100.0  # Applied load
        deflection = (P * sqrt(1.0 + x[2] * x[2]) * (1.0 + x[2] * x[2])) / \
                    (E * x[0] * x[1] * x[2])
        
        solution.objectives[0] = weight
        solution.objectives[1] = deflection
        return solution
    
    def get_name(self):
        return "RE7"


class RE8(FloatProblem):
    """Problem RE8: Spring design.
    
    This problem involves designing a compression spring to minimize
    volume and maximize safety factor.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE8, self).__init__()
        self.number_of_variables = 3
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ["Volume", "SafetyFactor"]
        self.lower_bound = [0.05, 0.25, 2.0]
        self.upper_bound = [2.0, 1.3, 15.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Volume calculation
        volume = pi * x[0] * x[0] * x[1] * x[2]
        
        # Safety factor (simplified calculation)
        # Actual calculation involves stress analysis
        safety_factor = x[0] * x[1] / x[2]  # Simplified
        
        solution.objectives[0] = volume
        solution.objectives[1] = -safety_factor  # Negate for maximization
        return solution
    
    def get_name(self):
        return "RE8"


class RE9(FloatProblem):
    """Problem RE9: Multiple disk clutch brake design.
    
    This problem involves designing a multiple disk clutch brake
    to minimize mass and maximize torque capacity.
    
    References:
        Deb, K., & Datta, R. (2012). Hybrid evolutionary multi-objective optimization
        and analysis of machining operations. Engineering Optimization, 44(6), 685-706.
    """
    
    def __init__(self):
        super(RE9, self).__init__()
        self.number_of_variables = 5
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MINIMIZE, self.MAXIMIZE]
        self.obj_labels = ["Mass", "Torque"]
        self.lower_bound = [55.0, 75.0, 1000.0, 11.0, 2.0]
        self.upper_bound = [80.0, 110.0, 3000.0, 20.0, 9.0]
        
    def evaluate_solution(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        
        # Mass calculation
        rho = 0.0000078  # Material density (kg/mm^3)
        mass = pi * (x[1] * x[1] - x[0] * x[0]) * x[2] * (x[4] + 1.0) * rho
        
        # Torque capacity (simplified)
        mu = 0.6  # Coefficient of friction
        P = 1.0  # Applied pressure
        torque = (2.0 / 3.0) * mu * x[4] * P * (x[1] * x[1] * x[1] - x[0] * x[0] * x[0]) / \
                (x[1] * x[1] - x[0] * x[0])
        
        solution.objectives[0] = mass
        solution.objectives[1] = -torque  # Negate for maximization
        return solution
    
    def get_name(self):
        return "RE9"

