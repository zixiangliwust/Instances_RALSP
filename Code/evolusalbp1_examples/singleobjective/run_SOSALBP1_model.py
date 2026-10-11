"""Example script for running mathematical programming models on SALBP1/UALBP1.

This script demonstrates how to use mathematical programming models to solve
Simple Assembly Line Balancing Problem Type 1 (SALBP1/UALBP1) exactly using CPLEX
or constraint programming.

Features:
- Mixed-integer programming (MIP) model for exact optimization
- Constraint programming (CP) model variants
- Handles precedence constraints through mathematical constraints
- Minimizes number of workstations for SALBP1/UALBP1
- Solves multiple problem instances from a directory
- Prints solution quality, optimality status, and solving time

This is useful for obtaining optimal solutions and verifying the quality of
heuristic algorithms on assembly line balancing problems.
"""
import sys
import os
import winsound
from typing import List
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolusalbp1.problem.singleobjective.SALBP1_problems import *
from evolusalbp1.problem.singleobjective.UALBP1_problems import *

def select_and_run_algorithm(algorithm_name: str, file_path: str, problem_name: str, problem_type: str = "SALBP1") -> None:
    """Select and run a mathematical programming model on the given SALBP1/UALBP1 problem.
    
    This function creates a FloatSALBP1 or FloatUALBP1 problem instance, configures a mathematical
    programming model (MIP or CP), solves it using CPLEX, and prints the results.
    
    Args:
        algorithm_name (str): Name of the model to run. Supported models:
            - "MPModel": Mixed-integer programming model using CPLEX
            - "CPModel": Constraint programming model
            - "CPModelNew": New constraint programming model variant
        file_path (str): Path to the directory containing the problem instance file.
        problem_name (str): Name of the problem instance file to load.
        problem_type (str): Type of problem - "SALBP1" or "UALBP1" (default: "SALBP1")
    
    Note:
        The models minimize the number of workstations while satisfying precedence
        constraints and cycle time constraints. Results include objective value
        (number of stations), optimality status, and solving time.
        UALBP1 allows bidirectional task assignment on U-shaped lines.
    """
    # Determine problem type based on problem_type parameter
    if problem_type == "UALBP1":
        problem: FloatUALBP1 = FloatUALBP1(file_path, problem_name)
    else:  # SALBP1
        problem: FloatSALBP1 = FloatSALBP1(file_path, problem_name)
    """
    Mathematical models
    """
    # SALBP1 models
    if algorithm_name == "MPModel":
        if problem_type == "UALBP1":
            from evolusalbp1.problem.singleobjective.UALBP1_Model import UALBP1MPModel
            algorithm = UALBP1MPModel(problem)
        else:
            from evolusalbp1.problem.singleobjective.SALBP1_Model import SALBP1MPModel
            algorithm = SALBP1MPModel(problem)
    if algorithm_name == "CPModel":
        if problem_type == "UALBP1":
            from evolusalbp1.problem.singleobjective.UALBP1_Model import UALBP1CPModel
            algorithm = UALBP1CPModel(problem)
        else:
            from evolusalbp1.problem.singleobjective.SALBP1_Model import SALBP1CPModel
            algorithm = SALBP1CPModel(problem)
    if algorithm_name == "CPModelNew":
        from evolusalbp1.problem.singleobjective.SALBP1_Model import SALBP1CPModelNew
        algorithm = SALBP1CPModelNew(problem)
    algorithm.run()


if __name__ == "__main__":
    file_path: str = "D:/GitHubInstances/Instances_SALBP1/Instances/"
    # 获取目录中的所有条目
    entries: List[str] = os.listdir(file_path)
    # 过滤出文件
    problem_name_list: List[str] = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]
    # problem_name_list = problem_name_list[0:10]
    problem_name_list = problem_name_list[1050:1060]

    algorithm_name_list: List[str] = [
        "MPModel",
        "CPModel",
        "CPModelNew",
    ]
    num_of_runs: int = 1
    for algorithm_name in algorithm_name_list:
        for problem_index in range(0, len(problem_name_list)):
            for num_run in range(0, num_of_runs):
                select_and_run_algorithm(algorithm_name, file_path, problem_name_list[problem_index])
    print("Execution completed")