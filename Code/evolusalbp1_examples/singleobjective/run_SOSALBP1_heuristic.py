"""Example script for running heuristic methods on SALBP1.

This script demonstrates how to use simple heuristic methods to quickly generate
solutions for Simple Assembly Line Balancing Problem Type 1 (SALBP1).

Features:
- Multiple heuristic priority rules:
  - Longest/Shortest Operation Time (LOT/SOT)
  - Maximum/Minimum Total Number of Successors
  - Maximum/Minimum Total Operation Time of Successors
  - Maximum/Minimum Ranked Positional Weight
  - Maximum/Minimum Ranked Reverse Positional Weight
- Fast solution generation for initial solutions or quick estimates
- Deterministic solutions based on priority rules

These heuristics are useful for:
- Generating initial solutions for metaheuristics
- Quick feasibility checking
- Providing benchmark solutions for comparison
"""
import sys
import os
import winsound
from typing import List
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evolusalbp1.problem.singleobjective.SALBP1_problems import *

def select_and_run_algorithm(algorithm_name: str, file_path: str, problem_name: str) -> None:
    """Select and run a heuristic method on the given SALBP1 problem.
    
    This function creates a PermutationSALBP1 problem instance, applies a
    heuristic method based on a priority rule, and prints the result.
    
    Args:
        algorithm_name (str): Name of the heuristic to run. Supported heuristics:
            - "LongestOperationTime": Prioritize tasks with longest processing time
            - "ShortestOperationTime": Prioritize tasks with shortest processing time
            - "MaximumTotalNumberOfSuccessors": Prioritize tasks with most successors
            - "MinimumTotalNumberOfSuccessors": Prioritize tasks with fewest successors
            - "MaximumTotalOperationTimeOfSuccessors": Prioritize by successor workload
            - "MinimumTotalOperationTimeOfSuccessors": Prioritize by successor workload
            - "MaximumRankedPositionalWeight": Prioritize by forward positional weight
            - "MinimumRankedPositionalWeight": Prioritize by forward positional weight
            - "MaximumRankedReversePositionalWeight": Prioritize by reverse positional weight
            - "MinimumRankedReversePositionalWeight": Prioritize by reverse positional weight
        file_path (str): Path to the directory containing the problem instance file.
        problem_name (str): Name of the problem instance file to load.
    
    Note:
        Each heuristic uses a different priority rule to select tasks for
        assignment to workstations. The result (number of workstations) is
        printed to console.
    """
    problem: PermutationSALBP1 = PermutationSALBP1(file_path, problem_name)
    max_evaluations: int = 1000
    """
    Heuristic methods
    """
    if algorithm_name == "LongestOperationTime":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.longest_operation_time_heuristic()
        print(solution.objectives[0])
    if algorithm_name == "ShortestOperationTime":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.shortest_operation_time_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MaximumTotalNumberOfSuccessors":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.maximum_total_number_of_successors_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MinimumTotalNumberOfSuccessors":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.minimum_total_number_of_successors_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MaximumTotalOperationTimeOfSuccessors":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.maximum_total_operation_time_of_successors_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MinimumTotalOperationTimeOfSuccessors":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.minimum_total_operation_time_of_successors_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MaximumRankedPositionalWeight":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.maximum_ranked_positional_weight_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MaximumAverageRankedPositionalWeight":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.maximum_average_ranked_positional_weight_heuristic();
        print(solution.objectives[0])
    if algorithm_name == "MaximumWeightedOperationTimeAndSuccessors":
        from evolusalbp1.problem.singleobjective.SALBP1_heuristic import SALBP1SimpleHeuristic
        SALBP1_simple_heuristic = SALBP1SimpleHeuristic(problem)
        solution = SALBP1_simple_heuristic.maximum_weighted_operation_time_and_successors_heuristic();
        print(solution.objectives[0])


if __name__ == "__main__":
    file_path: str = "D:/GitHubInstances/Instances_SALBP1/Instances/"
    # 获取目录中的所有条目
    entries: List[str] = os.listdir(file_path)
    # 过滤出文件
    problem_name_list: List[str] = [entry for entry in entries if os.path.isfile(os.path.join(file_path, entry))]
    problem_name_list = problem_name_list[525:]

    algorithm_name_list: List[str] = [
        "LongestOperationTime",
        "ShortestOperationTime",
        "MaximumTotalNumberOfSuccessors",
        "MinimumTotalNumberOfSuccessors",
        "MaximumTotalOperationTimeOfSuccessors",
        "MinimumTotalOperationTimeOfSuccessors",
        "MaximumRankedPositionalWeight",
        "MaximumAverageRankedPositionalWeight",
        "MaximumWeightedOperationTimeAndSuccessors",
    ]
    num_of_runs: int = 1
    for algorithm_name in algorithm_name_list:
        for problem_index in range(0, len(problem_name_list)):
            for num_run in range(0, num_of_runs):
                select_and_run_algorithm(algorithm_name, file_path, problem_name_list[problem_index])
    print("Execution completed")

    # Play beep sound when execution completes (Windows only)
    try:
        winsound.Beep(440, 10000)  # Frequency 440Hz, duration 10000ms
        print("End of beep.")
    except Exception:
        pass  # Ignore if winsound is not available on this platform
