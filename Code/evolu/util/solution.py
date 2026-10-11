# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import List

from evolu.core.solution import FloatSolution, Solution
from evolu.util.archive import Archive, NonDominatedSolutionsArchive

logger = logging.getLogger(__name__)
"""
module:: solutions
synopsis: Utils to print solutions.
moduleauthor:: Antonio J. Nebro <ajnebro@uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


def get_non_dominated_solutions(solutions: List[Solution]) -> List[Solution]:
    """Extract non-dominated solutions from a list of solutions.
    
    This function filters a list of solutions to keep only those that are
    non-dominated (i.e., not dominated by any other solution in the list).
    The result is a Pareto-optimal subset of the input solutions.
    
    Args:
        solutions (List[Solution]): List of solutions to filter.
    
    Returns:
        List[Solution]: List of non-dominated solutions (Pareto-optimal set).
    
    Example:
        >>> all_solutions = [sol1, sol2, sol3, sol4]
        >>> pareto_front = get_non_dominated_solutions(all_solutions)
        >>> len(pareto_front)  # Typically smaller than input list
    """
    archive: Archive = NonDominatedSolutionsArchive()
    for solution in solutions:
        archive.add(solution)
    return archive.solution_list


def read_solutions(file_name: str) -> List[FloatSolution]:
    """Read reference front solutions from a file.
    
    This function reads a file containing objective values (one per line,
    space-separated) and creates FloatSolution objects with those objective
    values. Commonly used to load reference Pareto fronts for quality
    indicator calculations.
    
    Args:
        file_name (str): Path to the file containing solution objective values.
            Each line should contain space-separated float values representing
            one solution's objective values.
    
    Returns:
        List[FloatSolution]: List of FloatSolution objects with objectives set
            from the file. Returns empty list if file not found.
    
    Note:
        - The file format expects one solution per line
        - Each line contains space-separated float values
        - All solutions are assumed to have the same number of objectives
        - A warning is logged if the file is not found, but no exception is raised
    
    Example:
        >>> # File content (e.g., reference.pf):
        >>> # 1.0 2.0 3.0
        >>> # 2.0 1.5 2.5
        >>> solutions = read_solutions("reference.pf")
        >>> len(solutions)  # 2 solutions
    """
    front = []
    if Path(file_name).is_file():
        with open(file_name) as file:
            for line in file:
                vector = [float(x) for x in line.split()]
                solution = FloatSolution([], [], len(vector))
                solution.objectives = vector
                front.append(solution)
    else:
        logger.warning("Reference front file was not found at {}".format(file_name))
    return front


def print_variables_to_file(solutions: List[Solution], file_name: str) -> None:
    """Print solution decision variables to a file.
    
    This function writes the decision variables of solutions to a text file,
    with one solution per line. Variables are space-separated.
    
    Args:
        solutions (List[Solution]): List of solutions to write. Can also be
            a single Solution object (will be converted to list).
        file_name (str): Path to the output file. Parent directories are
            created automatically if they don't exist.
    
    Note:
        - If solutions is a single Solution, it's automatically converted to a list
        - Each solution's variables are written on one line, space-separated
        - Existing files are overwritten
    """
    logger.info("Output file (variables): " + file_name)
    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    except FileNotFoundError:
        pass
    if type(solutions) is not list:
        solutions = [solutions]
    with open(file_name, "w") as of:
        for solution in solutions:
            for variables in solution.variables:
                of.write(str(variables) + " ")
            of.write("\n")


def print_variables_to_screen(solutions: List[Solution]) -> None:
    """Print solution decision variables to the console.
    
    This function displays the decision variables of solutions to stdout,
    showing only the first variable of each solution.
    
    Args:
        solutions (List[Solution]): List of solutions to display. Can also be
            a single Solution object (will be converted to list).
    
    Note:
        Only the first variable (solution.variables[0]) of each solution is printed.
    """
    if type(solutions) is not list:
        solutions = [solutions]
    for solution in solutions:
        print(solution.variables[0])


def print_function_values_to_file(solutions: List[Solution], file_name: str) -> None:
    """Print solution objective function values to a file.
    
    This function writes the objective values of solutions to a text file,
    with one solution per line. Objective values are space-separated.
    
    Args:
        solutions (List[Solution]): List of solutions to write. Can also be
            a single Solution object (will be converted to list).
        file_name (str): Path to the output file. Parent directories are
            created automatically if they don't exist.
    
    Note:
        - If solutions is a single Solution, it's automatically converted to a list
        - Each solution's objectives are written on one line, space-separated
        - Existing files are overwritten
        - This format is commonly used for saving Pareto fronts
    """
    logger.info("Output file (function values): " + file_name)
    try:
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    except FileNotFoundError:
        pass
    if type(solutions) is not list:
        solutions = [solutions]
    with open(file_name, "w") as of:
        for solution in solutions:
            for function_value in solution.objectives:
                of.write(str(function_value) + " ")
            of.write("\n")


def print_function_values_to_screen(solutions: List[Solution]) -> None:
    """Print solution objective function values to the console.
    
    This function displays the objective values of solutions to stdout,
    with each solution on its own line showing its index and objectives.
    
    Args:
        solutions (List[Solution]): List of solutions to display. Can also be
            a single Solution object (will be converted to list).
    
    Example:
        >>> solutions = [sol1, sol2, sol3]
        >>> print_function_values_to_screen(solutions)
        0: [1.5, 2.3, 3.1]
        1: [1.8, 2.0, 3.0]
        2: [2.0, 1.9, 2.9]
    """
    if type(solutions) is not list:
        solutions = [solutions]
    for solution in solutions:
        print(str(solutions.index(solution)) + ": ", sep="  ", end="", flush=True)
        print(solution.objectives, sep="  ", end="", flush=True)
        print()
