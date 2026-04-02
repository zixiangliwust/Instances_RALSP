"""evolu: Evolutionary Computation Framework for Assembly Line Balancing.

evolu is a comprehensive Python framework for solving assembly line balancing
problems using evolutionary computation techniques. The framework provides
implementations of state-of-the-art multi-objective and single-objective
optimization algorithms, along with specialized problem encodings for various
assembly line balancing variants.

Main Modules:
    - core: Core framework components (algorithms, problems, solutions, operators)
    - operator: Genetic operators (crossover, mutation, selection, replacement)
    - optimizers: Optimization algorithms (NSGA-II, MOEA/D, SPEA2, etc.)
    - problem: Problem definitions and benchmark problems
    - util: Utility functions and components
    - lab: Laboratory tools for experimentation and analysis

Example:
    Basic usage of the framework::
    
        from evolu.problem.multi_objective.float_problems import ZDT1
        from evolu.optimizers.multiobjective import NSGAII
        from evolu.util.termination_criterion import StoppingByEvaluations
        
        problem = ZDT1()
        algorithm = NSGAII(
            problem=problem,
            population_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=25000)
        )
        algorithm.run()
        solutions = algorithm.get_result()
"""
from evolu import optimizers, core, operator, problem
from evolu.logger import configure_logging

configure_logging()
__all__ = ["core", "optimizers", "operator", "problem"]
__version__ = "1.8.11"
