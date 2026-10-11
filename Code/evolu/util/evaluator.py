# -*- coding: utf-8 -*-
import functools
from abc import ABC, abstractmethod
from multiprocessing.pool import Pool, ThreadPool
from typing import Generic, List, Optional, TypeVar

try:
    import dask
except ImportError:
    dask = None  # type: ignore
try:
    from pyspark import SparkConf, SparkContext
except ImportError:
    SparkConf = None  # type: ignore
    SparkContext = None  # type: ignore
from evolu.core.problem import Problem

S = TypeVar("S")


class Evaluator(Generic[S], ABC):
    """Base class for evaluating populations of solutions.
    
    Evaluators are responsible for computing objective values and constraint
    violations for solutions. Different implementations can provide sequential,
    parallel, or distributed evaluation capabilities.
    
    Note:
        Subclasses must implement the evaluate() method.
    """
    
    @abstractmethod
    def evaluate(self, solution_list: List[S], problem: Problem[S]) -> List[S]:
        """Evaluate a list of solutions.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        
        Returns:
            List[S]: List of evaluated solutions with objective values set.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `evaluate`.")

    @staticmethod
    def evaluate_solution(solution: S, problem: Problem[S]) -> None:
        """Evaluate a single solution.
        
        This static method delegates the evaluation to the problem instance.
        It modifies the solution in-place by setting its objective values.
        
        Args:
            solution (S): Solution to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        """
        problem.evaluate_solution(solution)


class SequentialEvaluator(Evaluator[S]):
    """Sequential evaluator for solutions.
    
    This evaluator processes solutions one at a time in a sequential manner.
    It is the simplest evaluation strategy and requires no additional dependencies.
    Suitable for small populations or when parallelization overhead is not justified.
    """
    
    def evaluate(self, solution_list: List[S], problem: Problem[S]) -> List[S]:
        """Evaluate solutions sequentially.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        
        Returns:
            List[S]: List of evaluated solutions.
        """
        for solution in solution_list:
            Evaluator.evaluate_solution(solution, problem)
        return solution_list


class MapEvaluator(Evaluator[S]):
    """Thread pool-based evaluator for solutions.
    
    This evaluator uses a thread pool to evaluate solutions concurrently.
    Suitable for I/O-bound evaluation tasks where the GIL does not limit performance.
    
    Attributes:
        pool (ThreadPool): Thread pool used for concurrent evaluation.
    """
    
    def __init__(self, processes: Optional[int] = None) -> None:
        """Initialize with optional number of threads.
        
        Args:
            processes (Optional[int], optional): Number of threads in the pool.
                If None, uses default number of threads. Defaults to None.
        """
        self.pool = ThreadPool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem[S]) -> List[S]:
        """Evaluate solutions using thread pool.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        
        Returns:
            List[S]: List of evaluated solutions.
        """
        self.pool.map(lambda solution: Evaluator.evaluate_solution(solution, problem), solution_list)
        return solution_list


class MultiprocessEvaluator(Evaluator[S]):
    """Multiprocessing-based evaluator for solutions.
    
    This evaluator uses a process pool to evaluate solutions in parallel across
    multiple CPU cores. Suitable for CPU-bound evaluation tasks that can benefit
    from true parallelism (bypasses Python's GIL).
    
    Attributes:
        pool (Pool): Process pool used for parallel evaluation.
    
    Note:
        The problem instance must be picklable for this evaluator to work correctly.
    """
    
    def __init__(self, processes: Optional[int] = None) -> None:
        """Initialize with optional number of processes.
        
        Args:
            processes (Optional[int], optional): Number of processes in the pool.
                If None, uses number of CPU cores. Defaults to None.
        """
        super().__init__()
        self.pool = Pool(processes)

    def evaluate(self, solution_list: List[S], problem: Problem[S]) -> List[S]:
        """Evaluate solutions using multiprocessing pool.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        
        Returns:
            List[S]: List of evaluated solutions.
        """
        return self.pool.map(functools.partial(evaluate_solution, problem=problem), solution_list)


class SparkEvaluator(Evaluator[S]):
    """Apache Spark-based evaluator for distributed solution evaluation.
    
    This evaluator uses Apache Spark to distribute solution evaluation across
    a cluster or multiple cores. Suitable for large-scale parallel evaluation
    on distributed systems.
    
    Attributes:
        spark_conf (SparkConf): Spark configuration.
        spark_context (SparkContext): Spark context for distributed execution.
    
    Note:
        Requires PySpark to be installed. Use this for distributed evaluation
        scenarios or when dealing with very large populations.
    """
    
    def __init__(self, processes: int = 8) -> None:
        """Initialize Spark evaluator with specified number of processes.
        
        Args:
            processes (int, optional): Number of Spark executors. Defaults to 8.
        
        Raises:
            ImportError: If PySpark is not installed.
        """
        if SparkConf is None or SparkContext is None:
            raise ImportError("PySpark is not installed. Install it with: pip install pyspark")
        self.spark_conf = SparkConf().setAppName("jmetalpy").setMaster(f"local[{processes}]")
        self.spark_context = SparkContext(conf=self.spark_conf)
        logger = self.spark_context._jvm.org.apache.log4j
        logger.LogManager.getLogger("org").setLevel(logger.Level.WARN)

    def evaluate(self, solution_list: List[S], problem: Problem[S]) -> List[S]:
        """Evaluate solutions using Apache Spark.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        
        Returns:
            List[S]: List of evaluated solutions.
        """
        solutions_to_evaluate = self.spark_context.parallelize(solution_list)
        return solutions_to_evaluate.map(lambda s: problem.evaluate_solution(s)).collect()


def evaluate_solution(solution: S, problem: Problem[S]) -> S:
    """Evaluate a single solution using the problem's evaluation method.
    
    This is a standalone function that wraps the static method evaluation
    for use in parallel evaluation contexts (e.g., Dask, multiprocessing).
    
    Args:
        solution (S): Solution to evaluate.
        problem (Problem[S]): Problem instance to use for evaluation.
    
    Returns:
        S: Evaluated solution with objective values set.
    
    Note:
        This function is typically used as a helper for parallel evaluation
        frameworks that need a callable function rather than a method.
    """
    Evaluator[S].evaluate_solution(solution, problem)
    return solution


class DaskEvaluator(Evaluator[S]):
    """Dask-based evaluator for distributed solution evaluation.
    
    This evaluator uses Dask to evaluate solutions in parallel using a flexible
    task scheduling system. Supports multiple schedulers (threads, processes, distributed).
    
    Attributes:
        scheduler (str): Dask scheduler to use ('threads', 'processes', 'distributed', etc.).
    
    Note:
        Requires Dask to be installed. Provides flexibility in choosing the
        parallel execution strategy.
    """
    
    def __init__(self, scheduler: str = "processes") -> None:
        """Initialize Dask evaluator with specified scheduler.
        
        Args:
            scheduler (str, optional): Dask scheduler type. Options include
                'threads', 'processes', or 'distributed'. Defaults to "processes".
        """
        self.scheduler = scheduler

    def evaluate(self, solution_list: List[S], problem: Problem[S]) -> List[S]:
        """Evaluate solutions using Dask.
        
        Args:
            solution_list (List[S]): List of solutions to evaluate.
            problem (Problem[S]): Problem instance to use for evaluation.
        
        Returns:
            List[S]: List of evaluated solutions.
        
        Raises:
            ImportError: If Dask is not installed.
        """
        if dask is None:
            raise ImportError("Dask is not installed. Install it with: pip install dask")
        with dask.config.set(scheduler=self.scheduler):
            return list(
                dask.compute(
                    *[dask.delayed(evaluate_solution)(solution=solution, problem=problem) for solution in solution_list]
                )
            )
