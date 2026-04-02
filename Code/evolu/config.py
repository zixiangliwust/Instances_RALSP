# -*- coding: utf-8 -*-
"""Configuration module providing default instances for evolu framework components.
 
This module provides a singleton store containing default instances of commonly
used components (observable, evaluator, generator, termination criteria, etc.).
These defaults are used by algorithms and other components when no specific
instance is provided, reducing boilerplate code.
"""

from evolu.core.observer import Observable
from evolu.operator.mutation import BitFlipMutation, FloatPolynomialMutation
from evolu.util.comparator import DominanceComparator
from evolu.util.evaluator import Evaluator, SequentialEvaluator
from evolu.util.generator import RandomGenerator
from evolu.util.observable import DefaultObservable
from evolu.util.termination_criterion import StoppingByEvaluations


class _Store:
    """Singleton store for default framework component instances.
    
    This class provides lazy-initialized default instances of commonly used
    components. All properties use the singleton pattern to avoid creating
    multiple instances when the same default is needed.
    
    Access these defaults via the global 'store' instance:
        store.default_observable
        store.default_evaluator
        store.default_generator
        etc.
    """
    
    def __init__(self):
        """Initialize default instances to avoid repeated creation."""
        self._default_observable = None
        self._default_evaluator = None
        self._default_generator = None
        self._default_termination_criteria = None
        self._default_comparator = None
        self._default_mutation = None

    @property
    def default_observable(self) -> Observable:
        """Get default observable instance (singleton).
        
        Returns:
            Observable: DefaultObservable instance for managing observers.
        
        Note:
            Created on first access and reused for all subsequent accesses.
        """
        if self._default_observable is None:
            self._default_observable = DefaultObservable()
        return self._default_observable

    @property
    def default_evaluator(self) -> Evaluator:
        """Get default evaluator instance (singleton).
        
        Returns:
            Evaluator: SequentialEvaluator instance for solution evaluation.
        
        Note:
            Uses sequential (single-threaded) evaluation by default. For parallel
            evaluation, create MultiprocessEvaluator or other evaluator instances.
        """
        if self._default_evaluator is None:
            self._default_evaluator = SequentialEvaluator()
        return self._default_evaluator

    @property
    def default_generator(self):
        """Get default generator instance (singleton).
        
        Returns:
            Generator: RandomGenerator instance for creating random solutions.
        
        Note:
            Creates random solutions by calling problem.create_solution().
        """
        if self._default_generator is None:
            self._default_generator = RandomGenerator()
        return self._default_generator

    @property
    def default_termination_criteria(self):
        """Get default termination criteria instance (singleton).
        
        Returns:
            TerminationCriterion: StoppingByEvaluations instance with
                max_evaluations=25000.
        
        Note:
            Stops algorithm after 25,000 function evaluations by default.
            Customize by creating your own termination criterion instances.
        """
        if self._default_termination_criteria is None:
            self._default_termination_criteria = StoppingByEvaluations(max_evaluations=25000)
        return self._default_termination_criteria

    @property
    def default_comparator(self):
        """Get default comparator instance (singleton).
        
        Returns:
            Comparator: DominanceComparator instance for comparing solutions
                based on Pareto dominance.
        
        Note:
            Uses standard Pareto dominance comparison. For constrained problems,
            use DominanceWithConstraintsComparator instead.
        """
        if self._default_comparator is None:
            self._default_comparator = DominanceComparator()
        return self._default_comparator

    @property
    def default_mutation(self):
        """Get default mutation operators dictionary (singleton).
        
        Returns:
            dict: Dictionary mapping solution types to default mutation operators.
                Keys: "real" (FloatPolynomialMutation), "binary" (BitFlipMutation).
        
        Note:
            - "real": FloatPolynomialMutation with probability=0.15, distribution_index=20
            - "binary": BitFlipMutation with probability=0.15
        
        Example:
            >>> mutation = store.default_mutation["real"]  # For float solutions
            >>> mutation = store.default_mutation["binary"]  # For binary solutions
        """
        if self._default_mutation is None:
            self._default_mutation = {
                "real": FloatPolynomialMutation(probability=0.15, distribution_index=20),
                "binary": BitFlipMutation(0.15)
            }
        return self._default_mutation


# Global singleton store instance
store = _Store()
