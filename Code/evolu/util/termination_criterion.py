# -*- coding: utf-8 -*-
import threading
from abc import ABC, abstractmethod
from typing import Any
from evolu.core.observer import Observer
from evolu.core.quality_indicator import QualityIndicator

"""
module:: termination_criterion
synopsis: Implementation of stopping conditions.
moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class TerminationCriterion(Observer, ABC):
    """Base class for algorithm termination criteria.
    
    Termination criteria determine when an optimization algorithm should stop
    running. Common criteria include maximum evaluations, iterations, or time limits.
    This class implements the Observer pattern to receive algorithm updates.
    
    Note:
        Subclasses must implement update(), is_met, and get_name() methods.
    """
    
    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the termination criterion state based on algorithm progress.
        
        This method is called by the algorithm to notify the criterion of
        current algorithm state (evaluations, iterations, etc.).
        
        Args:
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments (typically contains keys like
                'EVALUATIONS', 'ITERATIONS', 'COMPUTING_TIME', etc.).
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `update`.")

    @property
    @abstractmethod
    def is_met(self) -> bool:
        """Check if the termination criterion is met.
        
        Returns:
            bool: True if the algorithm should terminate, False otherwise.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `is_met` property.")

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of the termination criterion.
        
        Returns:
            str: Name of the termination criterion (e.g., "StoppingByEvaluations").
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `name`.")


class StoppingByEvaluations(TerminationCriterion):
    """Termination criterion based on maximum number of function evaluations.
    
    The algorithm stops when the number of objective function evaluations
    reaches or exceeds the specified maximum.
    
    Attributes:
        max_evaluations (int): Maximum number of evaluations allowed.
        evaluations (int): Current number of evaluations performed.
    """
    
    def __init__(self, max_evaluations: int) -> None:
        """Initialize with maximum number of evaluations.
        
        Args:
            max_evaluations (int): Maximum number of function evaluations allowed.
        """
        super(StoppingByEvaluations, self).__init__()
        self.max_evaluations = max_evaluations
        self.evaluations = 0

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update evaluation count from algorithm state.
        
        Args:
            **kwargs: Must contain 'EVALUATIONS' key with current evaluation count.
        """
        self.evaluations = kwargs["EVALUATIONS"]

    @property
    def is_met(self) -> bool:
        """Check if maximum evaluations reached.
        
        Returns:
            bool: True if evaluations >= max_evaluations, False otherwise.
        """
        return self.evaluations >= self.max_evaluations

    def get_name(self) -> str:
        """Get the name of the termination criterion.
        
        Returns:
            str: "StoppingByEvaluations"
        """
        return "StoppingByEvaluations"


class StoppingByIterations(TerminationCriterion):
    """Termination criterion based on maximum number of iterations.
    
    The algorithm stops when the number of iterations reaches or exceeds
    the specified maximum.
    
    Attributes:
        max_iterations (int): Maximum number of iterations allowed.
        iterations (int): Current number of iterations performed.
    """
    
    def __init__(self, max_iterations: int) -> None:
        """Initialize with maximum number of iterations.
        
        Args:
            max_iterations (int): Maximum number of iterations allowed.
        """
        super(StoppingByIterations, self).__init__()
        self.max_iterations = max_iterations
        self.iterations = 0

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update iteration count from algorithm state.
        
        Args:
            **kwargs: Must contain 'ITERATIONS' key with current iteration count.
        """
        self.iterations = kwargs["ITERATIONS"]

    @property
    def is_met(self) -> bool:
        """Check if maximum iterations reached.
        
        Returns:
            bool: True if iterations >= max_iterations, False otherwise.
        """
        return self.iterations >= self.max_iterations

    def get_name(self) -> str:
        """Get the name of the termination criterion.
        
        Returns:
            str: "StoppingByIterations"
        """
        return "StoppingByIterations"


class StoppingByTime(TerminationCriterion):
    """Termination criterion based on maximum execution time.
    
    The algorithm stops when the total execution time reaches or exceeds
    the specified maximum time limit.
    
    Attributes:
        max_seconds (int): Maximum execution time in seconds.
        seconds (float): Current execution time in seconds.
    """
    
    def __init__(self, max_seconds: int) -> None:
        """Initialize with maximum time in seconds.
        
        Args:
            max_seconds (int): Maximum execution time in seconds.
        """
        super(StoppingByTime, self).__init__()
        self.max_seconds = max_seconds
        self.seconds = 0.0

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update elapsed time."""
        self.seconds = kwargs["TOTAL_TIME"]

    @property
    def is_met(self) -> bool:
        """Check if maximum time reached."""
        return self.seconds >= self.max_seconds

    def get_name(self) -> str:
        """Get the name of the termination criterion."""
        return "StoppingByTime"


def key_has_been_pressed(stopping_by_keyboard: "StoppingByKeyboard") -> None:
    """Wait for keyboard input to stop the algorithm.
    
    Helper function that runs in a separate thread, waiting for user input.
    When a key is pressed, sets the key_pressed flag in the StoppingByKeyboard
    instance to signal algorithm termination.
    
    Args:
        stopping_by_keyboard (StoppingByKeyboard): The termination criterion
            instance to update when key is pressed.
    
    Note:
        This function blocks waiting for input. Runs in a separate thread
        to avoid blocking the main algorithm execution.
    """
    input("PRESS ANY KEY + ENTER: ")
    stopping_by_keyboard.key_pressed = True


class StoppingByKeyboard(TerminationCriterion):
    """Termination criterion based on keyboard input.
    
    Allows the user to manually stop the algorithm by pressing any key and
    Enter. Uses a separate thread to monitor keyboard input without blocking
    the main algorithm execution.
    
    Attributes:
        key_pressed (bool): Flag indicating whether a key has been pressed.
    
    Note:
        When initialized, starts a background thread that waits for keyboard
        input. The algorithm stops when the user presses any key followed by
        Enter. Useful for interactive debugging and manual control.
    """
    
    def __init__(self) -> None:
        """Initialize keyboard termination criterion.
        
        Creates a background thread that waits for keyboard input.
        The thread will set key_pressed to True when input is received.
        """
        super(StoppingByKeyboard, self).__init__()
        self.key_pressed = False
        thread = threading.Thread(target=key_has_been_pressed, args=(self,))
        thread.start()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """No-op update method.
        
        Keyboard termination doesn't need to check observable data,
        so this method does nothing.
        """
        pass

    @property
    def is_met(self) -> bool:
        """Check if key has been pressed.
        
        Returns:
            bool: True if user has pressed a key to stop the algorithm,
                False otherwise.
        """
        return self.key_pressed

    def get_name(self) -> str:
        """Get the name of the termination criterion.
        
        Returns:
            str: "StoppingByKeyboard"
        """
        return "StoppingByKeyboard"


class StoppingByQualityIndicator(TerminationCriterion):
    """Termination criterion based on quality indicator value.
    
    Stops the algorithm when a quality indicator (e.g., Hypervolume, IGD)
    reaches a target value. Continuously monitors the quality indicator
    value computed on the current solution set and compares it against
    an expected value with a specified degree.
    
    Attributes:
        quality_indicator (QualityIndicator): Quality indicator to monitor.
        expected_value (float): Target value for the quality indicator.
        degree (float): Degree/multiplier applied to current value for comparison.
        value (float): Current quality indicator value (updated each iteration).
    
    Note:
        For minimization indicators: stops when value * degree < expected_value.
        For maximization indicators: stops when value * degree > expected_value.
        Useful for stopping when a certain solution quality is achieved.
    """
    
    def __init__(self, quality_indicator: QualityIndicator, expected_value: float, degree: float) -> None:
        """Initialize quality indicator termination criterion.
        
        Args:
            quality_indicator (QualityIndicator): Quality indicator to monitor.
            expected_value (float): Target value that should be reached.
            degree (float): Multiplier applied to current value for comparison.
        """
        super(StoppingByQualityIndicator, self).__init__()
        self.quality_indicator = quality_indicator
        self.expected_value = expected_value
        self.degree = degree
        self.value = 0.0

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update quality indicator value from current solutions.
        
        Computes the quality indicator value on the current solution set
        and stores it for comparison in is_met.
        
        Args:
            *args: Variable length argument list.
            **kwargs: Keyword arguments. Expects "SOLUTIONS" key containing
                the current solution list.
        """
        solutions = kwargs["SOLUTIONS"]
        if solutions:
            self.value = self.quality_indicator.compute(solutions)

    @property
    def is_met(self) -> bool:
        """Check if quality indicator meets the expected value.
        
        Returns:
            bool: True if the quality indicator has reached the expected
                value (considering degree multiplier), False otherwise.
        
        Note:
            Comparison depends on whether the indicator is minimization
            or maximization:
            - Minimization: value * degree < expected_value
            - Maximization: value * degree > expected_value
        """
        if self.quality_indicator.is_minimization:
            met = self.value * self.degree < self.expected_value
        else:
            met = self.value * self.degree > self.expected_value
        return met

    def get_name(self) -> str:
        """Get the name of the termination criterion.
        
        Returns:
            str: "StoppingByQualityIndicator"
        """
        return "StoppingByQualityIndicator"
