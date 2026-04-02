from abc import ABC, abstractmethod
from typing import Any

"""
module:: Observable
synopsis: Implementation of the observer-observable pattern.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Observer(ABC):
    """Base class for observers in the Observer pattern.
    
    Observers are objects that are notified when the state of an Observable
    object changes. This pattern is used in the framework to allow algorithms
    to notify various components (e.g., termination criteria, loggers, visualizers)
    about their progress without tight coupling.
    
    Note:
        Subclasses must implement the update() method to define how they respond
        to notifications from Observable objects.
    """
    
    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the observer based on notifications from Observable.
        
        This method is called by Observable objects to notify the observer
        of state changes or events.
        
        Args:
            *args: Variable positional arguments passed from Observable.
            **kwargs: Variable keyword arguments passed from Observable.
                Common keys include 'EVALUATIONS', 'ITERATIONS', 'COMPUTING_TIME',
                'SOLUTIONS', etc.
        
        Note:
            The specific arguments passed depend on what the Observable is
            notifying about. Check the Observable implementation for details.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `update`.")


class Observable(ABC):
    """Base class for observable objects in the Observer pattern.
    
    Observable objects maintain a list of Observer objects and notify them
    when their state changes. This allows for decoupled communication between
    components in the framework.
    
    Note:
        Subclasses must implement all abstract methods to manage observers
        and send notifications.
    """
    
    @abstractmethod
    def register(self, observer: Observer) -> None:
        """Register an observer to receive notifications.
        
        Args:
            observer (Observer): The observer to register. Will be notified
                when notify_all() is called.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `register`.")

    @abstractmethod
    def deregister(self, observer: Observer) -> None:
        """Deregister an observer so it no longer receives notifications.
        
        Args:
            observer (Observer): The observer to deregister.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `deregister`.")

    @abstractmethod
    def deregister_all(self) -> None:
        """Deregister all registered observers.
        
        After calling this method, no observers will receive notifications
        until new ones are registered.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `deregister_all`.")

    @abstractmethod
    def notify_all(self, *args: Any, **kwargs: Any) -> None:
        """Notify all registered observers.
        
        This method calls update() on all registered Observer objects, passing
        the provided arguments.
        
        Args:
            *args: Variable positional arguments to pass to observers.
            **kwargs: Variable keyword arguments to pass to observers.
                Common keys include 'EVALUATIONS', 'ITERATIONS', 'COMPUTING_TIME',
                'SOLUTIONS', etc.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement `notify_all`.")
