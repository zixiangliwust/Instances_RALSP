# -*- coding: utf-8 -*-
import time
import logging
import threading
from typing import Any, List
from evolu.core.observer import Observable, Observer

LOGGER = logging.getLogger("evolu")
"""
module:: observable
synopsis: Implementation of observable entities (using delegation)
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class DefaultObservable(Observable):
    """Default implementation of the Observable interface.
    
    This class provides a thread-unsafe but efficient implementation of the
    Observable pattern. It maintains a list of observers and notifies them
    when state changes occur.
    
    Attributes:
        observers (List[Observer]): List of registered observers to notify.
    """
    
    def __init__(self) -> None:
        """Initialize default observable with empty observer list."""
        self.observers: List[Observer] = []

    def register(self, observer: Observer) -> None:
        """Register an observer to receive notifications.
        
        Args:
            observer (Observer): Observer to register. Duplicate observers
                are not added multiple times.
        """
        if observer not in self.observers:
            self.observers.append(observer)

    def deregister(self, observer: Observer) -> None:
        """Deregister an observer so it no longer receives notifications.
        
        Args:
            observer (Observer): Observer to remove from the notification list.
        """
        if observer in self.observers:
            self.observers.remove(observer)

    def deregister_all(self) -> None:
        """Deregister all observers.
        
        After calling this method, no observers will receive notifications
        until new ones are registered.
        """
        if self.observers:
            del self.observers[:]

    def notify_all(self, *args: Any, **kwargs: Any) -> None:
        """Notify all registered observers.
        
        Calls update() on each registered observer, passing the provided arguments.
        
        Args:
            *args: Variable positional arguments to pass to observers.
            **kwargs: Variable keyword arguments to pass to observers.
                Common keys include 'EVALUATIONS', 'ITERATIONS', 'COMPUTING_TIME',
                'SOLUTIONS', etc.
        """
        for observer in self.observers:
            observer.update(*args, **kwargs)


class TimeCounter(threading.Thread):
    """Time-based counter that periodically notifies observers.
    
    A background thread that counts time intervals and notifies registered
    observers at regular intervals. The counter increments continuously and
    sends notification events with the current counter value.
    
    Attributes:
        observable (Observable): Observable instance to notify observers.
        delay (int): Delay in seconds between counter increments.
    
    Note:
        This thread runs indefinitely until stopped. Use thread.join() or
        set daemon=True to control its lifetime.
    
    Example:
        >>> counter = TimeCounter(delay=1, observable=my_observable)
        >>> counter.start()  # Start the counter thread
        >>> # Observers will receive COUNTER updates every second
    """
    
    def __init__(self, delay: int, observable: Observable = DefaultObservable()) -> None:
        """Initialize time counter thread.
        
        Args:
            delay (int): Delay in seconds between counter increments and
                observer notifications.
            observable (Observable, optional): Observable instance to use for
                notifying observers. Defaults to DefaultObservable().
        """
        super(TimeCounter, self).__init__()
        self.observable = observable
        self.delay = delay

    def run(self) -> None:
        """Run the time counter loop.
        
        Continuously increments a counter and notifies observers every `delay`
        seconds. The counter value is sent in the observable data dictionary
        with key "COUNTER".
        
        Note:
            This method runs in an infinite loop. To stop it, the thread
            should be terminated or daemon mode should be used.
        """
        counter = 0
        observable_data: dict = {}
        while True:
            time.sleep(self.delay)
            observable_data["COUNTER"] = counter
            self.observable.notify_all(**observable_data)
            counter += 1
