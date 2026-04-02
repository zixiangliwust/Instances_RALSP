# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, TypeVar
import numpy as np
from tqdm import tqdm
from evolu.core.observer import Observer
from evolu.core.problem import DynamicProblem
from evolu.core.quality_indicator import InvertedGenerationalDistance
from evolu.core.solution import Solution
from evolu.lab.visualization import Plot, StreamingPlot
from evolu.util.solution import print_function_values_to_file

S = TypeVar("S", bound=Solution)
LOGGER = logging.getLogger("evolu")
"""
module:: observer
synopsis: Implementation of algorithm's observers.
moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ProgressBarObserver(Observer):
    """Observer that displays a progress bar using tqdm.
    
    This observer shows a visual progress bar in the console displaying the
    algorithm's progress. It uses the tqdm library to create an animated
    progress meter showing evaluation count.
    
    Attributes:
        progress_bar: tqdm progress bar instance (created on first update).
        progress (int): Current number of evaluations tracked.
        _max (int): Maximum expected evaluations (total for progress bar).
    
    Example:
        >>> observer = ProgressBarObserver(max=10000)
        >>> algorithm.observable.register(observer)
        >>> algorithm.run()  # Progress bar will be displayed
    """
    
    def __init__(self, max: int) -> None:
        """Initialize progress bar observer.
        
        Args:
            max (int): Maximum number of expected evaluations. This sets the
                total for the progress bar.
        """
        self.progress_bar = None
        self.progress = 0
        self._max = max

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update progress bar with current evaluation count.
        
        Args:
            **kwargs: Must contain 'EVALUATIONS' key with current evaluation count.
        
        Note:
            The progress bar is automatically created on the first update call
            and closed when max evaluations is reached.
        """
        if not self.progress_bar:
            self.progress_bar = tqdm(total=self._max, ascii=True, desc="Progress")
        evaluations = kwargs["EVALUATIONS"]
        self.progress_bar.update(evaluations - self.progress)
        self.progress = evaluations
        if self.progress >= self._max:
            self.progress_bar.close()


class BasicObserver(Observer):
    """Observer that logs basic algorithm progress information.
    
    This observer displays essential algorithm information including:
    - Number of evaluations
    - Best fitness value (first solution's objectives)
    - Computing time
    
    Information is logged at a specified frequency to avoid overwhelming output.
    
    Attributes:
        display_frequency (int): Frequency of display updates. Information is
            logged every 'display_frequency' evaluations. Defaults to 1 (every evaluation).
    
    Example:
        >>> observer = BasicObserver(frequency=100)  # Log every 100 evaluations
        >>> algorithm.observable.register(observer)
        >>> algorithm.run()  # Periodic log messages will be displayed
    """
    
    def __init__(self, frequency: int = 1) -> None:
        """Initialize basic observer.
        
        Args:
            frequency (int, optional): Display frequency. Information is logged
                every 'frequency' evaluations. Defaults to 1.
        """
        self.display_frequency = frequency

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update and display basic algorithm information.
        
        Args:
            **kwargs: Must contain 'EVALUATIONS', 'SOLUTIONS', and 'TOTAL_TIME' keys.
        
        Note:
            Only logs when evaluations is a multiple of display_frequency and
            solutions are available. For list of solutions, uses first solution's
            objectives as "best fitness".
        """
        computing_time = kwargs["TOTAL_TIME"]
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives
            LOGGER.info(
                "Evaluations: {} \n Best fitness: {} \n Computing time: {}".format(evaluations, fitness, computing_time)
            )


class PrintObjectivesObserver(Observer):
    """Observer that logs evaluation count and objective values.
    
    This observer displays a simplified view showing only:
    - Number of evaluations
    - Objective values (fitness) of the best solution
    
    More concise than BasicObserver as it omits computing time information.
    
    Attributes:
        display_frequency (int): Frequency of display updates. Information is
            logged every 'display_frequency' evaluations. Defaults to 1.
    
    Example:
        >>> observer = PrintObjectivesObserver(frequency=50)
        >>> algorithm.observable.register(observer)
        >>> algorithm.run()  # Periodic objective logs will be displayed
    """
    
    def __init__(self, frequency: int = 1) -> None:
        """Initialize print objectives observer.
        
        Args:
            frequency (int, optional): Display frequency. Defaults to 1.
        """
        self.display_frequency = frequency

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update and display objective values.
        
        Args:
            **kwargs: Must contain 'EVALUATIONS' and 'SOLUTIONS' keys.
        
        Note:
            Only logs when evaluations is a multiple of display_frequency and
            solutions are available.
        """
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives
            LOGGER.info("Evaluations: {}. fitness: {}".format(evaluations, fitness))


class WriteFrontToFileObserver(Observer):
    """Observer that writes Pareto front approximations to files.
    
    This observer saves solution objective values to files whenever the algorithm
    notifies it. Files are saved with names FUN.0, FUN.1, FUN.2, etc., containing
    one solution per line with space-separated objective values.
    
    For dynamic problems, files are written only when termination criteria are met.
    For static problems, files are written on every update.
    
    Attributes:
        counter (int): Counter for file naming (FUN.{counter}).
        directory (str): Output directory where files are saved.
    
    Example:
        >>> observer = WriteFrontToFileObserver(output_directory="results/")
        >>> algorithm.observable.register(observer)
        >>> algorithm.run()  # Fronts will be saved to results/FUN.0, FUN.1, etc.
    """
    
    def __init__(self, output_directory: str) -> None:
        """Initialize write front to file observer.
        
        Args:
            output_directory (str): Directory path where front files will be saved.
                If the directory exists, its contents will be removed. If it doesn't
                exist, it will be created.
        """
        self.counter = 0
        self.directory = output_directory
        if Path(self.directory).is_dir():
            LOGGER.warning("Directory {} exists. Removing contents.".format(self.directory))
            for file in os.listdir(self.directory):
                os.remove("{0}/{1}".format(self.directory, file))
        else:
            LOGGER.warning("Directory {} does not exist. Creating it.".format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update and write front to file.
        
        Args:
            **kwargs: Must contain 'PROBLEM' and 'SOLUTIONS' keys. May contain
                'TERMINATION_CRITERIA_IS_MET' for dynamic problems.
        
        Note:
            - For dynamic problems: Only writes when termination criteria are met
            - For static problems: Writes on every update
            - Files are saved as FUN.{counter} in the output directory
        """
        problem = kwargs["PROBLEM"]
        solutions = kwargs["SOLUTIONS"]
        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get("TERMINATION_CRITERIA_IS_MET", None)
                if termination_criterion_is_met:
                    print_function_values_to_file(solutions, "{}/FUN.{}".format(self.directory, self.counter))
                    self.counter += 1
            else:
                print_function_values_to_file(solutions, "{}/FUN.{}".format(self.directory, self.counter))
                self.counter += 1


class PlotFrontToFileObserver(Observer):
    """Observer that plots and saves Pareto front approximations as image files.
    
    This observer creates visualization plots of the Pareto front approximations
    and saves them as image files at specified intervals. Useful for tracking
    the evolution of the Pareto front during optimization.
    
    Attributes:
        directory (str): Output directory where plot images are saved.
        plot_front (Plot): Plot object for creating front visualizations.
        last_front (List[S]): Previous front for comparison (used in dynamic problems).
        fronts (List[S]): Accumulated fronts (for dynamic problems).
        counter (int): Counter for file naming.
        step (int): Number of evaluations between plot saves.
    
    Example:
        >>> observer = PlotFrontToFileObserver(output_directory="plots/", step=500)
        >>> algorithm.observable.register(observer)
        >>> algorithm.run()  # Plots saved every 500 evaluations
    """
    
    def __init__(self, output_directory: str, step: int = 100, **kwargs) -> None:
        """Initialize plot front to file observer.
        
        Args:
            output_directory (str): Directory path where plot images will be saved.
                If the directory exists, its contents will be removed. If it doesn't
                exist, it will be created.
            step (int, optional): Number of evaluations between plot saves. Defaults to 100.
            **kwargs: Additional keyword arguments passed to Plot constructor
                (e.g., figure size, axis labels, etc.).
        """
        self.directory = output_directory
        self.plot_front = Plot(title="Pareto front approximation", **kwargs)
        self.last_front = []
        self.fronts = []
        self.counter = 0
        self.step = step
        if Path(self.directory).is_dir():
            LOGGER.warning("Directory {} exists. Removing contents.".format(self.directory))
            for file in os.listdir(self.directory):
                os.remove("{0}/{1}".format(self.directory, file))
        else:
            LOGGER.warning("Directory {} does not exist. Creating it.".format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update and plot front to file."""
        problem = kwargs["PROBLEM"]
        solutions = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]
        if solutions:
            if (evaluations % self.step) == 0:
                if isinstance(problem, DynamicProblem):
                    termination_criterion_is_met = kwargs.get("TERMINATION_CRITERIA_IS_MET", None)
                    if termination_criterion_is_met:
                        if self.counter > 0:
                            igd = InvertedGenerationalDistance(np.array([s.objectives for s in self.last_front]))
                            igd_value = igd.compute(np.array([s.objectives for s in solutions]))
                        else:
                            igd_value = 1
                        if igd_value > 0.005:
                            self.fronts += solutions
                            self.plot_front.plot(
                                [self.fronts],
                                label=problem.get_name(),
                                file_name=f"{self.directory}/front-{evaluations}",
                            )
                        self.counter += 1
                        self.last_front = solutions
                else:
                    self.plot_front.plot(
                        [solutions],
                        label=f"{evaluations} evaluations",
                        file_name=f"{self.directory}/front-{evaluations}",
                    )
                    self.counter += 1


class VisualizerObserver(Observer):
    """Observer that creates real-time streaming visualization of solutions.
    
    This observer creates an interactive, updating plot that shows the current
    Pareto front approximation in real-time. The plot updates as the algorithm
    progresses, allowing visual monitoring of convergence.
    
    Attributes:
        figure (StreamingPlot): Streaming plot object for real-time visualization.
        display_frequency (int): Frequency of plot updates. Plot is refreshed
            every 'display_frequency' evaluations. Defaults to 1.
        reference_point (Optional[List[float]]): Reference point for visualization
            (e.g., ideal point, nadir point).
        reference_front (Optional[List[S]]): Reference Pareto front for comparison
            (e.g., true Pareto front for benchmark problems).
    
    Example:
        >>> observer = VisualizerObserver(
        ...     reference_front=true_pareto_front,
        ...     reference_point=ideal_point,
        ...     display_frequency=10
        ... )
        >>> algorithm.observable.register(observer)
        >>> algorithm.run()  # Real-time plot will be displayed and updated
    """
    
    def __init__(self, reference_front: Optional[List[S]] = None, reference_point: Optional[List[float]] = None, 
                 display_frequency: int = 1) -> None:
        """Initialize visualizer observer.
        
        Args:
            reference_front (Optional[List[S]], optional): Reference Pareto front
                to display alongside current front. Defaults to None.
            reference_point (Optional[List[float]], optional): Reference point for
                visualization. Can be updated dynamically during optimization.
                Defaults to None.
            display_frequency (int, optional): Frequency of plot updates. Defaults to 1.
        """
        self.figure = None
        self.display_frequency = display_frequency
        self.reference_point = reference_point
        self.reference_front = reference_front

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update and visualize solutions.
        
        Args:
            **kwargs: Must contain 'EVALUATIONS' and 'SOLUTIONS' keys. May contain
                'REFERENCE_POINT' to dynamically update the reference point.
        
        Note:
            - The plot is created on first update
            - Plot title shows current evaluation count
            - Reference point can be updated dynamically if provided in kwargs
        """
        evaluations = kwargs["EVALUATIONS"]
        solutions = kwargs["SOLUTIONS"]
        if solutions:
            if self.figure is None:
                self.figure = StreamingPlot(reference_point=self.reference_point, reference_front=self.reference_front)
                self.figure.plot(solutions)
            if (evaluations % self.display_frequency) == 0:
                # check if reference point has changed
                reference_point = kwargs.get("REFERENCE_POINT", None)
                if reference_point:
                    self.reference_point = reference_point
                    self.figure.update(solutions, reference_point)
                else:
                    self.figure.update(solutions)
                self.figure.ax.set_title("Evaluations: {}".format(evaluations), fontsize=13)
