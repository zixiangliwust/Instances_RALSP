import logging
from typing import List, Tuple, TypeVar

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from evolu.core.exceptions import EmptyFrontException, InvalidParameterException

logger = logging.getLogger(__name__)
S = TypeVar("S")


class Plot:
    """Plot class for visualizing Pareto fronts and solutions.
    
    This class provides flexible plotting capabilities for multi-objective
    optimization results. It automatically selects the appropriate visualization
    based on the number of objectives:
    - 2D: Scatter plot (for 2 objectives)
    - 3D: 3D scatter plot (for 3 objectives)
    - Parallel coordinates: For 4+ objectives
    
    Attributes:
        plot_title (str): Title of the plot.
        axis_labels (list, optional): Labels for axes. If None, uses default labels.
        reference_point (list, optional): Reference point(s) to display on the plot.
            Can be a single point or list of points.
        reference_front (List[S], optional): Reference Pareto front to display
            alongside the obtained front (e.g., true Pareto front).
        dimension (int, optional): Number of objectives (automatically determined).
    """
    
    def __init__(self,
                 title: str = "Pareto front approximation",
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 axis_labels: list = None,
                 ):
        """Initialize plot with title and reference information.
        
        Args:
            title (str, optional): Title of the plot. Defaults to "Pareto front approximation".
            reference_front (List[S], optional): Reference Pareto front solutions to display.
                Defaults to None.
            reference_point (list, optional): Reference point(s) to display. Can be a single
                point [x, y] or list of points [[x1, y1], [x2, y2], ...]. Defaults to None.
            axis_labels (list, optional): Labels for plot axes. Length should match
                number of objectives. Defaults to None.
        """
        self.plot_title = title
        self.axis_labels = axis_labels
        if reference_point and not isinstance(reference_point[0], list):
            reference_point = [reference_point]
        self.reference_point = reference_point
        self.reference_front = reference_front
        self.dimension = None

    @staticmethod
    def get_points(solutions: List[S]) -> Tuple[pd.DataFrame, int]:
        """Extract objective values from solutions into a DataFrame.
        
        Args:
            solutions (List[S]): List of solution objects.
        
        Returns:
            Tuple[pd.DataFrame, int]: Tuple containing:
                - DataFrame with one row per solution and one column per objective
                - Number of objectives (dimensionality)
        
        Raises:
            EmptyFrontException: If solutions is None.
        """
        if solutions is None:
            raise EmptyFrontException("Front is None")
        points = pd.DataFrame(list(solution.objectives for solution in solutions))
        return points, points.shape[1]

    def plot(self, front, label="", normalize: bool = False, file_name: str = None, format: str = "eps"):
        """Plot Pareto front(s) with automatic visualization selection.
        
        Automatically selects 2D, 3D, or parallel coordinates plot based on
        the number of objectives in the solutions.
        
        Args:
            front: Single Pareto front (List[S]) or list of fronts (List[List[S]]).
            label (str or List[str], optional): Label(s) for the front(s). If multiple
                fronts, provide list of labels. Defaults to "".
            normalize (bool, optional): Whether to normalize data for parallel coordinates.
                Only used for 4+ objectives. Defaults to False.
            file_name (str, optional): Output file path (without extension). If None,
                plot is displayed but not saved. Defaults to None.
            format (str, optional): Output file format ('eps', 'png', 'pdf', etc.).
                Defaults to "eps".
        
        Raises:
            InvalidParameterException: If number of fronts and labels don't match.
        
        Note:
            - For 2 objectives: Creates 2D scatter plot
            - For 3 objectives: Creates 3D scatter plot
            - For 4+ objectives: Creates parallel coordinates plot
        """
        if not isinstance(front[0], list):
            front = [front]
        if not isinstance(label, list):
            label = [label]
        if len(front) != len(label):
            raise InvalidParameterException("Number of fronts and labels must be the same")
        dimension = front[0][0].number_of_objectives
        if dimension == 2:
            self.two_dim(front, label, file_name, format)
        elif dimension == 3:
            self.three_dim(front, label, file_name, format)
        else:
            self.pcoords(front, normalize, file_name, format)

    def two_dim(self, fronts: List[list], labels: List[str] = None, file_name: str = None, format: str = "eps"):
        """
        Plot any arbitrary number of fronts in 2D.
        param fronts: List of fronts (containing solutions).
        param labels: List of fronts title (if any).
        param file_name: Output file_name.
        """
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)
        reference = None
        if self.reference_front:
            reference, _ = self.get_points(self.reference_front)
        for i, _ in enumerate(fronts):
            points, _ = self.get_points(fronts[i])
            ax = fig.add_subplot(n, n, i + 1)
            points.plot(kind="scatter", x=0, y=1, ax=ax, s=10, color="#236FA4", alpha=1.0)
            if labels:
                ax.set_title(labels[i])
            if self.reference_front:
                reference.plot(x=0, y=1, ax=ax, color="k", legend=False)
            if self.reference_point:
                for point in self.reference_point:
                    plt.plot([point[0]], [point[1]], marker="o", markersize=5, color="r")
                    plt.axvline(x=point[0], color="r", linestyle=":")
                    plt.axhline(y=point[1], color="r", linestyle=":")
            if self.axis_labels:
                plt.xlabel(self.axis_labels[0])
                plt.ylabel(self.axis_labels[1])
        if file_name:
            _filename = file_name + "." + format
            plt.savefig(_filename, format=format, dpi=1000)
            logger.info("Figure {_filename} saved to file")
        else:
            plt.show()
        plt.close(fig=fig)

    def three_dim(self, fronts: List[list], labels: List[str] = None, file_name: str = None, format: str = "eps"):
        """
        Plot any arbitrary number of fronts in 3D.
        param fronts: List of fronts (containing solutions).
        param labels: List of fronts title (if any).
        param file_name: Output file_name.
        """
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)
        for i, _ in enumerate(fronts):
            ax = fig.add_subplot(n, n, i + 1, projection="3d")
            ax.scatter(
                [s.objectives[0] for s in fronts[i]],
                [s.objectives[1] for s in fronts[i]],
                [s.objectives[2] for s in fronts[i]],
            )
            if labels:
                ax.set_title(labels[i])
            if self.reference_front:
                ax.scatter(
                    [s.objectives[0] for s in self.reference_front],
                    [s.objectives[1] for s in self.reference_front],
                    [s.objectives[2] for s in self.reference_front],
                )
            if self.reference_point:
                # todo
                pass
            ax.relim()
            ax.autoscale_view(True, True, True)
            ax.view_init(elev=30.0, azim=15.0)
            ax.locator_params(nbins=4)
        if file_name:
            _filename = file_name + "." + format
            plt.savefig(_filename, format=format, dpi=1000)
            logger.info("Figure {_filename} saved to file")
        else:
            plt.show()
        plt.close(fig=fig)

    def pcoords(self, fronts: List[list], normalize: bool = False, file_name: str = None, format: str = "eps"):
        """
        Plot any arbitrary number of fronts in parallel coordinates.
        param fronts: List of fronts (containing solutions).
        param file_name: Output file_name.
        """
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)
        for i, _ in enumerate(fronts):
            points, _ = self.get_points(fronts[i])
            if normalize:
                points = (points - points.min()) / (points.max() - points.min())
            ax = fig.add_subplot(n, n, i + 1)
            min_, max_ = points.values.min(), points.values.max()
            points["scale"] = np.linspace(0, 1, len(points)) * (max_ - min_) + min_
            pd.plotting.parallel_coordinates(points, "scale", ax=ax)
            ax.get_legend().remove()
            if self.axis_labels:
                ax.set_xticklabels(self.axis_labels)
        if file_name:
            plt.savefig(file_name + "." + format, format=format, dpi=1000)
        else:
            plt.show()
        plt.close(fig=fig)
