import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def plot_posterior(
        sample,
        higher_is_better: bool = False,
        min_points_per_hexbin: int = 2,
        alg_names: list = None,
        file_name: str = "posterior.eps",
):
    """Plot posterior distribution from Bayesian statistical test.
    
    Visualizes the posterior probability distribution from a Bayesian comparison
    test (e.g., Bayesian sign test) using a ternary plot. The plot shows the
    probability distribution over three outcomes: algorithm 1 better, algorithms
    equivalent (ROPE), or algorithm 2 better.
    
    The plot uses barycentric coordinates projected onto a 2D triangle, with
    hexbin density visualization to show the concentration of probability mass.
    
    Args:
        sample: Array or DataFrame of shape (n_samples, 3) containing posterior
            probability samples. Each row is a sample [P(alg1<alg2), P(equiv), P(alg1>alg2)].
        higher_is_better (bool, optional): If True, higher objective values indicate
            better performance (affects label interpretation). Defaults to False.
        min_points_per_hexbin (int, optional): Minimum number of points required
            to display a hexbin. Defaults to 2.
        alg_names (list, optional): Names of the two algorithms being compared.
            If None, uses generic labels "Alg1" and "Alg2". Defaults to None.
        file_name (str, optional): Output file path for saving the plot.
            Defaults to "posterior.eps".
    
    Raises:
        ValueError: If sample doesn't have shape (n, 3) or has incorrect dimensions.
    
    Note:
        - The triangle vertices represent the three probability outcomes
        - Hexbin density shows where most probability mass is concentrated
        - Useful for understanding the uncertainty in Bayesian comparisons
    
    Example:
        >>> # From Bayesian sign test
        >>> probs, samples = bayesian_sign_test(data, return_sample=True)
        >>> plot_posterior(samples, alg_names=["NSGA-II", "MOEA/D"])
    """
    # Initial Checking
    if type(sample) == pd.DataFrame:
        sample = sample.values
    if sample.ndim == 2:
        nrow, ncol = sample.shape
        if ncol != 3:
            raise ValueError("Initialization ERROR. Incorrect number of dimensions in axis 1.")
    else:
        raise ValueError("Initialization ERROR. Incorrect number of dimensions for sample")

    def transform(p):
        lambda1, lambda2, lambda3 = p.T
        x = 0.1 * lambda1 + 0.5 * lambda2 + 0.9 * lambda3
        y = (0.2 * lambda1 + 1.4 * lambda2 + 0.2 * lambda3) / np.sqrt(3)
        return np.vstack((x, y)).T

    # Initialize figure
    fig = plt.figure(figsize=(5, 5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    # plot text
    if not higher_is_better:
        if not alg_names:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005, s="P(rope)", ha="center", va="bottom")
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005, s="P(alg1<alg2)", ha="right", va="top")
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005, s="P(alg1>alg2)", ha="left", va="top")
        else:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005, s="P(rope)", ha="center", va="bottom")
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005, s="P(" + alg_names[0] + ")", ha="right", va="top")
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005, s="P(" + alg_names[1] + ")", ha="left", va="top")
    else:
        if not alg_names:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005, s="P(rope)", ha="center", va="bottom")
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005, s="P(alg2<alg1)", ha="right", va="top")
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005, s="P(alg2>alg1)", ha="left", va="top")
        else:
            ax.text(x=0.5, y=1.4 / np.sqrt(3) + 0.005, s="P(rope)", ha="center", va="bottom")
            ax.text(x=0.15, y=0.175 / np.sqrt(3) - 0.005, s="P(" + alg_names[1] + ")", ha="right", va="top")
            ax.text(x=0.85, y=0.175 / np.sqrt(3) - 0.005, s="P(" + alg_names[0] + ")", ha="left", va="top")
    # Conversion between barycentric and Cartesian coordinates
    sample2d = np.zeros((sample.shape[0], 2))
    for p in range(sample.shape[0]):
        sample2d[p, :] = transform(sample[p, :])
    # Plot projected points
    ax.hexbin(sample2d[:, 0], sample2d[:, 1], mincnt=min_points_per_hexbin, cmap=plt.cm.plasma)
    # Plot triangle
    ax.plot([0.095, 0.505], [0.2 / np.sqrt(3), 1.4 / np.sqrt(3)], linewidth=3.0, color="white")
    ax.plot([0.505, 0.905], [1.4 / np.sqrt(3), 0.2 / np.sqrt(3)], linewidth=3.0, color="white")
    ax.plot([0.09, 0.905], [0.2 / np.sqrt(3), 0.2 / np.sqrt(3)], linewidth=3.0, color="white")
    ax.plot([0.1, 0.5], [0.2 / np.sqrt(3), 1.4 / np.sqrt(3)], linewidth=3.0, color="gray")
    ax.plot([0.5, 0.9], [1.4 / np.sqrt(3), 0.2 / np.sqrt(3)], linewidth=3.0, color="gray")
    ax.plot([0.1, 0.9], [0.2 / np.sqrt(3), 0.2 / np.sqrt(3)], linewidth=3.0, color="gray")
    # plot division lines
    ax.plot([0.5, 0.5], [0.2 / np.sqrt(3), 0.6 / np.sqrt(3)], linewidth=3.0, color="gray")
    ax.plot([0.3, 0.5], [0.8 / np.sqrt(3), 0.6 / np.sqrt(3)], linewidth=3.0, color="gray")
    ax.plot([0.5, 0.7], [0.6 / np.sqrt(3), 0.8 / np.sqrt(3)], linewidth=3.0, color="gray")
    if file_name:
        plt.savefig(file_name, bbox_inches="tight")
        logger.info("Figure {file_name} saved to file")
    plt.show()
