import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

LIST_LINESTYLES = [
    '-',  # solid line style
    '--',  # dashed line style
    '-.',  # dash-dot line style
    ':',  # point marker
    's',  # square marker
    '*',  # star marker
    'p',  # pentagon marker
    '+',  # plus marker
    'x',  # x marker
    'd',  # thin diamond marker
]
LIST_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def clean_file_name(file_name: str) -> str:
    """Clean file name by removing special characters.
    
    Removes special characters and non-ASCII characters from a file name to
    ensure cross-platform compatibility and avoid file system issues.
    
    Args:
        file_name (str): Original file name that may contain special characters.
    
    Returns:
        str: Cleaned file name with special characters removed and underscores
            replaced with hyphens.
    
    Note:
        - Removes non-ASCII characters
        - Removes special characters: `~!@#$%^&*:,\<>;+|`
        - Replaces underscores with hyphens (though this replacement appears
          to be incomplete in the current implementation)
    """
    chars_to_remove: List[str] = ["`", "~", "!", "@", "#", "$", "%", "^", "&", "*", ":", ",", "<", ">", ";", "+", "|"]
    regular_expression: str = '[' + re.escape(''.join(chars_to_remove)) + ']'
    temp: bytes = file_name.encode("ascii", "ignore")
    file_name = temp.decode()  # Removed all non-ascii characters
    file_name = re.sub(regular_expression, '', file_name)  # Removed all special characters
    file_name.replace("_", "-")  # Replaced _ by -
    return file_name


def check_file_path(file_name: str) -> str:
    """Check and create file path directory if necessary.
    
    Ensures the directory containing the file exists, creating parent
    directories if needed. Standardizes path separators.
    
    Args:
        file_name (str): File path that may include directory components.
    
    Returns:
        str: Same file path with directory created if needed.
    
    Note:
        - Converts backslashes to forward slashes for better cross-platform
          compatibility
        - Creates parent directories if they don't exist
        - Uses pathlib for directory creation
    """
    file_name.replace("\\", "/")  # For better handling the parent folder
    if "/" in file_name:
        list_names: List[str] = file_name.split("/")[:-1]  # Remove last element because it is file_name
        filepath: str = "/".join(list_names)
        Path(filepath).mkdir(parents=True, exist_ok=True)
    return file_name


def draw_line(data: Optional[List[float]] = None, title: Optional[str] = None, linestyle: str = '-', color: str = 'b',
              x_label: str = "#Iteration", y_label: str = "Function Value",
              file_name: Optional[str] = None, exts: Tuple[str, ...] = (".png", ".pdf"), verbose: bool = True) -> None:
    """Draw a simple line chart.
    
    Creates a line chart from a list of data points. The x-axis represents
    the iteration/step number, and the y-axis represents the function value.
    Useful for visualizing convergence or performance over time.
    
    Args:
        data (Optional[List[float]]): List of y-values to plot. If None, no
            plot is created. Defaults to None.
        title (Optional[str]): Plot title. Defaults to None.
        linestyle (str): Line style (e.g., '-', '--', '-.'). Defaults to '-'.
        color (str): Line color. Defaults to 'b' (blue).
        x_label (str): Label for x-axis. Defaults to "#Iteration".
        y_label (str): Label for y-axis. Defaults to "Function Value".
        file_name (Optional[str]): Output file path (without extension). If None,
            plot is displayed but not saved. Defaults to None.
        exts (Tuple[str, ...]): Tuple of file extensions to save (e.g., (".png", ".pdf")).
            Defaults to (".png", ".pdf").
        verbose (bool): If True, displays the plot briefly. Defaults to True.
    
    Note:
        - X-axis values are automatically generated as indices (0, 1, 2, ...)
        - Plot is automatically closed after saving/displaying
    """
    if data is None:
        return
    x: np.ndarray = np.arange(0, len(data))
    y: List[float] = data
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y, linestyle=linestyle, color=color, )
    plt.legend()  # show a legend on the plot
    if file_name is not None:
        filepath = check_file_path(clean_file_name(file_name))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
    if verbose:
        plt.pause(0.1)  # plt.show()
    plt.close()


def draw_multi_lines(data: Optional[List[List[float]]] = None, title: Optional[str] = None,
                     list_legends: Optional[List[str]] = None, list_styles: Optional[List[str]] = None,
                     list_colors: Optional[List[str]] = None,
                     x_label: str = "#Iteration", y_label: str = "Function Value",
                     file_name: Optional[str] = None, exts: Tuple[str, ...] = (".png", ".pdf"),
                     verbose: bool = True) -> None:
    x = np.arange(0, len(data[0]))
    for idx, y in enumerate(data):
        plt.plot(x, y, label=list_legends[idx], markerfacecolor=list_colors[idx], linestyle=list_styles[idx])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()  # show a legend on the plot
    if file_name is not None:
        filepath = check_file_path(clean_file_name(file_name))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
    if verbose:
        plt.pause(0.1)  # plt.show()
    plt.close()


def draw_multi_line_in_same_figure(data=None, title=None, list_legends=None, list_styles=None, list_colors=None,
                                   x_label="#Iteration", y_label="Objective", file_name=None, exts=(".png", ".pdf"),
                                   verbose=True):
    n_lines = len(data)
    len_lines = len(data[0])
    x = np.arange(0, len_lines)
    if n_lines == 1:
        fig, ax = plt.subplots()
        if list_legends is None:
            ax.plot(x, data[0])
        else:
            ax.plot(x, data[0], label=list_legends[0])
        ax.set_title(title)
    elif n_lines > 1:
        fig, ax_list = plt.subplots(n_lines, sharex=True)
        fig.suptitle(title)
        for idx, ax in enumerate(ax_list):
            if list_legends is None:
                ax.plot(x, data[idx], markerfacecolor=list_colors[idx], linestyle=list_styles[idx])
            else:
                ax.plot(x, data[idx], label=list_legends[idx], markerfacecolor=list_colors[idx],
                        linestyle=list_styles[idx])
            ax.set_ylabel(f"Objective {idx + 1}")
            if idx == (n_lines - 1):
                ax.set_xlabel(x_label)
    if file_name is not None:
        filepath = check_file_path(clean_file_name(file_name))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
    if verbose:
        plt.pause(0.1)  # plt.show()
    plt.close()


def export_convergence_chart(data=None, title="Convergence Chart", linestyle='-', color='b', x_label="#Iteration",
                             y_label="Function Value", file_name="convergence_chart", exts=(".png", ".pdf"),
                             verbose=True):
    draw_line(data, title=title, linestyle=linestyle, color=color, x_label=x_label, y_label=y_label,
              file_name=file_name, exts=exts, verbose=verbose)


def export_explore_exploit_chart(data=None, title="Exploration vs Exploitation Percentages",
                                 list_legends=("Exploration %", "Exploitation %"), list_styles=('-', '-'),
                                 list_colors=('blue', 'orange'), x_label="#Iteration", y_label="Percentage",
                                 file_name="explore_exploit_chart", exts=(".png", ".pdf"), verbose=True):
    draw_multi_lines(data=data, title=title, list_legends=list_legends, list_styles=list_styles,
                     list_colors=list_colors,
                     x_label=x_label, y_label=y_label, file_name=file_name, exts=exts, verbose=verbose)


def export_diversity_chart(data=None, title='Diversity Measurement Chart', list_legends=None, list_styles=None,
                           list_colors=None, x_label="#Iteration", y_label="Diversity Measurement",
                           file_name="diversity_chart", exts=(".png", ".pdf"), verbose=True):
    if list_styles is None:
        list_styles = LIST_LINESTYLES[:len(data)]
    if list_colors is None:
        list_colors = LIST_COLORS[:len(data)]
    draw_multi_lines(data=data, title=title, list_legends=list_legends, list_styles=list_styles,
                     list_colors=list_colors,
                     x_label=x_label, y_label=y_label, file_name=file_name, exts=exts, verbose=verbose)


def export_objectives_chart(data=None, title="Objectives chart", list_legends=None, list_styles=None, list_colors=None,
                            x_label="#Iteration", y_label="Function Value", file_name="Objective-chart",
                            exts=(".png", ".pdf"), verbose=True):
    if list_styles is None:
        list_styles = LIST_LINESTYLES[:len(data)]
    if list_colors is None:
        list_colors = LIST_COLORS[:len(data)]
    draw_multi_line_in_same_figure(data=data, title=title, list_legends=list_legends, list_styles=list_styles,
                                   list_colors=list_colors,
                                   x_label=x_label, y_label=y_label, file_name=file_name, exts=exts, verbose=verbose)


def export_trajectory_chart(data=None, n_dimensions=1, title="Trajectory of some first agents after generations",
                            list_legends=None, list_styles=None, list_colors=None, x_label="#Iteration", y_label="X1",
                            file_name="1d_trajectory", exts=(".png", ".pdf"), verbose=True):
    if list_styles is None:
        list_styles = LIST_LINESTYLES[:len(data)]
    if list_colors is None:
        list_colors = LIST_COLORS[:len(data)]
    if n_dimensions == 1:
        x = np.arange(0, len(data[0]))
        for idx, y in enumerate(data):
            plt.plot(x, y, label=list_legends[idx], markerfacecolor=list_colors[idx], linestyle=list_styles[idx])
    elif n_dimensions == 2:
        for idx, point in enumerate(data):
            plt.plot(point[0], point[1], label=list_legends[idx], markerfacecolor=list_colors[idx],
                     linestyle=list_styles[idx])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()  # show a legend on the plot
    if file_name is not None:
        filepath = check_file_path(clean_file_name(file_name))
        for idx, ext in enumerate(exts):
            plt.savefig(f"{filepath}{ext}", bbox_inches='tight')
    if verbose:
        plt.pause(0.1)  # plt.show()
    plt.close()
