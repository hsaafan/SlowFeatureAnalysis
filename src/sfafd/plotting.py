import warnings

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def _order_features(features: np.ndarray, speeds: np.ndarray) -> tuple:
    """ Order the speeds and corresponding features in ascending order

    Parameters
    ----------
    features: np.ndarray
        A numpy array of shape (m, n)
    speeds: np.ndarray
        A numpy array of shape (m)
    """
    if features.ndim != 2:
        raise TypeError("Expected a 2d array for the features")
    if speeds.ndim == 2:
        if not np.allclose(np.diag(np.diag(speeds)), speeds):
            warnings.warn("The feature speeds array is not a diagonal matrix")
        speeds = np.diag(speeds)  # Convert to 1d array
    elif speeds.ndim > 2:
        raise TypeError("Expected a 1d array or 2d diagonal array for the "
                        "feature speeds")

    m = features.shape[0]
    if m != speeds.shape[0]:
        raise RuntimeError("Mismatch between number of features and speeds")

    order = np.argsort(speeds)
    return(speeds[order], features[order, :])

def plot_slowest_features(features: np.ndarray,
                          speeds: np.ndarray,
                          num_features: int = 5,
                          title: str = "Feature Speeds",
                          w_in: float = 16,
                          h_in: float = 9,
                          feature_label: str = "y",
                          sample_label: str = "Sample",
                          speed_label: str = "\eta") -> Figure:
    """ Plot the slowest features

    Plots the slowest feature outputs in rows with the speed as a label on the
    plots

    Parameters
    ----------
    features: np.ndarray
        The numpy array of shape (m, n) containing the feature outputs where m
        is the number of features and n is the number of samples
    speeds: np.ndarray
        The array of shape (m) which contains the speeds of the different
        features being plotted
    num_features: int
        The number of features to plot
    title: str
        The title of the figure
    w_in
        The width in inches of the figure
    h_in
        The height in inches of the figure
    feature_label
        The y axis label corresponding to the feature values
    sample_label
        The x axis label corresponding to the sample number
    speed_label
        The legend label corresponding to the speed of a feature
    """
    if num_features < 1:
        raise ValueError("Must plot at least 1 feature")

    ordered_speeds, ordered_features = _order_features(features, speeds)

    m = ordered_speeds.shape[0]
    if num_features > m:
        warnings.warn(f"Not enough features to plot: {m} features received, "
                      f"{num_features} to be plotted")
        num_features = m

    fig, axs = plt.subplots(nrows=num_features, sharex=True)
    axs[0].set_title(title)
    for i in range(num_features):
        axs[i].plot(ordered_features[i, :],
                    label=f"{speed_label} = {ordered_speeds[i]}")
        axs[i].set_ylabel(f"${feature_label}_{i}$")
        axs[i].legend(loc='upper right')
    axs[num_features - 1].set_xlabel(f"${sample_label}$")
    fig.set_size_inches(w_in, h_in)
    fig.tight_layout()
    return(fig)


def plot_features(features: np.ndarray,
                  speeds: np.ndarray,
                  num_features: int = 5,
                  title_slow: str = "Slowest Features",
                  title_middle: str = "Middle Features",
                  title_fast: str = "Fastest Features",
                  w_in: float = 16,
                  h_in: float = 9,
                  feature_label: str = "y",
                  sample_label: str = "Sample",
                  speed_label: str = "\eta") -> Figure:
    """ Plot the feature outputs

    Plots the slowest, middle, and fastest feature outputs in 3 columns with
    the speeds as a label on the plots

    Parameters
    ----------
    features: np.ndarray
        The numpy array of shape (m, n) containing the feature outputs where m
        is the number of features and n is the number of samples
    speeds: np.ndarray
        The array of shape (m) which contains the speeds of the different
        features being plotted
    num_features: int
        The number of features to plot in each column
    title_slow, title_middle, title_fast: str
        The titles of the slow middle and fast columns
    w_in
        The width in inches of the figure
    h_in
        The height in inches of the figure
    feature_label
        The y axis label corresponding to the feature values
    sample_label
        The x axis label corresponding to the sample number
    speed_label
        The legend label corresponding to the speed of a feature
    """
    if num_features < 1:
        raise ValueError("Must plot at least 1 feature")

    ordered_speeds, ordered_features = _order_features(features, speeds)

    m = ordered_speeds.shape[0]
    if num_features > (m / 3):
        warnings.warn(f"Not enough features to plot: {m} features received, "
                      f"{num_features} to be plotted")
        num_features = int(m / 3)

    start_mid = int((m - num_features) / 2)
    start_end = m - num_features

    fig, axs = plt.subplots(nrows=num_features, ncols=3,
                            sharex=True, sharey=True)

    if num_features > 1:
        axs[0][0].set_title(title_slow)
        axs[0][1].set_title(title_middle)
        axs[0][2].set_title(title_fast)
    else:
        axs[0].set_title(title_slow)
        axs[1].set_title(title_middle)
        axs[2].set_title(title_fast)

    for i in range(3):
        for j in range(num_features):
            if i == 0:
                index = j
            elif i == 1:
                index = start_mid + j
            elif i == 2:
                index = start_end + j

            label = f"${speed_label}_{{{index + 1}}}$ = {ordered_speeds[index]}"
            axs[j][i].plot(ordered_features[index, :], label=label)
            axs[j][i].legend(loc="upper right")
            axs[j][i].set_ylabel(f"${feature_label}_{{{index}}}$")
            if j == num_features - 1:
                axs[j][i].set_xlabel(f"${sample_label}$")
    fig.set_size_inches(w_in, h_in)
    fig.tight_layout()
    return(fig)


def plot_monitors(stats: np.ndarray,
                  stats_crit: np.ndarray,
                  stats_labels: list = ["$T^2_d$", "$T^2_e$",
                                        "$S^2_d$", "$S^2_e$"],
                  title: str = "Monitoring Statistics",
                  w_in: float = 16,
                  h_in: float = 9,
                  sample_label: str = "Sample",
                  limit_label: str = "Control Limit") -> Figure:
    """ Plot the monitoring statistics with their control limits

    Plots the four different monitoring statistics and their critical
    values onto one figure

    Parameters
    ----------
    stats: np.ndarray
        The numpy array of shape (m, n) containing the m monitoring stats
        where n is the number of samples
    stats_crit: np.ndarray
        The numpy array of shape (m, n) or (m) containing the m control limits
    stats_labels: list
        The y axis label corresponding to the monitoring statistic
    title: str
        The title of the figure
    w_in
        The width in inches of the figure
    h_in
        The height in inches of the figure
    sample_label
        The x axis label corresponding to the sample number
    limit_label: str
        The legend label corresponding to the control limit
    """
    if stats.ndim > 2:
        raise TypeError("Expected a 2d array of monitoring statistics")
    elif stats.ndim == 1:
        stats = stats.reshape((1, -1))
    m, n = stats.shape

    if stats_crit.ndim == 1:
        if stats_crit.shape[0] == m:
            stats_crit = stats_crit.reshape((-1, 1)) * np.ones_like(stats)
        elif stats_crit.shape[0] == n:
            stats_crit = stats_crit.reshape((1, -1))

    if stats_crit.shape != stats.shape:
        raise TypeError("Shapes of stats and control limits do not match up")
    if len(stats_labels) != m:
        raise TypeError("Number of labels do not match up to number of "
                        "statistics")

    fig, axs = plt.subplots(nrows=m)

    axs[0].set_title(title)
    for i in range(m):
        axs[i].plot(stats[i, :])
        axs[i].plot(stats_crit[i, :], label=limit_label, linestyle='--')
        axs[i].set_ylabel(stats_labels[i])
        axs[i].legend(loc='upper right')
    axs[m - 1].set_xlabel(sample_label)
    fig.set_size_inches(w_in, h_in)
    fig.tight_layout()
    return(fig)


def plot_contributions(contributions: np.ndarray,
                       variable_labels: list = None,
                       n_to_plot: int = 5,
                       title: str = "Fault Contributions",
                       w_in: float = 16,
                       h_in: float = 9,
                       contribution_label: str = "Contribution",
                       index_label: str = "Variable Index") -> Figure:
    """ Plot the top n_to_plot contributing variables to a faulty sample

    Parameters
    ----------
    contributions: numpy.ndarray
        The numpy array of shape (n, ) containing the variable fault
        contributions of a sample
    n_to_plot: int
        The number of contributions to plot
    variable_labels: list
        The labels describing what each variable refers to, if left as None,
        no textbox will be drawn
    title: str
        The title of the figure
    w_in
        The width in inches of the figure
    h_in
        The height in inches of the figure
    contribution_label: str
        The y axis label corresponding to the contribution value
    index_label: str
        The x axis label corresponding to the variable index
    """
    fig, ax = plt.subplots()

    if n_to_plot > len(contributions):
        warnings.warn("Not enough contribution values to plot")
        n_to_plot = len(contributions)

    # Order contributions
    order = np.argsort(-1 * contributions)
    ordered_cont = contributions[order]
    cum_percent = np.cumsum(ordered_cont) / np.sum(contributions)
    bar_labels = [str(x) for x in order]

    if variable_labels is not None:
        # Box describing index variables
        if len(variable_labels) != len(contributions):
            raise TypeError("Number of labels and contribution do not match")
        ordered_labels = np.asarray(variable_labels)[order]
        text_box = 'Variable Index Descriptions'
        for i in range(n_to_plot):
            text_box += f'\nIndex {order[i]} | {ordered_labels[i]}'
        bbox = dict(boxstyle="square", ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),
                    alpha=0.7)
        ax.text(x=0.1, y=0.75, s=text_box, bbox=bbox, transform=ax.transAxes)

    ax.bar(bar_labels[:n_to_plot], ordered_cont[:n_to_plot])
    ax2 = ax.twinx()
    ax2.plot(cum_percent[:n_to_plot], 'r')

    ax.set_title(title)
    ax.set_xlabel(index_label)
    ax.set_ylabel(contribution_label)
    ax2.set_ylabel(f"Cumulative {contribution_label}")
    ax2.set_ylim([0, 1])
    fig.set_size_inches(w_in, h_in)
    fig.tight_layout()
    return(fig)
