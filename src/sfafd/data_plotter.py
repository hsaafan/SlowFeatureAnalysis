""" Plotting features and monitoring stats

Classes
-------
SFAPlotter: class
    Plotting class for generic plots in paper
"""

__author__ = "Hussein Saafan"

import numpy as np
import matplotlib.pyplot as plt


class SFAPlotter:
    """ Slow Feature Analysis Plotter

    Creates plots from the outputs of the different SFA classes

    Attributes
    ----------
    show: boolean
        Show the plots once they're created
    save: boolean
        Save the plots to the working directory
    figure_text: str
        Caption text at the bottom of each plot
    """

    show = True
    save = False
    figure_text = ""

    def __init__(self, show=True, save=False, figure_text=""):
        """ Class constructor

        Parameters
        ----------
        show: boolean
            Show the plots once they're created
        save: boolean
            Save the plots to the working directory
        figure_text: str
            Caption text at the bottom of each plot
        """
        self.show = show
        self.save = save
        self.figure_text = figure_text

    def plot_features(self, fig_name, features, eta, num_features=5):
        """ Plot the feature outputs

        Plots the slowest, middle, and fastest feature outputs in 3 columns
        with the speed index (eta) as a label on the plots

        Parameters
        ----------
        fig_name: str
            The name of the figure used for the file name if saving
        features: numpy.ndarray
            The numpy array of shape (J, n) containing the feature outputs
            where J is the number of features and n is the number of samples
        eta: numpy.ndarray
            The array of shape (J, ) which contains the speed indices of the
            different features being plotted
        num_features: int
            The number of features to plot in each column
        """
        _f = plt.figure(fig_name)
        plt.figtext(0.25, 0.05, self.figure_text)

        mid = int(features.shape[0]/2)
        slowest = features[:num_features, :]
        middle = features[mid:mid+num_features, :]
        fastest = features[-num_features:, :]

        speed_slowest = eta[:num_features]
        speed_middle = eta[mid:mid+num_features]
        speed_fastest = eta[-num_features:]

        y_range = (-3, 3)
        for i in range(num_features):
            # Subscripts for eta
            e1 = i + 1
            e2 = mid + i + 1
            e3 = len(eta) - num_features + i + 1
            # Plot labels
            lab1 = "$\eta_{" + f"{e1}" + "}$: " + f"{speed_slowest[i]}"
            lab2 = "$\eta_{" + f"{e2}" + "}$: " + f"{speed_middle[i]}"
            lab3 = "$\eta_{" + f"{e3}" + "}$: " + f"{speed_fastest[i]}"

            # Plot slowest
            plt.subplot(num_features, 3, 3*i+1)
            plt.plot(slowest[i, :], label=lab1)
            plt.ylim(y_range)
            plt.legend(loc="lower left")
            if i == 0:
                plt.title("Slowest")

            # Plot middle
            plt.subplot(num_features, 3, 3*i+2)
            plt.plot(middle[i, :], label=lab2)
            plt.ylim(y_range)
            plt.legend(loc="lower left")
            if i == 0:
                plt.title("Middle")

            # Plot fastest
            plt.subplot(num_features, 3, 3*i+3)
            plt.plot(fastest[i, :], label=lab3)
            plt.ylim(y_range)
            plt.legend(loc="lower left")
            if i == 0:
                plt.title("Fastest")

        _f.set_size_inches(16, 9)
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(fig_name, dpi=350)
            plt.close(fig=_f)

        return

    def plot_monitors(self, fig_name, stats, stats_crit=None):
        """ Plot the 4 monitoring statistics

        Plots the four different monitoring statistics and their critical
        values onto one figure

        Parameters
        ----------
        fig_name: str
            The name of the figure used for the file name if saving
        stats: numpy.ndarray
            The numpy array of shape (4, n) containing the monitoring stats
            where n is the number of samples
        stats_crit: numpy.ndarray
            The numpy array of shape (4, n) containing the critical monitoring
            stats where n is the number of samples
        """
        _s = plt.figure(fig_name)
        plt.subplots_adjust(wspace=0.4)
        plt.figtext(0.05, 0.02, self.figure_text)
        titles = ["$T^2$", "$T_e^2$", "$S^2$", "$S_e^2$"]
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.ylabel(titles[i])
            plt.xlabel("Sample")
            plt.plot(stats[i, 3:])
            if stats_crit is not None:
                plt.plot(stats_crit[i, 3:], linestyle='--')
        _s.set_size_inches(8, 6)
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(fig_name, dpi=350)
            plt.close(fig=_s)
        return

    def plot_standardized(self, fig_name, Z):
        """ Plot the first 60 whitened signals

        Plots the first 60 whitened signals which can be useful for
        debugging

        Parameters
        ----------
        fig_name: str
            The name of the figure used for the file name if saving
        Z: numpy.ndarray
            The numpy array of shape (K, n) containing the monitoring stats
            where n is the number of samples and K >= 60 is the number of
            whitened signals
        """
        # Plots first 60 variables of Z
        for i in range(3):
            _z = plt.figure(fig_name)
            plt.figtext(0.05, 0.02, self.figure_text)
            for j in range(20):
                plt.subplot(5, 4, j+1)
                plt.xlabel("Sample")
                plt.plot(Z[20*i + j, :])
            _z.set_size_inches(8, 6)
            if self.show:
                plt.show()
            if self.save:
                plt.savefig(fig_name + str(i), dpi=350)
                plt.close(fig=_z)
                _z = None
        return

    def plot_contributions(self, fig_name, title, contributions, n_to_plot=20):
        """ Plot the top n_to_plot contributing variables to a faulty sample

        Parameters
        ----------
        fig_name: str
            The name of the figure used for the file name if saving
        title: str
            The title on the figure
        contributions: numpy.ndarray
            The numpy array of shape (n, ) containing the variable
            fault contributions of a sample
        n_to_plot: int
            The number of contributions to plot
        """
        _f, ax = plt.subplots()
        _f.set_size_inches(8, 6)

        order = np.argsort(-1 * contributions)
        ordered_cont = contributions[order]
        cum_percent = np.cumsum(ordered_cont) / np.sum(contributions)
        bar_labels = [str(x) for x in order]

        ax.bar(bar_labels[:n_to_plot], ordered_cont[:n_to_plot])
        ax2 = ax.twinx()
        ax2.plot(cum_percent[:n_to_plot], 'r')

        ax.set_title(title)
        ax.set_xlabel("Variable Index")
        ax.set_ylabel("Fault Contribution")
        ax2.set_ylabel("Cumulative Contribution")
        ax2.set_ylim([0, 1])

        plt.figtext(0.05, 0.02, self.figure_text)
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(fig_name, dpi=350)
            plt.close(fig=_f)
            _f = None
