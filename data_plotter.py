""" Plotting features and monitoring stats """

__author__ = "Hussein Saafan"

import numpy as np
import matplotlib.pyplot as plt


class SFAPlotter:

    show = True
    save = False
    figure_text = ""

    def __init__(self, show=True, save=False, figure_text=""):
        self.show = show
        self.save = save
        self.figure_text = figure_text
        return

    def plot_features(self, fig_name, features, eta, num_features=5):
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
        _s = plt.figure(fig_name)
        plt.subplots_adjust(wspace=0.4)
        plt.figtext(0.05, 0.02, self.figure_text)
        titles = ["$T^2$", "$T_e^2$", "$S^2$", "$S_e^2$"]
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.title(titles[i])
            plt.xlabel("Sample")
            plt.plot(stats[i, 3:])
            if stats_crit is not None:
                plt.plot(stats_crit[i, 3:])
        _s.set_size_inches(16, 9)
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(fig_name, dpi=350)
            plt.close(fig=_s)
        return

    def plot_standardized(self, fig_name, Z):
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
