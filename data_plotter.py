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

        for i in range(num_features):
            plt.subplot(num_features, 3, 3*i+1)
            plt.plot(slowest[i, :], label=f"$\eta$: {speed_slowest[i]}")
            plt.legend(loc="lower left")
            if i == 0:
                plt.title("Slowest")

            plt.subplot(num_features, 3, 3*i+2)
            plt.plot(middle[i, :], label=f"$\eta$: {speed_middle[i]}")
            plt.legend(loc="lower left")
            if i == 0:
                plt.title("Middle")

            plt.subplot(num_features, 3, 3*i+3)
            plt.plot(fastest[i, :], label=f"$\eta$: {speed_fastest[i]}")
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
            plt.subplot(2, 2, i+1)
            plt.title(titles[i])
            plt.xlabel("Sample")
            plt.plot(stats[i])
            if stats_crit is not None:
                plt.plot(stats_crit[i])
        _s.set_size_inches(16, 9)
        if self.show:
            plt.show()
        if self.save:
            plt.savefig(fig_name, dpi=350)
            plt.close(fig=_s)
        return

    def plot_standardized(self, fig_name, Z):
        # Plots first 30 variables of Z
        for i in range(3):
            _z = plt.figure(fig_name)
            plt.figtext(0.05, 0.02, self.figure_text)
            for j in range(10):
                plt.subplot(5, 2, j+1)
                plt.xlabel("Sample")
                plt.plot(Z[10*i + j, :])
            _z.set_size_inches(16, 9)
            if self.show:
                plt.show()
            if self.save:
                plt.savefig(fig_name + str(i), dpi=350)
                plt.close(fig=_z)
                _z = None
        return
