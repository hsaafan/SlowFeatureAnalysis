""" Runs all the SFA algorithms """
import copy

import numpy as np
import matplotlib.pyplot as plt

import tep_import as imp
from data_plotter import SFAPlotter
from sfa import SFA
from incsfa import IncSFA
from rsfa import RSFA


# TODO: Document all these functions and clean them up
def run_sfa():
    # Import TEP
    X, T0, T4, T5, T10 = imp.import_tep_sets()
    plotter = SFAPlotter(show=False, save=True, figure_text="")
    d = 2    # Lagged copies
    q = 0.1  # Partition fraction
    n = 1    # Expansion order
    Md = 55  # Slow features to keep

    # Run SFA
    SlowFeature = SFA(X, d, n)
    SlowFeature.delta = 3
    Y = SlowFeature.train()
    SlowFeature.partition(q)
    SlowFeature.partition_manual(Md)
    eta = np.around((X.shape[1]/(2*np.pi)) *
                    np.sqrt(SlowFeature.features_speed), 2)

    # Plot slow features
    plotter.plot_features("SFA", Y, eta, num_features=5)

    # Plot monitors for test data
    T_dc, T_ec, S_dc, S_ec = SlowFeature.calculate_crit_values()
    data_iterable = [("IDV(0)", T0), ("IDV(4)", T4),
                     ("IDV(5)", T5), ("IDV(10)", T10)]
    num_data = len(data_iterable)
    for name, test in data_iterable:
        Td, Te, Sd, Se = SlowFeature.calculate_monitors(test)
        threshold = np.ones(test.shape[1])
        stats = [Td, Te, Sd, Se]
        Tdc = threshold * T_dc
        Tec = threshold * T_ec
        Sdc = threshold * S_dc
        Sec = threshold * S_ec
        stats_crit = [Tdc, Tec, Sdc, Sec]
        plotter.plot_monitors(name, stats, stats_crit)

    plt.show()


def run_incsfa():
    epochs = 5
    plot_last_epoch = True
    # Import data
    X, T0, T4, T5, T10 = imp.import_tep_sets()

    # IncSFA parameters
    num_vars = X.shape[0]
    data_points = X.shape[1]
    theta = [10, 100, 4, 100, 0.001, 0.01, 1000]
    K = 99
    J = 99
    cutoff = 55
    d = 2    # Lagged copies
    n = 1    # Expansion order

    figure_text = (f"Epochs: {epochs} | $t_1=$ {theta[0]} | "
                   f"$t_2=$ {theta[1]} | $c=$ {theta[2]} | "
                   f"$r=$ {theta[3]} | $\eta_l=$ {theta[4]} | "
                   f"$\eta_h=$ {theta[5]} | $T=$ {theta[6]} | "
                   f"$K=$ {K} | $J=$ {J} | Lagged Copies= {d} | "
                   f"Expansion Order= {n}")
    plotter = SFAPlotter(show=False, save=False, figure_text=figure_text)

    # Create IncSFA object
    SlowFeature = IncSFA(num_vars, J, K, theta, n, d)
    SlowFeature.delta = 3
    SlowFeature.Md = cutoff
    SlowFeature.Me = J - cutoff

    # Create empty arrays to store outputs
    total_data_points = data_points * epochs
    Y = np.zeros((J, total_data_points))

    stats = np.zeros((4, total_data_points))
    stats_crit = np.zeros((4, total_data_points))
    # Train model
    for j in range(epochs):
        print("Running epoch " + str(j+1) + "/" + str(epochs))
        for i in range(X.shape[1]):
            pos = j*X.shape[1]+i
            run = SlowFeature.add_data(X[:, i])
            # Store data
            Y[:, pos] = run[0].flat
            stats[:, pos] = run[1]

    if plot_last_epoch:
        Y = Y[:, -data_points:]
        stats = stats[:, -data_points:]
    eta = np.around(Y.shape[1]/(2*np.pi)
                    * np.sqrt(SlowFeature.features_speed), 2)
    order = eta.argsort()
    Y = Y[order, :]
    eta = eta[order]
    # Plot slow features
    plotter.plot_features("IncSFA", Y, eta, num_features=5)
    # Plot monitors for test data
    test_data = [("IDV(0)", T0), ("IDV(4)", T4),
                 ("IDV(5)", T5), ("IDV(10)", T10)]
    num_tests = len(test_data)
    for name, test in test_data:
        print("Evaluating test: " + name)
        data_points = test.shape[1]
        stats = np.zeros((4, data_points))
        stats_crit = np.zeros((4, data_points))

        for i in range(data_points):
            _, stats[:, i], stats_crit[:, i] = SlowFeature.evaluate(test[:, i],
                                                                    alpha=0.01)
        plotter.plot_monitors(name, stats, stats_crit)
    plt.show()


def run_rsfa(plot_last_epoch=True, epochs=5):
    X, T0, T4, T5, T10 = imp.import_tep_sets()

    # RSFA parameters
    num_vars = X.shape[0]
    data_points = X.shape[1]
    d = 2    # Lagged copies
    n = 1    # Expansion order

    # Create RSFA object
    SlowFeature = RSFA(num_vars, n, d)
    SlowFeature.delta = 3
    SlowFeature.Md = 55

    # Create empty arrays to store output
    total_data_points = data_points * epochs
    Y = np.zeros((SlowFeature.output_variables, total_data_points))
    stats = np.zeros((4, total_data_points))
    stats_crit = np.zeros((4, total_data_points))

    for j in range(epochs):
        print("Running epoch " + str(j+1) + "/" + str(epochs))
        for i in range(X.shape[1]):
            pos = j*X.shape[1]+i
            run = SlowFeature.add_data(X[:, i])
            # Store data
            Y[:, pos] = run[0].flat
            stats[:, pos] = run[1]
            stats_crit[:, pos] = run[2]

    if plot_last_epoch:
        Y = Y[:, -data_points:]
        stats = stats[:, -data_points:]
        stats_crit = stats_crit[:, -data_points:]
    # TODO: Order features within class
    y_dot = np.diff(Y) / SlowFeature.delta
    speeds = (y_dot @ y_dot.T) / (Y.shape[1] - 1)
    speeds = np.diag(speeds)
    order = speeds.argsort()
    speeds = speeds[order]
    Y = Y[order, :]
    eta = np.around((Y.shape[1]/(2*np.pi)) * np.sqrt(speeds), 2)

    figure_text = (f"Epochs: {epochs} | $Md=$ {SlowFeature.Md} | "
                   f"Lagged Copies= {d} | Expansion Order= {n}")
    plotter = SFAPlotter(show=False, save=True, figure_text=figure_text)
    plotter.plot_features("RSFA", Y, eta, num_features=5)
    plotter.plot_monitors("RSFA Stats", stats, stats_crit)
    # Plot monitors for test data
    test_data = [("IDV(0)", T0), ("IDV(4)", T4),
                 ("IDV(5)", T5), ("IDV(10)", T10)]
    num_tests = len(test_data)
    for name, test in test_data:
        test_obj = copy.deepcopy(SlowFeature)
        print("Evaluating test: " + name)
        data_points = test.shape[1]
        stats = np.zeros((4, data_points))
        stats_crit = np.zeros((4, data_points))

        for i in range(data_points):
            run = test_obj.add_data(test[:, i])
            # Store data
            stats[:, i] = run[1]
            stats_crit[:, i] = run[2]
        plotter.plot_monitors(name, stats, stats_crit)
    plt.show()


if __name__ == "__main__":
    while True:
        choice = input("Pick an algorithm"
                       "\n[1] SFA"
                       "\n[2] IncSFA"
                       "\n[3] RSFA"
                       "\n[q] Quit"
                       "\nChoice: ")
        if choice == "q":
            break
        if choice.isnumeric():
            choice = int(choice)
            if choice == 1:
                run_sfa()
                break
            elif choice == 2:
                run_incsfa()
                break
            elif choice == 3:
                run_rsfa()
                break
            else:
                print("Invalid Choice")
        else:
            print("Invalid Choice!")
    print("\a\a")
