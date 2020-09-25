""" Runs all the SFA algorithms """
import copy

import numpy as np
import matplotlib.pyplot as plt

import tep_import as imp
from data_plotter import SFAPlotter
from sfa import SFA
from incsfa import IncSFA
from rsfa import RSFA


def run_sfa(dynamic_copies=2, expansion_order=1, cut_off=55):
    """ Batch Slow Feature Analysis

    Runs the batch slow feature analysis algorithm on the Tennessee Eastman
    process data set

    Parameters
    ----------
    dynamic_copies: int
        The number of lagged copies per data point
    expansion_order: int
        The order of nonlinear expansion performed on the data
    cut_off: int
        The index to split the features into the slow and fast groups
    """

    """ Import data """
    X, T0, T4, T5, T10 = imp.import_tep_sets()

    """ Create plotter object """
    figure_text = (f"Lagged Copies= {dynamic_copies} | "
                   f"Expansion Order= {expansion_order} | "
                   f"$M_d$= {cut_off}")
    plotter = SFAPlotter(show=False, save=True, figure_text=figure_text)

    """ Train model """
    SlowFeature = SFA(data=X,
                      dynamic_copies=dynamic_copies,
                      expansion_order=expansion_order)
    SlowFeature.delta = 3
    Y = SlowFeature.train()
    SlowFeature.partition_manual(cut_off)
    # Calculate speed indices for features
    eta = np.around((X.shape[1]/(2*np.pi)) *
                    np.sqrt(SlowFeature.features_speed), 2)

    # Plot features
    plotter.plot_features("SFA", Y, eta, num_features=5)

    """ Test model """
    T_dc, T_ec, S_dc, S_ec = SlowFeature.calculate_crit_values()
    data_iterable = [("SFA_IDV(0)", T0), ("SFA_IDV(4)", T4),
                     ("SFA_IDV(5)", T5), ("SFA_IDV(10)", T10)]
    for name, test in data_iterable:
        print("Evaluating test: " + name)
        stats = SlowFeature.calculate_monitors(test).T
        # Create critical value lines for plots
        threshold = np.ones((test.shape[1], 1))
        Tdc = threshold * T_dc
        Tec = threshold * T_ec
        Sdc = threshold * S_dc
        Sec = threshold * S_ec
        stats_crit = np.concatenate((Tdc, Tec, Sdc, Sec), axis=1).T
        # Plot stats
        plotter.plot_monitors(name, stats, stats_crit)

    plt.show()


def run_incsfa(dynamic_copies=2, expansion_order=1, cut_off=55,
               num_whitened_signals=99, num_features=99,
               sample_weight_parameter=2, conv_tol=0, epochs=1,
               plot_last_epoch=True):
    """ Incremental Slow Feature Analysis

    Runs the incremental slow feature analysis algorithm on the Tennessee
    Eastman process data set

    Parameters
    ----------
    dynamic_copies: int
        The number of lagged copies per data point
    expansion_order: int
        The order of nonlinear expansion performed on the data
    cut_off: int
        The index to split the features into the slow and fast groups
    num_whitened_signals: int
        The number of principal components to calculate in the whitening step
    num_features: int
        The number of features to calculate
    sample_weight_parameter: float
        The sample weighting parameter used for setting the learning rate
    conv_tol: float
        The tolerance for convergance
    epochs: int
        The number of times to pass the training data to the model
    plot_last_epoch: boolean
        Only plot the last epoch of data for the features plot
    """

    """ Import data """
    X, T0, T4, T5, T10 = imp.import_tep_sets()
    num_vars, data_points = X.shape

    """ Create plotter object """
    # figure_text = (f"Lagged Copies= {dynamic_copies} | "
    #                f"Expansion Order= {expansion_order} | "
    #                f"$M_d$= {cut_off} | "
    #                f"Epochs: {epochs}  | "
    #                f"K= {num_whitened_signals} | "
    #                f"J= {num_features} | "
    #                f"L= {sample_weight_parameter} | "
    #                f"Tolerance= {conv_tol}")
    # plotter = SFAPlotter(show=False, save=True, figure_text=figure_text)

    """ Train model """
    # Create IncSFA object
    SlowFeature = IncSFA(input_variables=num_vars,
                         num_features=num_features,
                         num_components=num_whitened_signals,
                         L=sample_weight_parameter,
                         expansion_order=expansion_order,
                         dynamic_copies=dynamic_copies,
                         conv_tol=conv_tol)
    SlowFeature.delta = 3
    SlowFeature.Md = cut_off
    SlowFeature.Me = num_features - cut_off

    # Create empty arrays to store outputs
    total_data_points = data_points * epochs
    Y = np.zeros((num_features, total_data_points))
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
            stats_crit[:, pos] = run[2]

    SlowFeature.converged = True

    # if plot_last_epoch:
    #     Y = Y[:, -data_points:]
    #     stats = stats[:, -data_points:]
    #     stats_crit = stats_crit[:, -data_points:]
    # # Calculate speed indices for features
    # eta = np.around(Y.shape[1]/(2*np.pi)
    #                 * np.sqrt(SlowFeature.features_speed), 2)

    # # Plot features
    # plotter.plot_features("IncSFA", Y, eta, num_features=5)

    # """ Test model """
    # test_data = [("IncSFA_IDV(0)", T0), ("IncSFA_IDV(4)", T4),
    #              ("IncSFA_IDV(5)", T5), ("IncSFA_IDV(10)", T10)]
    # for name, test in test_data:
    #     print("Evaluating test: " + name)
    #     test_obj = copy.deepcopy(SlowFeature)
    #     data_points = test.shape[1]
    #     stats = np.zeros((4, data_points))
    #     stats_crit = np.zeros((4, data_points))

    #     for i in range(data_points):
    #         _, stats[:, i], stats_crit[:, i] = test_obj.add_data(test[:, i],
    #                                                              alpha=0.01)
    #     # Plot stats
    #     plotter.plot_monitors(name, stats, stats_crit)

    # plt.show()


def run_rsfa(dynamic_copies=2, expansion_order=1, cut_off=55,
             num_whitened_signals=99, num_features=99,
             sample_weight_parameter=2, epochs=1, conv_tol=0,
             plot_last_epoch=True):
    """ Recursive Slow Feature Analysis

    Runs the recursive slow feature analysis algorithm on the Tennessee
    Eastman process data set

    Parameters
    ----------
    dynamic_copies: int
        The number of lagged copies per data point
    expansion_order: int
        The order of nonlinear expansion performed on the data
    cut_off: int
        The index to split the features into the slow and fast groups
    num_whitened_signals: int
        The number of principal components to calculate in the whitening step
    num_features: int
        The number of features to calculate
    sample_weight_parameter: float
        The sample weighting parameter used for setting the learning rate
    conv_tol: float
        The tolerance for convergance
    epochs: int
        The number of times to pass the training data to the model
    plot_last_epoch: boolean
        Only plot the last epoch of data for the features plot
    """

    """ Import data """
    X, T0, T4, T5, T10 = imp.import_tep_sets()
    num_vars, data_points = X.shape

    """ Create plotter object """
    # figure_text = (f"Lagged Copies= {dynamic_copies} | "
    #                f"Expansion Order= {expansion_order} | "
    #                f"$M_d$= {cut_off} | "
    #                f"Epochs: {epochs}  | "
    #                f"K= {num_whitened_signals} | "
    #                f"J= {num_features} | "
    #                f"L= {sample_weight_parameter} | "
    #                f"Tolerance= {conv_tol}")
    # plotter = SFAPlotter(show=False, save=True, figure_text=figure_text)

    """ Train model """
    # Create RSFA object
    SlowFeature = RSFA(input_variables=num_vars,
                       num_features=num_features,
                       num_components=num_whitened_signals,
                       L=sample_weight_parameter,
                       expansion_order=expansion_order,
                       dynamic_copies=dynamic_copies,
                       conv_tol=conv_tol)
    SlowFeature.delta = 3
    SlowFeature.Md = cut_off
    SlowFeature.Me = num_features - cut_off

    # Create empty arrays to store output
    total_data_points = data_points * epochs
    Y = np.zeros((num_features, total_data_points))
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

    SlowFeature.converged = True

    # if plot_last_epoch:
    #     Y = Y[:, -data_points:]
    #     stats = stats[:, -data_points:]
    #     stats_crit = stats_crit[:, -data_points:]
    # # Calculate speed indices for features
    # eta = np.around(Y.shape[1]/(2*np.pi)
    #                 * np.sqrt(SlowFeature.features_speed), 2)

    # # Plot features
    # plotter.plot_features("RSFA", Y, eta, num_features=5)

    # """ Test model """
    # test_data = [("RSFA_IDV(0)", T0), ("RSFA_IDV(4)", T4),
    #              ("RSFA_IDV(5)", T5), ("RSFA_IDV(10)", T10)]
    # for name, test in test_data:
    #     print("Evaluating test: " + name)
    #     test_obj = copy.deepcopy(SlowFeature)
    #     data_points = test.shape[1]
    #     stats = np.zeros((4, data_points))
    #     stats_crit = np.zeros((4, data_points))

    #     for i in range(data_points):
    #         _, stats[:, i], stats_crit[:, i] = test_obj.add_data(test[:, i],
    #                                                              alpha=0.01)

    #     # Plot stats
    #     plotter.plot_monitors(name, stats, stats_crit)

    # plt.show()


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
