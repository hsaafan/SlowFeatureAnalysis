import copy

import numpy as np
import matplotlib.pyplot as plt

import tepimport as imp
from data_plotter import SFAPlotter
from sfa import SFA
from incsfa import IncSFA
from rsfa import RSFA

if __name__ == "__main__":
    # Define parameters
    m = 33
    k = 1
    d = 2
    K = 99
    J = 99
    M_d = 55
    alpha = 0.01
    L = 2

    # Import data set
    X, T0, T4, T5, T10 = imp.import_tep_sets(lagged_samples=0)

    # Create models
    ModelSFA = SlowFeature = SFA(data=X,
                                 dynamic_copies=d,
                                 expansion_order=k)
    ModelRSFA = RSFA(input_variables=m,
                     num_features=J,
                     num_components=K,
                     L=L,
                     expansion_order=k,
                     dynamic_copies=d)
    ModelIncSFA = IncSFA(input_variables=m,
                         num_features=J,
                         num_components=K,
                         L=L,
                         expansion_order=k,
                         dynamic_copies=d)

    # Modify models
    ModelSFA.delta = 3
    ModelRSFA.delta = 3
    ModelIncSFA.delta = 3

    ModelRSFA.Md = M_d
    ModelRSFA.Me = J - M_d
    ModelIncSFA.Md = M_d
    ModelIncSFA.Me = J - M_d

    ModelIncSFA.conv_tol = 0
    # Train models
    print("Training SFA")
    SlowFeature.train()
    SlowFeature.partition_manual(M_d)

    print("Training RSFA")
    for i in range(X.shape[1]):
        ModelRSFA.add_data(X[:, i], alpha)

    print("Training IncSFA")
    for i in range(X.shape[1]):
        ModelIncSFA.add_data(X[:, i], alpha)

    # Force convergence
    ModelRSFA.converged = True
    ModelIncSFA.converged = True

    # Test models
    models = [("SFA", ModelSFA), ("RSFA", ModelRSFA), ("IncSFA", ModelIncSFA)]
    tests = [("IDV(0)", T0), ("IDV(4)", T4), ("IDV(5)", T5), ("IDV(10)", T10)]
    SFA_stats = []
    RSFA_stats = []
    IncSFA_stats = []
    for model_name, model_obj in models:
        print(f"Testing {model_name} model")

        for test_name, test_data in tests:
            print(f"Evaluating test: {test_name}")

            data_points = test_data.shape[1]
            if model_name == "SFA":
                stats = model_obj.calculate_monitors(test_data).T
                T_dc, T_ec, S_dc, S_ec = model_obj.calculate_crit_values()
                d_len = np.ones((data_points, 1))
                stats_crit = np.concatenate((d_len * T_dc,
                                             d_len * T_ec,
                                             d_len * S_dc,
                                             d_len * S_ec), axis=1).T
                SFA_stats.append((test_name, stats, stats_crit))
            else:
                model_copy = copy.deepcopy(model_obj)
                stats = np.zeros((4, data_points))
                stats_crit = np.zeros((4, data_points))

                for i in range(data_points):
                    output = model_copy.add_data(test_data[:, i], alpha)
                    _, stats[:, i], stats_crit[:, i] = output
                if model_name == "RSFA":
                    RSFA_stats.append((test_name, stats, stats_crit))
                elif model_name == "IncSFA":
                    IncSFA_stats.append((test_name, stats, stats_crit))

    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.subplots_adjust(0.17, 0.05, 0.95, 0.95, 0, 0.05)
    lag = 5  # To prevent plotting discontinuities in sequential models
    data_array = [("SFA", SFA_stats),
                  ("RSFA", RSFA_stats),
                  ("IncSFA", IncSFA_stats)]
    for model_ind, (model_name, model) in enumerate(data_array):
        for test_ind, (test_name, stats, crits) in enumerate(model):
            if test_name != "IDV(0)":
                continue
            for stat_ind in range(4):
                # plt.subplot(4, 4, 4*stat_ind + test_ind + 1)
                plt.subplot(4, 1, stat_ind + 1)
                plt.plot(stats[stat_ind, lag:], label=model_name)
                # plt.plot(crits[stat_ind, lag:],
                #          label="Control Limit", linestyle="--")
                if stat_ind == 0:
                    plt.title(test_name)
                    plt.legend()
                if stat_ind != 3:
                    plt.xticks([])
                # if test_ind == 0:
                if True:
                    if stat_ind == 0:
                        plt.ylabel("$T^2_d$")
                    if stat_ind == 1:
                        plt.ylabel("$T^2_e$")
                    if stat_ind == 2:
                        plt.ylabel("$S^2_d$")
                    if stat_ind == 3:
                        plt.ylabel("$S^2_e$")
    plt.show()
    print("Done!")
