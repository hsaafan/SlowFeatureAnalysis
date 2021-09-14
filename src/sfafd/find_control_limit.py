import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as ST

import tepimport as imp
from incsfa import IncSFA
from sfa import SFA


def incsfa_significance(num_whitened_signals=99,
                        num_features=99,
                        sample_weight_parameter=2,
                        expansion_order=1,
                        dynamic_copies=2,
                        conv_tol=0,
                        cut_off=55,
                        epochs=1,
                        use_SVD=True,
                        normal_data_included=0.99):
    """ Import data """
    X, T0, _, _, _ = imp.import_tep_sets(lagged_samples=0)
    num_vars, data_points = X.shape

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
            run = SlowFeature.add_data(X[:, i],
                                       use_svd_whitening=use_SVD)
            # Store data
            Y[:, pos] = run[0].flat
            stats[:, pos] = run[1]
            stats_crit[:, pos] = run[2]

    SlowFeature.converged = True

    """ Test model """
    data_points = T0.shape[1]
    stats = np.zeros((4, data_points))
    stats_crit = np.zeros((4, data_points))

    ub = 1
    lb = 0
    all_significance = [0, 0, 0, 0]

    for j in range(4):
        while True:
            # Smaller significance includes more data
            significance = (ub + lb) / 2
            print(f"Checking alpha = {significance}")
            test_obj = copy.deepcopy(SlowFeature)
            for i in range(data_points):
                output = test_obj.add_data(T0[:, i], alpha=significance)
                _, stats[:, i], stats_crit[:, i] = output

            normal_fraction = (np.sum(stats[j, :] < stats_crit[j, :])
                               / data_points)

            if normal_fraction < normal_data_included:
                ub = significance
            else:
                ub = 1
                all_significance[j] = significance
                break

    return(all_significance)


def sfa_significance(dynamic_copies=2,
                     expansion_order=1,
                     cut_off=55,
                     normal_data_included=0.99):
    """ Import data """
    X, T0, _, _, _ = imp.import_tep_sets()

    """ Train model """
    SlowFeature = SFA(data=X,
                      dynamic_copies=dynamic_copies,
                      expansion_order=expansion_order)
    SlowFeature.delta = 3
    Y = SlowFeature.train()
    SlowFeature.partition_manual(cut_off)

    """ Test model """
    data_points = T0.shape[1]
    stats = np.zeros((4, data_points))
    stats_crit = np.zeros((4, data_points))

    ub = 1
    lb = 0
    all_significance = [0, 0, 0, 0]

    for j in range(4):
        while True:
            # Smaller significance includes more data
            significance = (ub + lb) / 2
            print(f"Checking alpha = {significance}")
            crits = SlowFeature.calculate_crit_values(alpha=significance)
            stats = SlowFeature.calculate_monitors(T0).T

            normal_fraction = np.sum(stats[j, :] < crits[j]) / data_points

            if normal_fraction < normal_data_included:
                ub = significance
            else:
                ub = 1
                all_significance[j] = significance
                break

    return(all_significance)


def plot_incsfa(significance):
    X, T0, T4, T5, T10 = imp.import_tep_sets()
    num_vars, train_data_points = X.shape

    """ Train model """
    # Create IncSFA object
    SlowFeature = IncSFA(33, 99, 99, 2, 1, 2, 0)
    SlowFeature.delta = 3
    SlowFeature.Md = 55
    SlowFeature.Me = 99 - 55

    # Train model
    for i in range(X.shape[1]):
        _ = SlowFeature.add_data(X[:, i], use_svd_whitening=True)
    SlowFeature.converged = True

    # """ Test model """
    test_data = [("IncSFA_IDV(0)", T0), ("IncSFA_IDV(4)", T4),
                 ("IncSFA_IDV(5)", T5), ("IncSFA_IDV(10)", T10)]
    for name, test in test_data:
        test_obj = copy.deepcopy(SlowFeature)
        data_points = test.shape[1]
        stats = np.zeros((4, data_points))

        for i in range(data_points):
            run = test_obj.add_data(test[:, i], alpha=0.05)
            stats[:, i] = run[1]

        crit_vals = []
        for alpha in significance:
            p = 1 - alpha
            n = train_data_points
            Md = 55
            Me = 99 - 55
            gd = (Md*(n**2-2*n))/((n-1)*(n-Md-1))
            ge = (Me*(n**2-2*n))/((n-1)*(n-Me-1))

            T_d_crit = ST.chi2.ppf(p, Md)
            T_e_crit = ST.chi2.ppf(p, Me)
            S_d_crit = gd*ST.f.ppf(p, Md, n-Md-1)
            S_e_crit = ge*ST.f.ppf(p, Me, n-Me-1)

            crit_vals.append([T_d_crit, T_e_crit, S_d_crit, S_e_crit])

        _fig = plt.figure(name)
        plt.subplots_adjust(wspace=0.4)
        figure_text = ("Lagged Copies= 2 | Expansion Order= 1 | $M_d$= 55 | "
                       "Epochs= 1  | K= 99 | J= 99 | L= 2 ")
        plt.figtext(0.05, 0.02, figure_text)
        titles = ["$T^2$", "$T_e^2$", "$S^2$", "$S_e^2$"]
        for i in range(4):
            plt.subplot(4, 1, i+1)
            plt.ylabel(titles[i])
            plt.xlabel("Sample")
            plt.plot(stats[i, 3:])
            for j in range(len(significance)):
                c_vals = np.ones((data_points)) * crit_vals[j][i]
                line_label = f"alpha = {significance[j]}"
                plt.plot(c_vals, linestyle='--', label=line_label)
            if i == 0:
                plt.legend(loc='upper left')
        _fig.set_size_inches(21, 9)
        plt.savefig(name, dpi=350)
        plt.close(fig=_fig)


if __name__ == "__main__":
    plot_incsfa([1.49e-08, 5.55e-17, 1.91e-6, 5.68e-14])

    """
    Results:
    --------
    SFA (0.99):
        Td = 4.77e-07
        Te = 3.05e-05
        Sd = 1.53e-05
        Se = 0.000977
    SFA (0.95):
        Td = 3.05e-05
        Te = 0.000977
        Sd = 0.000977
        Se = 0.007813
    IncSFA (0.99):
        Td = 1.82e-12
        Te = 1.49e-08
        Sd = 5.55e-17
        Se = 2.33e-10
    IncSFA (0.95):
        Td = 2.98e-08
        Te = 1.91e-06
        Sd = 5.68e-14
        Se = 5.96e-08
    """
