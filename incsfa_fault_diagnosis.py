import copy
import numpy as np

import fault_diagnosis as fd
import tep_import as imp
import data_plotter
from incsfa import IncSFA


def print_tallies(tallies):
    output = ""
    for heading, stat_list in tallies:
        output += f"{heading}\n"
        for stat, tally_list in stat_list:
            output += f"    {stat}\n"
            for variable, count in tally_list.items():
                output += f"        Variable {variable}: {count}\n"
    print(output)


if __name__ == "__main__":
    crit_stats = [175, 145, 300, 200]
    """ Import Data """
    X, T0, T4, T5, T10 = imp.import_tep_sets()
    alpha = 1.91e-6  # From find_control_limit results
    num_vars, train_data_points = X.shape

    """ Train model """
    # Create IncSFA object
    SlowFeature = IncSFA(33, 99, 99, 2, 1, 2, 0)
    SlowFeature.delta = 3
    SlowFeature.Md = 55
    SlowFeature.Me = 99 - 55

    for i in range(train_data_points):
        _ = SlowFeature.add_data(X[:, i], use_svd_whitening=True)
    SlowFeature.converged = True

    """ Fault Diagnosis Setup """
    Q = SlowFeature.standardization_node.whitening_matrix
    P = SlowFeature.transformation_matrix
    W = (Q @ P).T

    Omega = SlowFeature.features_speed

    order = Omega.argsort()
    Omega = Omega[order]
    W = W[order, :]

    W_d = W[:SlowFeature.Md, :]
    W_e = W[SlowFeature.Md:, :]

    Omega_d = Omega[:SlowFeature.Md]
    Omega_e = Omega[SlowFeature.Md:]

    M_t2_d = W_d.T @ W_d
    M_t2_e = W_e.T @ W_e

    M_s2_d = W_d.T @ np.diag(Omega_d ** -1) @ W_d
    M_s2_e = W_e.T @ np.diag(Omega_e ** -1) @ W_e

    """ Test model """
    test_data = [("IncSFA_IDV(0)", T0), ("IncSFA_IDV(4)", T4),
                 ("IncSFA_IDV(5)", T5), ("IncSFA_IDV(10)", T10)]
    test_faults = []
    for name, test in test_data:
        test_obj = copy.deepcopy(SlowFeature)
        data_points = test.shape[1]
        stats = np.zeros((4, data_points))
        T_d_faults = []
        T_e_faults = []

        S_d_faults = []
        S_e_faults = []

        for i in range(data_points):
            run = test_obj.add_data(test[:, i], alpha=alpha)
            stats = run[1]
            # crit_stats = run[2]

            if stats[0] > crit_stats[0]:
                T_d_faults.append(i)

            if stats[1] > crit_stats[1]:
                T_e_faults.append(i)

            if stats[2] > crit_stats[2]:
                S_d_faults.append(i)

            if stats[3] > crit_stats[3]:
                S_e_faults.append(i)

        fault_list = [("T_d", T_d_faults), ("T_e", T_e_faults),
                      ("S_d", S_d_faults), ("S_e", S_e_faults)]
        test_faults.append((name, test, fault_list))

    """ Fault Diagnosis """
    def get_data_point(i):
        x = np.concatenate((test[:, i], test[:, i - 1], test[:, i - 2]))
        x = x.reshape((-1, 1))
        return(SlowFeature.standardization_node.center_similar(x))

    def get_data_point_derivative(i):
        return((get_data_point(i) - get_data_point(i - 1)) / SlowFeature.delta)

    all_test_tallies = []
    for name, test, all_faults in test_faults:
        tallies = []
        for test_stat_name, fault_list in all_faults:
            if test_stat_name == "T_d":
                D = M_t2_d
                data_point = get_data_point
            elif test_stat_name == "T_e":
                D = M_t2_e
                data_point = get_data_point
            elif test_stat_name == "S_d":
                D = M_s2_d
                data_point = get_data_point_derivative
            elif test_stat_name == "S_e":
                D = M_s2_e
                data_point = get_data_point_derivative
            else:
                raise RuntimeError

            fault_tally = dict()
            for i in fault_list:
                # Get index and limit for each fault sample
                cont = fd.contribution_index(D, data_point(i), 'RBC')
                highest_contrib = int(np.argmax(cont['RBC']))
                if highest_contrib in fault_tally:
                    fault_tally[highest_contrib] += 1
                else:
                    fault_tally[highest_contrib] = 1
            tallies.append((test_stat_name, fault_tally))
        all_test_tallies.append((name, tallies))

    print_tallies(all_test_tallies)
    """ Contribution Plots """
    sample = 500  # Sample to plot
    plotter = data_plotter.SFAPlotter(show=False, save=True)
    plot = plotter.plot_contributions
    for name, test in test_data:
        plot(
            f"{name}_RBC_T_d",
            f"$T_d$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_t2_d,
                                           get_data_point(sample),
                                           'RBC'
                                           )['RBC']
                    ).reshape((-1, ))
        )
        plot(
            f"{name}_RBC_T_e",
            f"$T_e$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_t2_e,
                                           get_data_point(sample),
                                           'RBC'
                                           )['RBC']
                    ).reshape((-1, ))
        )
        plot(
            f"{name}_RBC_S_d",
            f"$S_d$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_s2_d,
                                           get_data_point_derivative(sample),
                                           'RBC'
                                          )['RBC']
                    ).reshape((-1, ))
        )
        plot(
            f"{name}_RBC_S_e",
            f"$S_e$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_s2_e,
                                           get_data_point_derivative(sample),
                                           'RBC'
                                          )['RBC']
                    ).reshape((-1, ))
        )

    """
    Using RBC Fault Diagnosis with manually set control limits
        T_d limit: 175
        T_e limit: 145
        S_d limit: 300
        S_e limit: 200

    Variables 0  - 21: XMEAS(1) - (22)
    Variables 22 - 32: XMV(1) - XMV(11)
    Variables 33 - 65: Same as above but for previous sample
    Variables 66 - 98: Same as above but for second previous sample

    IncSFA_IDV(0)
        T_d
            Variable 16: 1
            Variable 98: 1
            Variable 3: 1
            Variable 4: 1
            Variable 18: 1
            Variable 31: 1
            Variable 26: 1
        T_e
            Variable 34: 1
            Variable 82: 1
        S_d
            Variable 16: 1
            Variable 50: 1
            Variable 98: 1
            Variable 3: 1
            Variable 48: 1
        S_e
            Variable 34: 1
            Variable 84: 1
            Variable 82: 1
            Variable 89: 1
    IncSFA_IDV(4)
        T_d
            Variable 17: 1
            Variable 96: 2
            Variable 31: 32             XMV(10) - Reactor Cooling Water Flow
            Variable 64: 765            XMV(10) - Reactor Cooling Water Flow
            Variable 97: 3
        T_e
            Variable 17: 1
            Variable 96: 1
            Variable 31: 4
            Variable 64: 563            XMV(10) - Reactor Cooling Water Flow
            Variable 97: 233            XMV(10) - Reactor Cooling Water Flow
        S_d
            Variable 17: 1
            Variable 51: 1
            Variable 96: 1
            Variable 63: 1
            Variable 8: 1
            Variable 64: 1
            Variable 74: 2
            Variable 26: 1
            Variable 98: 1
        S_e
            Variable 17: 1
            Variable 63: 1
            Variable 96: 1
            Variable 31: 1
            Variable 84: 1
            Variable 86: 1
    IncSFA_IDV(5)
        T_d
            Variable 16: 1
            Variable 98: 1
            Variable 96: 1
            Variable 32: 1
            Variable 82: 35
            Variable 27: 679            XMV(6) - Purge Valve
            Variable 9: 14
            Variable 52: 1
            Variable 0: 70              XMEAS(1) - A Feed
        T_e
            Variable 16: 4
            Variable 98: 1
            Variable 32: 238            XMV(11) - Condenser Cooling Water Flow
            Variable 82: 521            XMEAS(17) - Stripper Underflow
            Variable 74: 11
            Variable 17: 27
        S_d
            Variable 16: 1
            Variable 65: 1
            Variable 98: 2
            Variable 63: 11
            Variable 32: 1
            Variable 82: 1
            Variable 87: 1
            Variable 51: 2
            Variable 48: 1
            Variable 97: 1
            Variable 26: 1
        S_e
            Variable 32: 1
            Variable 65: 1
            Variable 82: 2
            Variable 16: 1
            Variable 63: 1
    IncSFA_IDV(10)
        T_d
            Variable 18: 2
            Variable 96: 75             XMV(9) - Stripper Steam Valve
            Variable 59: 1
            Variable 17: 617            XMEAS(18) - Stripper Temperature
            Variable 84: 17
            Variable 89: 1
            Variable 83: 7
            Variable 63: 2
            Variable 6: 1
            Variable 12: 1
            Variable 71: 1
            Variable 45: 1
            Variable 26: 1
            Variable 92: 1
            Variable 19: 1
            Variable 9: 1
            Variable 36: 2
            Variable 20: 1
            Variable 53: 1
            Variable 52: 2
            Variable 35: 1
            Variable 46: 1
            Variable 85: 1
        T_e
            Variable 30: 6
            Variable 96: 5
            Variable 17: 599            XMEAS(18) - Stripper Temperature
            Variable 2: 2
            Variable 83: 10
            Variable 84: 13
            Variable 87: 7
            Variable 1: 10
            Variable 52: 6
            Variable 53: 2
            Variable 12: 12
            Variable 92: 3
            Variable 70: 2
            Variable 20: 2
            Variable 33: 3
            Variable 98: 1
            Variable 59: 1
            Variable 81: 1
            Variable 10: 1
            Variable 85: 3
            Variable 8: 1
            Variable 82: 1
            Variable 21: 1
            Variable 5: 1
            Variable 77: 1
            Variable 72: 1
        S_d
            Variable 30: 2
            Variable 63: 8
            Variable 96: 7
            Variable 84: 5
            Variable 17: 37             XMEAS(18) - Stripper Temperature
            Variable 50: 1
            Variable 86: 1
            Variable 27: 1
            Variable 48: 1
            Variable 52: 1
            Variable 51: 1
        S_e
            Variable 30: 1
            Variable 63: 1
            Variable 96: 1
            Variable 17: 6
            Variable 1: 1
    """
