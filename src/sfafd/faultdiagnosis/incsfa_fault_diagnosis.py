import copy
import numpy as np
import matplotlib.pyplot as plt
from math import floor

import fault_diagnosis as fd
import tepimport as imp
from incsfa import IncSFA

INDEX_LABELS = [
    ("XMEAS(1)", "A Feed  (stream 1)", "kscmh"),
    ("XMEAS(2)", "D Feed  (stream 2)", "kg/hr"),
    ("XMEAS(3)", "E Feed  (stream 3)", "kg/hr"),
    ("XMEAS(4)", "A and C Feed  (stream 4)", "kscmh"),
    ("XMEAS(5)", "Recycle Flow  (stream 8)", "kscmh"),
    ("XMEAS(6)", "Reactor Feed Rate  (stream 6)", "kscmh"),
    ("XMEAS(7)", "Reactor Pressure", "kPa gauge"),
    ("XMEAS(8)", "Reactor Level", "%"),
    ("XMEAS(9)", "Reactor Temperature", "Deg C"),
    ("XMEAS(10)", "Purge Rate (stream 9)", "kscmh"),
    ("XMEAS(11)", "Product Sep Temp", "Deg C"),
    ("XMEAS(12)", "Product Sep Level", "%"),
    ("XMEAS(13)", "Prod Sep Pressure", "kPa gauge"),
    ("XMEAS(14)", "Prod Sep Underflow (stream 10)", "m3/hr"),
    ("XMEAS(15)", "Stripper Level", "%"),
    ("XMEAS(16)", "Stripper Pressure", "kPa gauge"),
    ("XMEAS(17)", "Stripper Underflow (stream 11)", "m3/hr"),
    ("XMEAS(18)", "Stripper Temperature", "Deg C"),
    ("XMEAS(19)", "Stripper Steam Flow", "kg/hr"),
    ("XMEAS(20)", "Compressor Work", "kW"),
    ("XMEAS(21)", "Reactor Cooling Water Outlet Temp", "Deg C"),
    ("XMEAS(22)", "Separator Cooling Water Outlet Temp", "Deg C"),
    ("XMV(1)", "D Feed Flow (stream 2)", ""),
    ("XMV(2)", "E Feed Flow (stream 3)", ""),
    ("XMV(3)", "A Feed Flow (stream 1)", ""),
    ("XMV(4)", "A and C Feed Flow (stream 4)", ""),
    ("XMV(5)", "Compressor Recycle Valve", ""),
    ("XMV(6)", "Purge Valve (stream 9)", ""),
    ("XMV(7)", "Separator Pot Liquid Flow (stream 10)", ""),
    ("XMV(8)", "Stripper Liquid Product Flow (stream 11)", ""),
    ("XMV(9)", "Stripper Steam Valve", ""),
    ("XMV(10)", "Reactor Cooling Water Flow", ""),
    ("XMV(11)", "Condenser Cooling Water Flow", ""),
]


def plot_test_stats(fig_name: str, title: str,
                    T_d_values: np.ndarray, T_e_values: np.ndarray,
                    S_d_values: np.ndarray, S_e_values: np.ndarray,
                    T_d_crit: float, T_e_crit: float,
                    S_d_crit: float, S_e_crit: float,
                    show: bool = False, save: bool = True) -> None:
    """ Plot the S2 and T2 test statistics

    Parameters
    ----------
    fig_name: str
        The name of the figure used for the file name if saving
    title: str
        The title on the figure
    T2_values: numpy.ndarray
        The T^2 values to plot
    S2_values:
        The S^2 values to plot
    show: bool
        Show the plot
    save: bool
        Save the plot to the current directory
    """
    _f, axs2d = plt.subplots(nrows=4, ncols=1, sharex=True)
    _f.set_size_inches(8, 6)

    T_d_plot = axs2d[0]
    T_d_plot.set_title(title)
    T_d_plot.set_ylabel("$T_d^2$")
    T_d_plot.plot(T_d_values)
    T_d_plot.plot([T_d_crit] * len(T_d_values))

    T_e_plot = axs2d[1]
    T_e_plot.set_ylabel("$T_e^2$")
    T_e_plot.plot(T_e_values)
    T_e_plot.plot([T_e_crit] * len(T_e_values))

    S_d_plot = axs2d[2]
    S_d_plot.set_ylabel("$S_d^2$")
    S_d_plot.plot(S_d_values)
    S_d_plot.plot([S_d_crit] * len(S_d_values))

    S_e_plot = axs2d[3]
    S_e_plot.set_ylabel("$S_e^2$")
    S_e_plot.plot(S_e_values)
    S_e_plot.plot([S_e_crit] * len(S_e_values))
    S_e_plot.set_xlabel("Variable Index")

    if show:
        plt.show()
    if save:
        plt.savefig(fig_name, dpi=350)
    plt.close(fig=_f)
    _f = None


def plot_contributions(fig_name: str, title: str,
                       contributions: np.ndarray, n_to_plot: int = 5,
                       show: bool = False, save: bool = True) -> None:
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
    show: bool
        Show the plot
    save: bool
        Save the plot to the current directory
    """
    _f, ax = plt.subplots()
    _f.set_size_inches(8, 6)

    order = np.argsort(-1 * contributions)
    ordered_cont = contributions[order]
    cum_percent = np.cumsum(ordered_cont) / np.sum(contributions)
    bar_labels = [str(x) for x in order]

    text_box = 'Variable Index Descriptions'
    for i in range(n_to_plot):

        label = order[i] % len(INDEX_LABELS)
        var_label, var_desc, var_units = INDEX_LABELS[label]
        text_box += f'\nIndex {order[i]} | {var_label}: {var_desc}'
        if var_units != '':
            text_box += f' - {var_units}'

        if order[i] >= len(INDEX_LABELS):
            lag = floor(order[i] / len(INDEX_LABELS))
            text_box += f' (t - {lag})'

    ax.bar(bar_labels[:n_to_plot], ordered_cont[:n_to_plot])
    ax2 = ax.twinx()
    ax2.plot(cum_percent[:n_to_plot], 'r')

    ax.set_title(title)
    bbox = dict(boxstyle="square", ec=(0.0, 0.0, 0.0), fc=(1., 1.0, 1.0),
                alpha=0.7)
    ax.text(x=0.1, y=0.75, s=text_box, bbox=bbox, transform=ax.transAxes)

    ax.set_xlabel("Variable Index")
    ax.set_ylabel("Fault Contribution")
    ax2.set_ylabel("Cumulative Contribution")
    ax2.set_ylim([0, 1])

    if show:
        plt.show()
    if save:
        plt.savefig(fig_name, dpi=350)
    plt.close(fig=_f)
    _f = None


def print_tallies(tallies):
    output = ""
    for heading, stat_list in tallies:
        output += f"{heading}\n"
        for stat, tally_list in stat_list:
            output += f"    {stat}: Faults = {sum(tally_list.values())}\n"
            sorted_tallies = dict(sorted(tally_list.items(),
                                         key=lambda item: item[1]))
            for variable, count in reversed(sorted_tallies.items()):
                label = variable % len(INDEX_LABELS)
                var_label, var_desc, var_units = INDEX_LABELS[label]
                variable_name = var_label + ' ' + var_desc

                if variable >= len(INDEX_LABELS):
                    lag = floor(variable / len(INDEX_LABELS))
                    variable_name += f' (t - {lag})'

                output += f"        {variable_name}: {count}\n"
    return(output)


def main(alpha: float = 0.05, sample: int = 500) -> None:
    if alpha > 1 or alpha < 0:
        raise ValueError("Confidence level should be between 0 and 1")
    print('Importing data...')
    crit_stats = [175, 145, 300, 200]
    """ Import Data """
    X, T0, T4, T5, T10 = imp.import_tep_sets(lagged_samples=0)
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
        T_d_faults = []
        T_e_faults = []
        S_d_faults = []
        S_e_faults = []

        T_d_values = []
        T_e_values = []
        S_d_values = []
        S_e_values = []

        for i in range(data_points):
            run = test_obj.add_data(test[:, i], alpha=alpha)
            if i < 5:
                continue
            stats = run[1]
            T_d_values.append(stats[0])
            T_e_values.append(stats[1])
            S_d_values.append(stats[2])
            S_e_values.append(stats[3])

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
        plot_name = f'{name}_Test_Stats'
        plot_title = f'{name[13:]} Test Statistics $alpha$={alpha}'
        plot_test_stats(plot_name, plot_title,
                        T_d_values, T_e_values,
                        S_d_values, S_e_values,
                        crit_stats[0], crit_stats[1],
                        crit_stats[2], crit_stats[3])

    """ Fault Diagnosis """
    def get_data_point(i, data):
        x = np.append(data[:, i], data[:, i - 1])
        x = np.append(x, data[:, i - 2])
        x = SlowFeature.standardization_node.center_similar(x.reshape((-1, 1)))
        return(x.reshape((-1, 1)))

    def get_data_point_derivative(i, data):
        return((get_data_point(i, data) - get_data_point(i - 1, data))
               / SlowFeature.delta)

    all_test_tallies = []
    for name, test, all_faults in test_faults:
        tallies = []
        for test_stat_name, fault_list in all_faults:
            if test_stat_name == "T_d":
                D = M_t2_d

                def data_point(i):
                    return(get_data_point(i, test))
            elif test_stat_name == "T_e":
                D = M_t2_e

                def data_point(i):
                    return(get_data_point(i, test))
            elif test_stat_name == "S_d":
                D = M_s2_d

                def data_point(i):
                    return(get_data_point_derivative(i, test))
            elif test_stat_name == "S_e":
                D = M_s2_e

                def data_point(i):
                    return(get_data_point_derivative(i, test))
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

    with open('Tallies.txt', 'w') as f:
        f.write(print_tallies(all_test_tallies))

    """ Contribution Plots """
    for name, test in test_data:
        plot_contributions(
            f"{name}_RBC_T_d",
            f"$T_d$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_t2_d,
                                           get_data_point(sample, test),
                                           'RBC'
                                           )['RBC']
                    ).reshape((-1, ))
        )
        plot_contributions(
            f"{name}_RBC_T_e",
            f"$T_e$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_t2_e,
                                           get_data_point(sample, test),
                                           'RBC'
                                           )['RBC']
                    ).reshape((-1, ))
        )
        plot_contributions(
            f"{name}_RBC_S_d",
            f"$S_d$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_s2_d,
                                           get_data_point_derivative(sample,
                                                                     test),
                                           'RBC'
                                          )['RBC']
                    ).reshape((-1, ))
        )
        plot_contributions(
            f"{name}_RBC_S_e",
            f"$S_e$ Fault Contributions Sample {sample} for {name[7:]}",
            np.array(
                     fd.contribution_index(
                                           M_s2_e,
                                           get_data_point_derivative(sample,
                                                                     test),
                                           'RBC'
                                          )['RBC']
                    ).reshape((-1, ))
        )


if __name__ == "__main__":
    main(alpha=0.05, sample=500)
