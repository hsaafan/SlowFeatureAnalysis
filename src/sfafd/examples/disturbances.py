""" Plots the disturbances for some common test sets of the Tennessee Eastman
Process
"""
import matplotlib.pyplot as plt
import numpy as np
import tepimport as imp


def plot_disturbances(show: bool = True, save: bool = False,
                      w_in: float = 8, h_in: float = 6) -> None:
    training_sets = imp.import_sets([4, 5, 10], skip_training=True)
    _, T4 = training_sets[0]
    _, T5 = training_sets[1]
    _, T10 = training_sets[2]

    ignored_var = list(range(22, 41))
    T4 = np.delete(T4, ignored_var, axis=0)
    T5 = np.delete(T5, ignored_var, axis=0)
    T10 = np.delete(T10, ignored_var, axis=0)

    """ IDV(4) """
    T4_reactor_temp = T4[8, :]
    reactor_temp_setpoint = 120.4 * np.ones_like(T4_reactor_temp)
    T4_rector_cooling_water_flow = T4[31, :]

    plt.rcParams.update({'font.size': 16})
    fig4, ax4 = plt.subplots(nrows=2, sharex=True)
    ax4[0].set_title("Reactor Cooling Water Flow")
    ax4[0].set_ylabel("Flow $(m^3h^{-1})$")
    ax4[0].plot(T4_rector_cooling_water_flow)

    ax4[1].set_title("Reactor Temperature")
    ax4[1].set_ylabel("Temperature ($\degree C$)")
    ax4[1].set_xlabel("Sample Index")
    ax4[1].plot(T4_reactor_temp, label="Reactor Temperature")
    ax4[1].plot(reactor_temp_setpoint, label="Setpoint")
    ax4[1].legend()

    fig4.set_size_inches(w_in, h_in)
    fig4.tight_layout()
    """ IDV(5) """
    T5_condenser_cooling_water_flow = T5[32, :]
    T5_reactor_pressure = T5[6, :]
    reactor_pressure_setpoint = 2705*np.ones_like(T5_reactor_pressure)

    fig5, ax5 = plt.subplots(nrows=2, sharex=True)
    ax5[0].set_title("Condenser Cooling Water Flow")
    ax5[0].set_ylabel("Flow $(m^3h^{-1})$")
    ax5[0].plot(T5_condenser_cooling_water_flow)

    ax5[1].set_title("Reactor Pressure")
    ax5[1].set_ylabel("Pressure (kPag)")
    ax5[1].set_xlabel("Sample Index")
    ax5[1].plot(T5_reactor_pressure, label="Reactor Pressure")
    ax5[1].plot(reactor_pressure_setpoint, label="Setpoint")
    ax5[1].legend()

    fig5.set_size_inches(w_in, h_in)
    fig5.tight_layout()
    """ IDV(10) """
    T10_c_flow = T10[3, :]
    T10_stripper_temp = T10[17, :]
    stripper_temp_setpoint = 65.731 * np.ones_like(T10_stripper_temp)

    fig10, ax10 = plt.subplots(nrows=2, sharex=True)
    ax10[0].set_title("A and C Feed Flow")
    ax10[0].set_ylabel("Flow (KSCMH)")
    ax10[0].plot(T10_c_flow)

    ax10[1].set_title("Stripper Temperature")
    ax10[1].set_ylabel("Temperature ($\degree C$)")
    ax10[1].set_xlabel("Sample Index")
    ax10[1].plot(T10_stripper_temp, label="Stripper Temperature")
    ax10[1].plot(stripper_temp_setpoint, label="Setpoint")
    ax10[1].legend()

    fig10.set_size_inches(w_in, h_in)
    fig10.tight_layout()
    if show:
        plt.show()
    if save:
        fig4.savefig("IDV(4)")
        fig5.savefig("IDV(5)")
        fig10.savefig("IDV(10)")


if __name__ == "__main__":
    plot_disturbances(show=True, save=True)
