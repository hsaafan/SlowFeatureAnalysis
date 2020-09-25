""" Plots the disturbances for each test set """

import matplotlib.pyplot as plt
import numpy as np
import tep_import as imp

if __name__ == "__main__":
    X, T0, T4, T5, T10 = imp.import_tep_sets()

    plt.rcParams.update({'font.size': 16})
    plt.subplots_adjust(0.17, 0.05, 0.95, 0.95, 0, 0.05)
    """ IDV(4) """
    T4_reactor_temp = T4[8, :]
    reactor_temp_setpoint = 120.4 * np.ones_like(T4_reactor_temp)
    T4_rector_cooling_water_flow = T4[31, :]

    plt.figure("IDV(4)")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.title("Reactor Cooling Water Flow")
    plt.ylabel("Flow $(m^3h^{-1})$")
    plt.plot(T4_rector_cooling_water_flow)

    plt.subplot(2, 1, 2)
    plt.title("Reactor Temperature")
    plt.ylabel("Temperature ($\degree C$)")
    plt.xlabel("Sample Index")
    plt.plot(T4_reactor_temp, label="Reactor Temperature")
    plt.plot(reactor_temp_setpoint, label="Setpoint")
    plt.legend()

    """ IDV(5) """
    T5_condenser_cooling_water_flow = T5[32, :]
    T5_reactor_pressure = T5[6, :]
    reactor_pressure_setpoint = 2705*np.ones_like(T5_reactor_pressure)

    plt.figure("IDV(5)")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.title("Condenser Cooling Water Flow")
    plt.ylabel("Flow $(m^3h^{-1})$")
    plt.plot(T5_condenser_cooling_water_flow)

    plt.subplot(2, 1, 2)
    plt.title("Reactor Pressure")
    plt.ylabel("Pressure (kPag)")
    plt.xlabel("Sample Index")
    plt.plot(T5_reactor_pressure, label="Reactor Pressure")
    plt.plot(reactor_pressure_setpoint, label="Setpoint")
    plt.legend()

    """ IDV(10) """
    T10_c_flow = T10[3, :]
    T10_stripper_temp = T10[17, :]
    stripper_temp_setpoint = 65.731*np.ones_like(T10_stripper_temp)

    plt.figure("IDV(10)")
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(2, 1, 1)
    plt.title("A and C Feed Flow")
    plt.ylabel("Flow (KSCMH)")
    plt.plot(T10_c_flow)

    plt.subplot(2, 1, 2)
    plt.title("Stripper Temperature")
    plt.ylabel("Temperature ($\degree C$)")
    plt.xlabel("Sample Index")
    plt.plot(T10_stripper_temp, label="Stripper Temperature")
    plt.plot(stripper_temp_setpoint, label="Setpoint")
    plt.legend()
    plt.show()
