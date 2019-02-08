import numpy as np
import matplotlib.pyplot as plt
from dolfin import plot

import heat_generation as heat
from current_profile import current
import open_circuit_potentials as ocp
from utilities import get_vars_time
from voltage import Voltage


def plot_voltage(soln, mesh, R_cc, param):
    # Create voltage object
    voltage = Voltage(soln, mesh, R_cc, param)

    # Convert to dimensional time
    t = soln.t * param.tau_d_star

    # LIONSIMBA results
    if param.h_star > 0.5:
        t_LION = np.linspace(0, 2700, 10)
        V_LION = np.array([4.0030, 3.9396, 3.8904, 3.8474, 3.8100,
                           3.7770, 3.7460, 3.7125, 3.6605, 3.5408])
    elif param.h_star > 0.5*1E-2:
        t_LION = np.linspace(0, 2400, 9)
        V_LION = np.array([4.0030, 3.9398, 3.8908, 3.8478, 3.8105,
                           3.7773, 3.7460, 3.7114, 3.6559])
    else:
        t_LION = np.linspace(0, 2400, 9)
        V_LION = np.array([4.0030, 3.9398, 3.8908, 3.8479, 3.8105,
                           3.7773, 3.7459, 3.7113, 3.6555])

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Make plots
    fig = plt.figure()
    plt.plot(t, voltage.v_term, label="SPMe")
    plt.plot(t_LION, V_LION, 'o', label='LIONSIMBA')
    plt.xlim([t[0], t[-1]])
    plt.ylim([param.V_min, param.V_max])
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Voltage [V]', fontsize=11)
    plt.title('Voltage', fontsize=11)
    plt.legend()
    fig.tight_layout()


def plot_temperature(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)

    # LIONSIMBA results
    if param.h_star > 0.5:
        t_LION = np.linspace(0, 2700, 10)
        T_LION = np.array([298.1500, 299.1755, 299.7207, 300.0215, 300.1525,
                           300.2436, 300.4308, 300.7147, 301.1607, 301.7106])
    elif param.h_star > 0.5*1E-2:
        t_LION = np.linspace(0, 2400, 9)
        T_LION = np.array([298.1500, 300.0446, 302.6419, 305.5707, 308.5822,
                           311.6190, 314.8875, 318.5352, 322.8777])
    else:
        t_LION = np.linspace(0, 2400, 9)
        T_LION = np.array([298.1500, 300.0582, 302.7058, 305.7287, 308.8814,
                           312.1069, 315.6139, 319.5548, 324.2574])
    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot temperature
    fig = plt.figure()
    plt.plot(soln.t * param.tau_d_star,
             T0 * param.Delta_T_star + param.T_inf_star,
             label='SPMe')
    plt.plot(t_LION, T_LION, 'o', label='LIONSIMBA')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$T$', fontsize=11)
    plt.legend()
    fig.tight_layout()


def plot_surface_concentration(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)

    # Surface concentration for BV
    c_n_surf = c_n[-1, :] + (c_n[-1, :] - c_n[-2, :]) / 2
    c_p_surf = c_p[-1, :] + (c_p[-1, :] - c_p[-2, :]) / 2

    # LIONSIMBA results
    if param.h_star > 0.5:
        t_LION = np.linspace(0, 2700, 10)
        c_n_LION = [0.7333, 0.6597, 0.5878, 0.5159, 0.4440,
                    0.3721, 0.3002, 0.2283, 0.1564, 0.0845]
        c_p_LION = [0.5674, 0.6091, 0.6474, 0.6858, 0.7241,
                    0.7624, 0.8008, 0.8391, 0.8774, 0.9157]
    else:
        t_LION = [0]
        c_n_LION = [0]
        c_p_LION = [0]

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot surface concentrations
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(soln.t * param.tau_d_star, c_n_surf, label='SPMe')
    plt.plot(t_LION, c_n_LION, 'o', label='LIONSIMBA')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{n}}}}$', fontsize=11)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(soln.t * param.tau_d_star, c_p_surf, label='SPMe')
    plt.plot(t_LION, c_p_LION, 'o', label='LIONSIMBA')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{p}}}}$', fontsize=11)
    plt.legend()
    fig.tight_layout()


def plot_electrolyte_concentration(soln, mesh, param, time):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)

    # LIONSIMBA results
    if time == 1800:
        c_e_LION = [167.2502, 163.5963, 156.2971, 145.3744, 130.8719, 112.8752,
                    91.5430, 67.1453, 40.1014, 10.9754, -5.1804, -6.6067,
                    -8.0319, -9.4560, -10.8789, -12.3008, -13.7216, -15.1413,
                    -16.5598, -17.9773, -32.3561, -56.7032, -77.9743, -96.3081,
                    -111.8209, -124.6087, -134.7493, -142.3031, -147.3146,
                    -149.8131]
        x_LION = [0.0228, 0.0684, 0.1140, 0.1596, 0.2052, 0.2508, 0.2964,
                  0.3420, 0.3876, 0.4332, 0.4624, 0.4754, 0.4883, 0.5013,
                  0.5142, 0.5272, 0.5402, 0.5531, 0.5661, 0.5790, 0.6062,
                  0.6477, 0.6891, 0.7306, 0.7720, 0.8135, 0.8549, 0.8964,
                  0.9378, 0.9793]
    else:
        raise ValueError('No LIONSIMBA data for this time!')
    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot electrolyte concentration at time

    # Find index closest to time in seconds
    idx = (np.abs(soln.t * param.tau_d_star - time)).argmin()

    fig = plt.figure()
    plt.plot((mesh.x_n[1:] + mesh.x_n[0:-1])/2,
             param.c_e_typ_star * param.delta * c_e_n[:, idx], '-',
             c='#1f77b4')
    plt.plot((mesh.x_s[1:] + mesh.x_s[0:-1])/2,
             param.c_e_typ_star * param.delta * c_e_s[:, idx], '-',
             c='#1f77b4')
    plt.plot((mesh.x_p[1:] + mesh.x_p[0:-1])/2,
             param.c_e_typ_star * param.delta * c_e_p[:, idx], '-',
             c='#1f77b4',
             label="SPMe")
    plt.plot(x_LION, c_e_LION, 'o', c='#ff7f0e', label="LIONSIMBA")
    plt.xlim([0, 1])
    plt.xlabel(r'$x$', fontsize=11)
    plt.ylabel(r'$c_{{\mathrm{{e}}}}^* - c_{{\mathrm{{e,typ}}}}^*$'
               r'[mol/m$^3$]', fontsize=11)
    plt.legend()
    fig.tight_layout()


def plot_heat_generation(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)
    t = soln.t

    # Surface concentration for BV
    c_n_surf = c_n[-1, :] + (c_n[-1, :] - c_n[-2, :]) / 2
    c_p_surf = c_p[-1, :] + (c_p[-1, :] - c_p[-2, :]) / 2

    # Electrode avergaed electrolyte concentrations and the values at the
    # electrode/separator interfaces needed for heat source terms
    c_e_n_bar = np.trapz(c_e_n,
                         dx=mesh.dx_n, axis=0) / param.L_n
    c_e_p_bar = np.trapz(c_e_p,
                         dx=mesh.dx_p, axis=0) / param.L_p
    c_e_neg_sep = (c_e_n[-1, :] + c_e_s[0, :]) / 2
    c_e_pos_sep = (c_e_s[-1, :] + c_e_p[0, :]) / 2

    # Evaluate I_app
    I_app = current(t, param)

    # LIONSIMBA results
    t_LION = np.linspace(0, 2700, 10)
    Ohm_n = [1289.0, 6493.3, 6871.1, 6837.5, 6770.9,
             7044.4, 8172.6, 9348.8, 8133.8, 7253.3]
    Ohm_s = [2650, 12063, 11911, 11821, 11777,
             11742, 11685, 11634, 11526, 11345]
    Ohm_p = [1253.5, 6694.7, 6643.9, 6582.1, 6543.4,
             6517.6, 6500.1, 6526.3, 6607.7, 6711.6]
    Rxn_n = [2125.8, 1802.5, 1705.8, 1681.6, 1698.2,
             1736.5, 1828.5, 2067.9, 2293.2, 2963.6]
    Rxn_p = [2295.5, 2397.2, 2443.5, 2510.7, 2605.1,
             2734.9, 2914.7, 3167.5, 3536.7, 4163.5]
    Rev_n = [-10164, -10199, -10217, -11333, -15162,
             -17341, -17295, -15992, -7123, 5294]
    Rev_p = [12511, 26476, 34395, 40373, 45665,
             50759, 55845, 60893, 65673, 69641]

    scale = param.I_star * param.Phi_star / param.Lx_star

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Make plots
    fig = plt.figure(figsize=(15, 9))

    plt.subplot(1, 3, 1)
    plt.plot(t * param.tau_d_star,
             param.delta
             * heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app),
             label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_0(T0, c_n_surf, param, I_app)
             + param.delta
             * heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_n_0(T0, c_n_surf, param, I_app)
             + param.delta
             * heat.rev_n_1(T1, c_n_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_0(T0, c_n_surf, param, I_app)
             + heat.rev_n_0(T0, c_n_surf, param, I_app)
             + param.delta
             * heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app)
             + param.delta
             * heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app)
             + param.delta
             * heat.rev_n_1(T1, c_n_surf, param, I_app),
             label="Total")
    plt.plot(t_LION, np.array(Ohm_n) / scale, 'o')
    plt.plot(t_LION, np.array(Rxn_n) / scale, 'x')
    plt.plot(t_LION, np.array(Rev_n) / scale, '^')
    plt.plot(t_LION, np.array(Ohm_n) / scale
             + np.array(Rxn_n) / scale
             + np.array(Rev_n) / scale, 's')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Negative electrode', fontsize=11)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(t * param.tau_d_star,
             param.delta
             * heat.ohmic_p_1(c_e_neg_sep, c_e_pos_sep, param, I_app),
             label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_0(T0, c_p_surf, param, I_app)
             + param.delta
             * heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_p_0(T0, c_p_surf, param, I_app)
             + param.delta
             * heat.rev_p_1(T1, c_p_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_0(T0, c_p_surf, param, I_app)
             + heat.rev_p_0(T0, c_p_surf, param, I_app)
             + param.delta
             * heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app)
             + param.delta
             * heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app)
             + param.delta
             * heat.rev_p_1(T1, c_p_surf, param, I_app),
             label="Total")
    plt.plot(t_LION, np.array(Ohm_p) / scale, 'o')
    plt.plot(t_LION, np.array(Rxn_p) / scale, 'x')
    plt.plot(t_LION, np.array(Rev_p) / scale, '^')
    plt.plot(t_LION, np.array(Ohm_p) / scale
             + np.array(Rxn_n) / scale
             + np.array(Rev_n) / scale, 's')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Positive electrode', fontsize=11)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(t * param.tau_d_star,
             param.delta
             * heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app),
             label="Ohm")
    plt.plot(t_LION, np.array(Ohm_s) / scale, 'o')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Separator', fontsize=11)
    plt.legend()

    fig.tight_layout()
