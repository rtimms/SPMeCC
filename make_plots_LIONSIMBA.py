import numpy as np
import matplotlib.pyplot as plt

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
    # t column 0, V column 1
    t_LION, V_LION = np.loadtxt('LIONSIMBA001_t.txt',
                                usecols=(0, 1),
                                unpack=True)

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
    # t columun 0, V column 2
    t_LION, T_LION = np.loadtxt('LIONSIMBA001_t.txt',
                                usecols=(0, 2),
                                unpack=True)

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot temperature
    fig = plt.figure()
    plt.plot(soln.t * param.tau_d_star,
             (T0 + param.delta * T1) * param.Delta_T_star + param.T_inf_star,
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
    # t column 0, c_n column 3, c_p column 4
    t_LION, c_n_LION, c_p_LION = np.loadtxt('LIONSIMBA001_t.txt',
                                            usecols=(0, 3, 4),
                                            unpack=True)

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot surface concentrations
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(soln.t * param.tau_d_star, c_n_surf * param.c_n_max_star,
             label='SPMe')
    plt.plot(t_LION, c_n_LION, 'o', label='LIONSIMBA')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{n}}}}$', fontsize=11)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(soln.t * param.tau_d_star, c_p_surf * param.c_p_max_star,
             label='SPMe')
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
    # t column 0, heat generation columns 5-13
    t_LION, Q_n, Q_p = np.loadtxt('LIONSIMBA001_t.txt',
                                  usecols=(0, 12, 13),
                                  unpack=True)
    Ohm_n, Rxn_n, Rev_n = np.loadtxt('LIONSIMBA001_t.txt',
                                     usecols=(5, 8, 10),
                                     unpack=True)
    Ohm_s = np.loadtxt('LIONSIMBA001_t.txt',
                       usecols=(6),
                       unpack=True)
    Ohm_p, Rxn_p, Rev_p = np.loadtxt('LIONSIMBA001_t.txt',
                                     usecols=(7, 9, 11),
                                     unpack=True)

    # Scale for heat generation in SPMe
    scale = param.I_star * param.Phi_star / param.Lx_star

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Make plots
    fig = plt.figure(figsize=(15, 9))

    plt.subplot(1, 3, 1)
    plt.plot(t * param.tau_d_star,
             param.delta * scale
             * heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app),
             label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_0(T0, c_n_surf, param, I_app) * scale
             + param.delta * scale
             * heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_n_0(T0, c_n_surf, param, I_app) * scale
             + param.delta * scale
             * heat.rev_n_1(T1, c_n_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_0(T0, c_n_surf, param, I_app) * scale
             + heat.rev_n_0(T0, c_n_surf, param, I_app) * scale
             + param.delta * scale
             * heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app)
             + param.delta * scale
             * heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app)
             + param.delta * scale
             * heat.rev_n_1(T1, c_n_surf, param, I_app),
             label="Total")
    plt.plot(t_LION, Ohm_n, 'o', c='#1f77b4')
    plt.plot(t_LION, Rxn_n, 'x', c='#ff7f0e')
    plt.plot(t_LION, Rev_n, '^', c='#2ca02c')
    plt.plot(t_LION, Q_n, 's', c='#d62728')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Negative electrode', fontsize=11)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(t * param.tau_d_star,
             param.delta * scale
             * heat.ohmic_p_1(c_e_neg_sep, c_e_pos_sep, param, I_app))
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_0(T0, c_p_surf, param, I_app) * scale
             + param.delta * scale
             * heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app))
    plt.plot(t * param.tau_d_star,
             heat.rev_p_0(T0, c_p_surf, param, I_app) * scale
             + param.delta * scale
             * heat.rev_p_1(T1, c_p_surf, param, I_app))
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_0(T0, c_p_surf, param, I_app) * scale
             + heat.rev_p_0(T0, c_p_surf, param, I_app) * scale
             + param.delta * scale
             * heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app)
             + param.delta * scale
             * heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app)
             + param.delta * scale
             * heat.rev_p_1(T1, c_p_surf, param, I_app))
    plt.plot(t_LION, Ohm_p, 'o', c='#1f77b4', label="Ohm")
    plt.plot(t_LION, Rxn_p, 'x', c='#ff7f0e', label="rxn")
    plt.plot(t_LION, Rev_p, '^', c='#2ca02c', label="rev")
    plt.plot(t_LION, Q_p, 's', c='#d62728', label="Total")
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Positive electrode', fontsize=11)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(t * param.tau_d_star,
             param.delta * scale
             * heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app))
    plt.plot(t_LION, Ohm_s, 'o', c='#1f77b4')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Separator', fontsize=11)

    fig.tight_layout()


def plot_OCP(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)
    t = soln.t

    # Surface concentration for BV
    c_n_surf = c_n[-1, :] + (c_n[-1, :] - c_n[-2, :]) / 2
    c_p_surf = c_p[-1, :] + (c_p[-1, :] - c_p[-2, :]) / 2

    # LIONSIMBA results
    # t column 0, OCP 14 - 17
    t_LION, U_n, U_p, dUdT_n, dUdT_p = np.loadtxt('LIONSIMBA001_t.txt',
                                                  usecols=(0, 14, 15, 16, 17),
                                                  unpack=True)
    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot OCP and entropic coefficient at a fixed T
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t * param.tau_d_star,
             ocp.U_n(c_n_surf, T0, param) * param.Phi_star)
    plt.plot(t_LION, U_n, 'o')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$U_{{\mathrm{{n}}}}$', fontsize=11)
    plt.subplot(2, 2, 3)
    plt.plot(t * param.tau_d_star,
             ocp.dUdT_n(c_n_surf, param) * param.Phi_star / param.Delta_T_star)
    plt.plot(t_LION, dUdT_n, 'o')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\mathrm{{d}}U_{{\mathrm{{n}}}}$'
               r'$ / \mathrm{{d}}T$', fontsize=11)
    plt.subplot(2, 2, 2)
    plt.plot(t * param.tau_d_star,
             ocp.U_p(c_p_surf, T0, param) * param.Phi_star)
    plt.plot(t_LION, U_p, 'o')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$U_{{\mathrm{{p}}}}$', fontsize=11)
    plt.subplot(2, 2, 4)
    plt.plot(t * param.tau_d_star,
             ocp.dUdT_p(c_p_surf, param) * param.Phi_star / param.Delta_T_star)
    plt.plot(t_LION, dUdT_p, 'o')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\mathrm{{d}}U_{{\mathrm{{p}}}}$'
               r'$ / \mathrm{{d}}T$', fontsize=11)
    fig.tight_layout()
