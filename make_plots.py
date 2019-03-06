import numpy as np
import matplotlib.pyplot as plt
from dolfin import plot

import heat_generation as heat
from current_profile import current
import open_circuit_potentials as ocp
from utilities import get_vars_time
from voltage import Voltage


def plot_terminal_voltage(soln, mesh, R_cc, param):
    # Create voltage object
    voltage = Voltage(soln, mesh, R_cc, param)

    # Convert to dimensional time
    t = soln.t * param.tau_d_star

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Make plots
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(t, voltage.v_term, label="V")
    plt.xlim([t[0], t[-1]])
    plt.ylim([param.V_min, param.V_max])
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Voltage [V]', fontsize=11)
    plt.title('Voltage', fontsize=11)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.stackplot(t,
                  voltage.U_eq - voltage.U_eq_init,
                  voltage.eta_r,
                  voltage.eta_c,
                  voltage.Delta_Phi_elec,
                  voltage.Delta_Phi_solid,
                  voltage.Delta_Phi_cc,
                  labels=[r'$\mathcal{{U}}_{{\mathrm{{eq}}}}$'
                          r'$- \mathcal{{U}}_{{\mathrm{{eq,init}}}}$ [V]',
                          r'$\eta_{{\mathrm{{r}}}}$',
                          r'$\eta_{{\mathrm{{c}}}}$',
                          r'$\Delta \Phi_{{\mathrm{{elec}}}}$',
                          r'$\Delta \Phi_{{\mathrm{{solid}}}}$',
                          r'$\Delta \Phi_{{\mathrm{{cc}}}}$'])
    plt.xlim([t[0], t[-1]])
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Voltage [V]', fontsize=11)
    plt.title('Voltage', fontsize=11)
    plt.legend()
    fig.tight_layout()

    fig2 = plt.figure()
    plt.subplot(2, 3, 1)
    plt.plot(t, voltage.U_eq - voltage.U_eq_init)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\mathcal{{U}}_{{\mathrm{{eq}}}}$'
               r'$- \mathcal{{U}}_{{\mathrm{{eq,init}}}}$ [V]')
    plt.subplot(2, 3, 2)
    plt.plot(t, voltage.eta_r)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\eta_{{\mathrm{{r}}}}$ [V]')
    plt.subplot(2, 3, 3)
    plt.plot(t, voltage.eta_c)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\eta_{{\mathrm{{c}}}}$ [V]')
    plt.subplot(2, 3, 4)
    plt.plot(t, voltage.Delta_Phi_elec)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\Delta \Phi_{{\mathrm{{elec}}}}$ [V]')
    plt.subplot(2, 3, 5)
    plt.plot(t, voltage.Delta_Phi_solid)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\Delta \Phi_{{\mathrm{{solid}}}}$ [V]')
    plt.subplot(2, 3, 6)
    plt.plot(t, voltage.Delta_Phi_cc)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$\Delta \Phi_{{\mathrm{{cc}}}}$ [V]')
    fig2.tight_layout()


def plot_voltage_breakdown(soln, mesh, R_cc, param):
    # Create voltage object
    voltage = Voltage(soln, mesh, R_cc, param)

    # Convert to dimensional time
    t = soln.t * param.tau_d_star

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick',labelsize=12)
    plt.rc('ytick',labelsize=12)

    # Make plots
    fig, ax1 = plt.subplots(figsize=(22/2.54, 12/2.54))
    left, bottom, width, height = [0.23, 0.23, 0.29, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    ax1.stackplot(t,
                  voltage.U_eq - voltage.U_eq_init,
                  voltage.eta_r,
                  voltage.eta_c,
                  voltage.Delta_Phi_elec,
                  voltage.Delta_Phi_solid,
                  voltage.Delta_Phi_cc,
                  labels=['OCV',
                          'Reaction overpotential',
                          'Concetration overpotential',
                          'Electrolyte Ohmic',
                          'Solid Ohmic',
                          'Current Collector Ohmic'])
    ax1.set_xlim([t[0], t[-1]])
    ax1.set_xlabel(r'$t$ [s]', fontsize=18)
    ax1.set_ylabel('Voltage loss [V]', fontsize=18)
    ax1.legend()
    ax2.plot(t, voltage.v_term, label="V")
    plt.xlim([t[0], t[-1]])
    plt.xlabel(r'$t$ [s]', fontsize=18)
    plt.ylabel('Voltage [V]', fontsize=18)
    plt.legend()
    plt.savefig('V_SPMeCC.eps', format='eps', dpi=1000)


def plot_heat_generation(soln, mesh, R_cn, R_cp, param):
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

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Make plots
    fig = plt.figure(figsize=(15, 9))

    plt.subplot(2, 5, 2)
    plt.plot(0, 0, label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_0(T0, c_n_surf, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_n_0(T0, c_n_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_0(T0, c_n_surf, param, I_app)
             + heat.rev_n_0(T0, c_n_surf, param, I_app),
             label="Total")
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Negative leading-order', fontsize=11)
    plt.legend()

    plt.subplot(2, 5, 4)
    plt.plot(0, 0, label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_0(T0, c_p_surf, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_p_0(T0, c_p_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_0(T0, c_p_surf, param, I_app)
             + heat.rev_p_0(T0, c_p_surf, param, I_app),
             label="Total")
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Positive leading-order', fontsize=11)
    plt.legend()

    plt.subplot(2, 5, 7)
    plt.plot(t * param.tau_d_star,
             heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app),
             label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_n_1(T1, c_n_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app)
             + heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app)
             + heat.rev_n_1(T1, c_n_surf, param, I_app),
             label='Total')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Negative first-order', fontsize=11)
    plt.legend()

    plt.subplot(2, 5, 8)
    plt.plot(t * param.tau_d_star,
             heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app),
             label="Ohm")
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Separator first-order', fontsize=11)
    plt.legend()

    plt.subplot(2, 5, 9)
    plt.plot(t * param.tau_d_star,
             heat.ohmic_p_1(c_e_p_bar, c_e_pos_sep, param, I_app),
             label="Ohm")
    plt.plot(t * param.tau_d_star,
             heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app),
             label="rxn")
    plt.plot(t * param.tau_d_star,
             heat.rev_p_1(T1, c_p_surf, param, I_app),
             label="rev")
    plt.plot(t * param.tau_d_star,
             heat.ohmic_p_1(c_e_p_bar, c_e_pos_sep, param, I_app)
             + heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app)
             + heat.rev_p_1(T1, c_p_surf, param, I_app),
             label='Total')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Positive first-order', fontsize=11)
    plt.legend()

    if R_cn == 0:
        plt.subplot(2, 5, 1)
        plt.plot(t * param.tau_d_star,
                 (I_app / param.Ly) ** 2 / param.sigma_cn,
                 label="Ohm")
        plt.xlabel(r'$t$ [s]', fontsize=11)
        plt.title('Negative c.c. 1D approx.', fontsize=11)
        plt.legend()
    else:
        plt.subplot(2, 5, 6)
        plt.plot(t * param.tau_d_star,
                 heat.ohmic_cc_1(R_cn, param, I_app),
                 label="Ohm")
        plt.xlabel(r'$t$ [s]', fontsize=11)
        plt.title('Negative c.c. first-order', fontsize=11)
        plt.legend()

    if R_cp == 0:
        plt.subplot(2, 5, 5)
        plt.plot(t * param.tau_d_star,
                 (I_app / param.Ly) ** 2 / param.sigma_cp,
                 label="Ohm")
        plt.xlabel(r'$t$ [s]', fontsize=11)
        plt.title('Positive c.c. 1D approx.', fontsize=11)
        plt.legend()
    else:
        plt.subplot(2, 5, 10)
        plt.plot(t * param.tau_d_star,
                 heat.ohmic_cc_1(R_cp, param, I_app),
                 label="Ohm")
        plt.xlabel(r'$t$ [s]', fontsize=11)
        plt.title('Positive c.c. first-order', fontsize=11)
        plt.legend()

    fig.tight_layout()


def plot_electrolyte_concentration(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)

    # Evaluate steady profiles
    # (need to multiply by I_app to get value at a given time)
    c_e_n_steady = ((1 / param.Ly) * (param.nu * (1 - param.t_plus))
                    * (2
                       * (param.L_p ** 2 / param.epsilon_p ** param.brug
                           - param.L_n ** 2 / param.epsilon_n ** param.brug)
                        + 3 * param.L_s * (param.L_p - param.L_n + 1)
                        / param.epsilon_s ** param.brug
                        + 3 * (param.L_n ** 2 - mesh.x_n ** 2)
                        / param.L_n / param.epsilon_n ** param.brug)
                    / 6 / param.electrolyte_diffusivity(1))
    c_e_s_steady = ((1 / param.Ly) * (param.nu * (1 - param.t_plus))
                    * (2
                    * (param.L_p ** 2 / param.epsilon_p ** param.brug
                       - param.L_n ** 2 / param.epsilon_n ** param.brug)
                    + 3 * (param.L_n ** 2 - param.L_p ** 2 + 1
                           - 2 * mesh.x_s) / param.epsilon_s ** param.brug)
                    / 6 / param.electrolyte_diffusivity(1))
    c_e_p_steady = ((1 / param.Ly) * (param.nu * (1 - param.t_plus))
                    * (2
                       * (param.L_p ** 2 / param.epsilon_p ** param.brug
                           - param.L_n ** 2 / param.epsilon_n ** param.brug)
                        + 3 * param.L_s * (param.L_p - param.L_n - 1)
                        / param.epsilon_s ** param.brug
                        + 3 * ((1 - mesh.x_p) ** 2 - param.L_p ** 2)
                        / param.L_p / param.epsilon_p ** param.brug)
                    / 6 / param.electrolyte_diffusivity(1))

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot over time
    fig = plt.figure()
    for i in range(1, np.size(soln.t)):
        # Evaluate I_app
        I_app = current(soln.t[i-1], param)
        plt.clf()
        plt.plot((mesh.x_n[1:] + mesh.x_n[0:-1])/2,
                 param.c_e_typ_star * param.delta * c_e_n[:, i], 'b-')
        plt.plot(mesh.x_n,
                 param.c_e_typ_star * param.delta * c_e_n_steady * I_app,
                 'r--')
        plt.plot((mesh.x_s[1:] + mesh.x_s[0:-1])/2,
                 param.c_e_typ_star * param.delta * c_e_s[:, i], 'b-')
        plt.plot(mesh.x_s,
                 param.c_e_typ_star * param.delta * c_e_s_steady * I_app,
                 'r--')
        plt.plot((mesh.x_p[1:] + mesh.x_p[0:-1])/2,
                 param.c_e_typ_star * param.delta * c_e_p[:, i], 'b-',
                 label="Unsteady")
        plt.plot(mesh.x_p,
                 param.c_e_typ_star * param.delta * c_e_p_steady * I_app,
                 'r--', label="Steady")
        plt.xlim([0, 1])
        plt.xlabel(r'$x$', fontsize=11)
        plt.ylabel(r'$c_{{\mathrm{{e}}}}^* - c_{{\mathrm{{e,typ}}}}^*$'
                   r'[mol/m$^3$]', fontsize=11)
        plt.legend()
        fig.tight_layout()
        plt.pause(1)


def plot_surface_concentration(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)

    # Surface concentration for BV
    c_n_surf = c_n[-1, :] + (c_n[-1, :] - c_n[-2, :]) / 2
    c_p_surf = c_p[-1, :] + (c_p[-1, :] - c_p[-2, :]) / 2

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot surface concentrations
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(soln.t * param.tau_d_star, c_n_surf)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{n}}}}$', fontsize=11)
    plt.subplot(1, 2, 2)
    plt.plot(soln.t * param.tau_d_star, c_p_surf)
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{p}}}}$', fontsize=11)
    fig.tight_layout()


def plot_temperature(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot temperature
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(soln.t * param.tau_d_star,
             T0 * param.Delta_T_star + param.T_inf_star,
             label=r'$T^0 + \delta T^1$')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$T$', fontsize=11)
    plt.subplot(2, 1, 2)
    plt.plot(soln.t * param.tau_d_star,
             (T0 + param.delta * T1) * param.Delta_T_star + param.T_inf_star,
             label=r'$T^0 + \delta T^1&')
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel(r'$T$', fontsize=11)
    fig.tight_layout()


def plot_psi_W(psi, W, R_CC, param):
    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot psi and W
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    p1 = plot(psi)
    plt.xlabel(r'$y$', fontsize=11)
    plt.ylabel(r'$z$', fontsize=11)
    plt.title(r"$\psi$", fontsize=11)
    plt.colorbar(p1, fraction=0.055, pad=0.04)
    plt.subplot(1, 2, 2)
    p2 = plot(W)
    plt.xlabel(r'$y$', fontsize=11)
    plt.ylabel(r'$z$', fontsize=11)
    plt.title(r"$\mathcal{W}^1$", fontsize=11)
    plt.colorbar(p2, fraction=0.055, pad=0.04)
    fig.suptitle(r'$\alpha^{{\prime}} = {:.3f}$. '
                 '$R_{{\mathrm{{CC}}}}$ = {:.3f}'
                 .format(param.alpha_prime, R_CC), fontsize=16)
    fig.tight_layout()


def plot_OCP(c, T, param):
    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot OCP and entropic coefficient at a fixed T
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(c, ocp.U_n(c, T, param) * param.Phi_star)
    plt.xlabel(r'$c_{{\mathrm{{n}}}}$')
    plt.ylabel(r'$U_{{\mathrm{{n}}}}$')
    plt.subplot(2, 2, 3)
    plt.plot(c, ocp.dUdT_n(c, param) * param.Phi_star / param.Delta_T_star)
    plt.xlabel(r'$c_{{\mathrm{{n}}}}$')
    plt.ylabel(r'$\frac{{\mathrm{{d}} U_{{\mathrm{{n}}}}}}{{\mathrm{{d}}T}}$')
    plt.subplot(2, 2, 2)
    plt.plot(c, ocp.U_p(c, T, param) * param.Phi_star)
    plt.xlabel(r'$c_{{\mathrm{{p}}}}$')
    plt.ylabel(r'$U_{{\mathrm{{p}}}}$')
    plt.subplot(2, 2, 4)
    plt.plot(c, ocp.dUdT_p(c, param) * param.Phi_star / param.Delta_T_star)
    plt.xlabel(r'$c_{{\mathrm{{p}}}}$')
    plt.ylabel(r'$\frac{{\mathrm{{d}} U_{{\mathrm{{p}}}}}}{{\mathrm{{d}}T}}$')
    fig.tight_layout()
