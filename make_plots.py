import numpy as np
import matplotlib.pyplot as plt
from dolfin import plot

import heat_generation as heat
from current_profile import current
import coefficient_functions as coeff


def get_vars(soln, mesh):
    c_n_idx = mesh.Nr - 1
    c_p_idx = 2 * (mesh.Nr - 1)
    c_e_n_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1))
    c_e_s_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1) + (mesh.Nx_s - 1))
    c_e_p_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1))

    c_n = soln[:, 0:c_n_idx]
    c_p = soln[:, c_n_idx:c_p_idx]
    c_e_n = soln[:, c_p_idx:c_e_n_idx]
    c_e_s = soln[:, c_e_n_idx:c_e_s_idx]
    c_e_p = soln[:, c_e_s_idx:c_e_p_idx]
    T0 = soln[:, -2]
    T1 = soln[:, -1]

    return c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1


def plot_heat_generation(soln, mesh, R_cn, R_cp, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(soln, mesh)
    t = mesh.t

    # Surface concentration for BV
    c_n_surf = c_n[:, -1]
    c_p_surf = c_p[:, -1]

    # Electrode avergaed electrolyte concentrations and the values at the
    # electrode/separator interfaces needed for heat source terms
    c_e_n_bar = np.trapz(c_e_n,
                         dx=mesh.dx_n, axis=1) / param.L_n
    c_e_p_bar = np.trapz(c_e_p,
                         dx=mesh.dx_p, axis=1) / param.L_p
    c_e_neg_sep = (c_e_n[:, -1] + c_e_s[:, 0]) / 2
    c_e_pos_sep = (c_e_s[:, -1] + c_e_p[:, 0]) / 2

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

    plt.subplot(2, 5, 6)
    plt.plot(t * param.tau_d_star,
             heat.ohmic_cc_1(R_cn, param, I_app),
             label="Ohm")
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.title('Negative c.c. first-order', fontsize=11)
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
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(soln, mesh)

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
                    / 6 / coeff.electrolyte_diffusivity(1, param))
    c_e_s_steady = ((1 / param.Ly) * (param.nu * (1 - param.t_plus))
                    * (2
                    * (param.L_p ** 2 / param.epsilon_p ** param.brug
                       - param.L_n ** 2 / param.epsilon_n ** param.brug)
                    + 3 * (param.L_n ** 2 - param.L_p ** 2 + 1
                           - 2 * mesh.x_s) / param.epsilon_s ** param.brug)
                    / 6 / coeff.electrolyte_diffusivity(1, param))
    c_e_p_steady = ((1 / param.Ly) * (param.nu * (1 - param.t_plus))
                    * (2
                       * (param.L_p ** 2 / param.epsilon_p ** param.brug
                           - param.L_n ** 2 / param.epsilon_n ** param.brug)
                        + 3 * param.L_s * (param.L_p - param.L_n - 1)
                        / param.epsilon_s ** param.brug
                        + 3 * ((1 - mesh.x_p) ** 2 - param.L_p ** 2)
                        / param.L_p / param.epsilon_p ** param.brug)
                    / 6 / coeff.electrolyte_diffusivity(1, param))

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot over time
    fig = plt.figure()
    for i in range(1, np.size(mesh.t)):
        # Evaluate I_app
        I_app = current(mesh.t[i-1], param)
        plt.clf()
        plt.plot((mesh.x_n[1:] + mesh.x_n[0:-1])/2, c_e_n[i, :], 'b-')
        plt.plot(mesh.x_n, c_e_n_steady * I_app, 'r--')
        plt.plot((mesh.x_s[1:] + mesh.x_s[0:-1])/2, c_e_s[i, :], 'b-')
        plt.plot(mesh.x_s, c_e_s_steady * I_app, 'r--')
        plt.plot((mesh.x_p[1:] + mesh.x_p[0:-1])/2, c_e_p[i, :], 'b-',
                 label="Unsteady")
        plt.plot(mesh.x_p, c_e_p_steady * I_app, 'r--',
                 label="Steady")
        plt.xlim([0, 1])
        plt.xlabel(r'$x$', fontsize=11)
        plt.ylabel(r'$c_{{\mathrm{{e}}}}^1', fontsize=11)
        plt.legend()
        fig.tight_layout()
        plt.pause(1)


def plot_surface_concentration(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(soln, mesh)

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot surface concentrations
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(mesh.t * param.tau_d_star, c_n[:, -1])
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{n}}}}$', fontsize=11)
    plt.subplot(1, 2, 2)
    plt.plot(mesh.t * param.tau_d_star, c_p[:, -1])
    plt.xlabel(r'$t$ [s]', fontsize=11)
    plt.ylabel('Surface 'r'$c_{{\mathrm{{p}}}}$', fontsize=11)
    fig.tight_layout()


def plot_temperature(soln, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(soln, mesh)

    # Font stuff
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Plot temperature
    fig = plt.figure()
    plt.plot(mesh.t * param.tau_d_star, T0 + param.delta * T1)
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
