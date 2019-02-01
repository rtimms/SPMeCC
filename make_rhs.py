import numpy as np

from current_profile import current
import coefficient_functions as coeff
import heat_generation as heat


def rhs_spmecc(soln, t, mesh, R_cn, R_cp, param):
    """
    Computes the rhs needed for the method of lines solution.
    Parameters
    ----------
    soln: array_like
        Array of the discretised solution at the current time.
    t: float
        Current time.
    mesh: object
        Object containing information about the mesh.
    R_cn: float
        The effective negative current collector resistance calculated for
        Ohmic heating.
    R_cp: float
        The effective positive current collector resistance calculated for
        Ohmic heating.
    param: object
        Object containing model parameters.
    Returns
    ----------
    array_like
        Array containing the discretised verisons of the right hand sides of
        the SPMeCC model equations.
    """
    # Evaluate I_app
    I_app = current(t, param)

    # Extract components
    c_k = soln[0:2*(mesh.Nr-1)]
    c_e = soln[2*(mesh.Nr-1):-2]
    T = soln[-2:]

    # Surface concentration for BV
    c_n_surf = c_k[mesh.Nr-1]  # + (c_k[mesh.Nr-1] - c_k[mesh.Nr-2]) / 2
    c_p_surf = c_k[-1]  # + (c_k[-1] - c_k[-2]) / 2

    # Electrode avergaed electrolyte concentrations and the values at the
    # electrode/separator interfaces needed for heat source terms
    c_e_n_bar = np.trapz(c_e[0:mesh.Nx_n-1],
                         dx=mesh.dx_n) / param.L_n
    c_e_p_bar = np.trapz(c_e[(mesh.Nx_n-1) + (mesh.Nx_s-1):],
                         dx=mesh.dx_p) / param.L_p
    c_e_neg_sep = (c_e[mesh.Nx_n-2] + c_e[mesh.Nx_n-1]) / 2
    c_e_pos_sep = (c_e[-(mesh.Nx_p-1)-1] + c_e[-(mesh.Nx_p-1)]) / 2

    # Particle concentrations
    dck_dt = rhs_particle(c_k, t, mesh, param, I_app)

    # Electrolyte concentration
    dce_dt = rhs_electrolye(c_e, t, mesh, param, I_app)

    # Temperature
    dT_dt = rhs_temperature(T, t, c_n_surf, c_p_surf,
                            c_e_n_bar, c_e_p_bar, c_e_neg_sep, c_e_pos_sep,
                            R_cn, R_cp, param, I_app)
    # dT_dt = np.zeros(np.size(T))
    # Concatenate RHS
    soln_rhs = np.concatenate((dck_dt, dce_dt, dT_dt))

    return soln_rhs


def rhs_particle(c, t, mesh, param, I_app):
    """
    Computes the rhs needed for the method of lines solution in the solid
    particles.
    Parameters
    ----------
    c: array_like
        Array of the discretised concentration at the current time.
    t: float
        Current time.
    mesh: object
        Object containing information about the mesh.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    array_like
        Array containing the discretised verisons of the right hand sides of
        the solid particle model equations.
    """
    # Get individual concentrations
    c_n = c[0:mesh.Nr-1]
    c_p = c[mesh.Nr-1:]

    # Compute fluxes
    q_n = - (param.gamma_n *
             coeff.solid_diffusivity_n((c_n[1:] + c_n[0:-1]) / 2, param)
             * mesh.r[1:-1] ** 2 * (c_n[1:] - c_n[0:-1]) / mesh.dr)
    q_n_surf = I_app / param.Ly / param.L_n / param.beta_n / param.C_hat_n
    q_p = - (param.gamma_p *
             coeff.solid_diffusivity_p((c_p[1:] + c_p[0:-1]) / 2, param)
             * mesh.r[1:-1] ** 2 * (c_p[1:] - c_p[0:-1]) / mesh.dr)
    q_p_surf = -I_app / param.Ly / param.L_p / param.beta_p / param.C_hat_p

    # Append boundary conditions
    q_n = np.concatenate(([0], q_n, q_n_surf))
    q_p = np.concatenate(([0], q_p, q_p_surf))

    # Compute discretised dc/dt
    V = (1 / 3) * (mesh.r[1:] ** 3 - mesh.r[0:-1] ** 3)
    dc_ndt = - (q_n[1:] - q_n[0:-1]) / V
    dc_pdt = - (q_p[1:] - q_p[0:-1]) / V
    dc_dt = np.concatenate((dc_ndt, dc_pdt))

    return dc_dt


def rhs_electrolye(c, t, mesh, param, I_app):
    """
    Computes the rhs needed for the method of lines solution in the
    electrolyte.
    Parameters
    ----------
    c: array_like
        Array of the discretised concentration at the current time.
    t: float
        Current time.
    mesh: object
        Object containing information about the mesh.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    array_like
        Array containing the discretised verisons of the right hand sides of
        the electrolyte model equations.
    """
    # Compute concatenated spacing
    dx = np.concatenate((mesh.dx_n*np.ones(mesh.Nx_n - 1),
                         mesh.dx_s*np.ones(mesh.Nx_s - 1),
                         mesh.dx_p*np.ones(mesh.Nx_p - 1)))

    # Bruggman correction of the effective diffusivity
    brug = np.concatenate((param.epsilon_n ** param.brug
                           * np.ones(mesh.Nx_n - 1),
                          param.epsilon_s ** param.brug
                          * np.ones(mesh.Nx_s - 1),
                          param.epsilon_p ** param.brug
                          * np.ones(mesh.Nx_p - 1)))

    # Take harmonic mean of the brug coefficients (as in LIONSIMBA paper)
    beta = dx[0:1] / (dx[0:-1] + dx[1:])
    brug_eff = (brug[0:-1] * brug[1:]
                / (beta * brug[0:-1] + (1 - beta) * brug[1:]))

    # Compute flux
    N = - (brug_eff * coeff.electrolyte_diffusivity(1, param)
           * (c[1:] - c[0:-1]) / (dx[1:] / 2 + dx[0:-1] / 2))

    # Append boundary conditions
    N = np.concatenate(([0], N, [0]))

    # Compute source terms
    R_n = ((param.nu * (1 - param.t_plus) * I_app / param.Ly / param.L_n)
           * np.ones(mesh.Nx_n-1))
    R_s = np.zeros(mesh.Nx_s-1)
    R_p = - ((param.nu * (1 - param.t_plus) * I_app / param.Ly / param.L_p)
             * np.ones(mesh.Nx_p-1))
    R = np.concatenate((R_n, R_s, R_p))

    # Compute discretised dc/dt
    dc_dt = (((-(N[1:] - N[0:-1]) / dx + R)
             / np.concatenate((param.epsilon_n * np.ones(mesh.Nx_n - 1),
                              param.epsilon_s * np.ones(mesh.Nx_s - 1),
                              param.epsilon_p * np.ones(mesh.Nx_p - 1))))
             / param.delta)

    return dc_dt


def rhs_temperature(T, t, c_n_surf, c_p_surf,
                    c_e_n_bar, c_e_p_bar, c_e_neg_sep, c_e_pos_sep,
                    R_cn, R_cp, param, I_app):
    """
    Computes the rhs needed for the governing ODEs for the temperature.
    Parameters
    ----------
    T: array_like
        Array of the leading- and first-order temperature at the current time.
    t: float
        Current time.
    c_n_surf: float
        The value of the concetration at the surface of the negative electrode
        particle.
    c_p_surf: float
        The value of the concetration at the surface of the positive electrode
        particle.
    c_e_n_bar: float
        The x-averaged electrolyte concetration in the negative electrode.
    c_e_p_bar: float
        The x-averaged electrolyte concetration in the positive electrode.
    c_e_neg_sep: float
        The value of the electrolyte concetration at the boundary of the
        negative electrode and separator.
    c_e_pos_sep: float
        The value of the electrolyte concetration at the boundary of the
        positive electrode and separator.
    R_cn: float
        The effective negative current collector resistance calculated for
        Ohmic heating.
    R_cp: float
        The effective positive current collector resistance calculated for
        Ohmic heating.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    array_like
        Array containing the right hand sides of the ODEs governing the
        temperature.
    """
    # Split T0 and T1, and the concentration
    T0, T1 = T

    # Compute heat source terms
    Q_bar_0 = (heat.rxn_n_0(T0, c_n_surf, param, I_app)
               + heat.rxn_p_0(T0, c_p_surf, param, I_app)
               + heat.rev_n_0(T0, c_n_surf, param, I_app)
               + heat.rev_p_0(T0, c_p_surf, param, I_app))
    Q_bar_1 = (heat.ohmic_cc_1(R_cn, param, I_app)
               + heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app)
               + heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app)
               + heat.ohmic_p_1(c_e_p_bar, c_e_pos_sep, param, I_app)
               + heat.ohmic_cc_1(R_cp, param, I_app)
               + heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app)
               + heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app)
               + heat.rev_n_1(T1, c_n_surf, param, I_app)
               + heat.rev_p_1(T1, c_p_surf, param, I_app))
    Q_loss_0 = - 2 * param.h_prime * T0 / param.L
    Q_loss_1 = (- 2 * param.h_prime * T1 / param.L
                - ((2 * (param.Ly + 1)
                   - (param.L_tab_n + param.L_tab_p))
                   * (param.h_prime * T0 / param.Ly))
                - (((param.L_tab_n + param.L_tab_p) / param.L)
                   * (param.h_tab_prime*(param.L_cn + param.L_cp)
                      + param.h_prime) * T0 / param.Ly))
    # Compute discretised dT/dt
    dT0_dt = ((param.gamma_th / param.rho)
              * (param.B * Q_bar_0 + Q_loss_0)*np.ones(1))

    dT1_dt = ((param.gamma_th / param.rho)
              * (param.B * Q_bar_1 + Q_loss_1)*np.ones(1))

    dT_dt = np.concatenate((dT0_dt, dT1_dt))

    return dT_dt
