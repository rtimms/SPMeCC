import numpy as np

from current_profile import current
import open_circuit_potentials as ocp
import heat_generation as heat
import utilities as ut


def rhs_spmecc(t, soln, mesh, R_cn, R_cp, param):
    """
    Computes the rhs needed for the method of lines solution.
    Parameters
    ----------
    t: float
        Current time.
    soln: array_like
        Array of the discretised solution at the current time.
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

    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = ut.get_vars(soln, mesh)

    # Surface concentration for BV
    c_n_surf = c_n[-1] + (c_n[-1] - c_n[-2]) / 2
    c_p_surf = c_p[-1] + (c_p[-1] - c_p[-2]) / 2

    # Electrode avergaed electrolyte concentrations and the values at the
    # electrode/separator interfaces needed for heat source terms
    c_e_n_bar = np.trapz(c_e_n, dx=mesh.dx_n) / param.L_n
    c_e_p_bar = np.trapz(c_e_p, dx=mesh.dx_p) / param.L_p
    c_e_neg_sep = (c_e_n[-1] + c_e_s[0]) / 2
    c_e_pos_sep = (c_e_s[-1] + c_e_p[0]) / 2

    # Particle concentrations
    dck_dt = rhs_particle(t, c_n, c_p, mesh, param, I_app)

    # Electrolyte concentration
    c_e = np.concatenate([c_e_n, c_e_s, c_e_p])
    dce_dt = rhs_electrolye(t, c_e, mesh, param, I_app)

    # Temperature
    dT_dt = rhs_temperature(
        t,
        T0,
        T1,
        c_n_surf,
        c_p_surf,
        c_e_n_bar,
        c_e_p_bar,
        c_e_neg_sep,
        c_e_pos_sep,
        R_cn,
        R_cp,
        param,
        I_app,
    )

    # Concatenate RHS
    soln_rhs = np.concatenate((dck_dt, dce_dt, dT_dt))

    return soln_rhs


def rhs_spme(t, soln, mesh, param):
    """
    Computes the rhs needed for the method of lines solution of 1D SPMe.
    Parameters
    ----------
    t: float
        Current time.
    soln: array_like
        Array of the discretised solution at the current time.

    mesh: object
        Object containing information about the mesh.
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

    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = ut.get_vars(soln, mesh)

    # Surface concentration for BV
    c_n_surf = c_n[-1] + (c_n[-1] - c_n[-2]) / 2
    c_p_surf = c_p[-1] + (c_p[-1] - c_p[-2]) / 2

    # Electrode avergaed electrolyte concentrations and the values at the
    # electrode/separator interfaces needed for heat source terms
    c_e_n_bar = np.trapz(c_e_n, dx=mesh.dx_n) / param.L_n
    c_e_p_bar = np.trapz(c_e_p, dx=mesh.dx_p) / param.L_p
    c_e_neg_sep = (c_e_n[-1] + c_e_s[0]) / 2
    c_e_pos_sep = (c_e_s[-1] + c_e_p[0]) / 2

    # Particle concentrations
    dck_dt = rhs_particle(t, c_n, c_p, mesh, param, I_app)

    # Electrolyte concentration
    c_e = np.concatenate([c_e_n, c_e_s, c_e_p])
    dce_dt = rhs_electrolye(t, c_e, mesh, param, I_app)

    # Temperature
    dT_dt = rhs_temperature_spme(
        t,
        T0,
        T1,
        c_n_surf,
        c_p_surf,
        c_e_n_bar,
        c_e_p_bar,
        c_e_neg_sep,
        c_e_pos_sep,
        param,
        I_app,
    )

    # Concatenate RHS
    soln_rhs = np.concatenate((dck_dt, dce_dt, dT_dt))

    return soln_rhs


def rhs_particle(t, c_n, c_p, mesh, param, I_app):
    """
    Computes the rhs needed for the method of lines solution in the solid
    particles.
    Parameters
    ----------
    t: float
        Current time.
    c_n: array_like
        Array of the discretised concentration in the negative electrode.
    c_p: array_like
        Array of the discretised concentration in the positive electrode.
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
    # Compute fluxes
    q_n = -(
        param.gamma_n
        * param.solid_diffusivity_n((c_n[1:] + c_n[0:-1]) / 2)
        * mesh.r[1:-1] ** 2
        * (c_n[1:] - c_n[0:-1])
        / mesh.dr
    )
    q_n_surf = I_app / param.Ly / param.L_n / param.beta_n / param.C_hat_n
    q_p = -(
        param.gamma_p
        * param.solid_diffusivity_p((c_p[1:] + c_p[0:-1]) / 2)
        * mesh.r[1:-1] ** 2
        * (c_p[1:] - c_p[0:-1])
        / mesh.dr
    )
    q_p_surf = -I_app / param.Ly / param.L_p / param.beta_p / param.C_hat_p

    # Append boundary conditions
    q_n = np.concatenate(([0], q_n, q_n_surf))
    q_p = np.concatenate(([0], q_p, q_p_surf))

    # Compute discretised dc/dt
    V = (1 / 3) * (mesh.r[1:] ** 3 - mesh.r[0:-1] ** 3)
    dc_ndt = -(q_n[1:] - q_n[0:-1]) / V
    dc_pdt = -(q_p[1:] - q_p[0:-1]) / V
    dc_dt = np.concatenate((dc_ndt, dc_pdt))

    return dc_dt


def rhs_electrolye(t, c, mesh, param, I_app):
    """
    Computes the rhs needed for the method of lines solution in the
    electrolyte.
    Parameters
    ----------
    t: float
        Current time.
    c: array_like
        Array of the discretised concentration at the current time.
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
    dx = np.concatenate(
        (
            mesh.dx_n * np.ones(mesh.Nx_n - 1),
            mesh.dx_s * np.ones(mesh.Nx_s - 1),
            mesh.dx_p * np.ones(mesh.Nx_p - 1),
        )
    )

    # Bruggman correction of the effective diffusivity
    brug = np.concatenate(
        (
            param.epsilon_n ** param.brug * np.ones(mesh.Nx_n - 1),
            param.epsilon_s ** param.brug * np.ones(mesh.Nx_s - 1),
            param.epsilon_p ** param.brug * np.ones(mesh.Nx_p - 1),
        )
    )

    # Take harmonic mean of the brug coefficients (as in LIONSIMBA paper)
    beta = dx[0:1] / (dx[0:-1] + dx[1:])
    brug_eff = brug[0:-1] * brug[1:] / (beta * brug[0:-1] + (1 - beta) * brug[1:])

    # Compute flux
    N = -(
        brug_eff
        * param.electrolyte_diffusivity(1)
        * (c[1:] - c[0:-1])
        / (dx[1:] / 2 + dx[0:-1] / 2)
    )

    # Append boundary conditions
    N = np.concatenate(([0], N, [0]))

    # Compute source terms
    R_n = (param.nu * (1 - param.t_plus) * I_app / param.Ly / param.L_n) * np.ones(
        mesh.Nx_n - 1
    )
    R_s = np.zeros(mesh.Nx_s - 1)
    R_p = -(
        (param.nu * (1 - param.t_plus) * I_app / param.Ly / param.L_p)
        * np.ones(mesh.Nx_p - 1)
    )
    R = np.concatenate((R_n, R_s, R_p))

    # Compute discretised dc/dt
    dc_dt = (
        (-(N[1:] - N[0:-1]) / dx + R)
        / np.concatenate(
            (
                param.epsilon_n * np.ones(mesh.Nx_n - 1),
                param.epsilon_s * np.ones(mesh.Nx_s - 1),
                param.epsilon_p * np.ones(mesh.Nx_p - 1),
            )
        )
    ) / param.delta

    return dc_dt


def rhs_temperature(
    t,
    T0,
    T1,
    c_n_surf,
    c_p_surf,
    c_e_n_bar,
    c_e_p_bar,
    c_e_neg_sep,
    c_e_pos_sep,
    R_cn,
    R_cp,
    param,
    I_app,
):
    """
    Computes the rhs needed for the governing ODEs for the temperature.
    Parameters
    ----------
    t: float
        Current time.
    T0: array_like
        Array of the leading-order temperature at the current time.
    T: array_like
        Array of the first-order temperature at the current time.
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
    # Compute heat source terms
    Q_bar_0 = (
        heat.rxn_n_0(T0, c_n_surf, param, I_app)
        + heat.rxn_p_0(T0, c_p_surf, param, I_app)
        + heat.rev_n_0(T0, c_n_surf, param, I_app)
        + heat.rev_p_0(T0, c_p_surf, param, I_app)
    )
    Q_bar_1 = (
        heat.ohmic_cc_1(R_cn, param, I_app)
        + heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app)
        + heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app)
        + heat.ohmic_p_1(c_e_p_bar, c_e_pos_sep, param, I_app)
        + heat.ohmic_cc_1(R_cp, param, I_app)
        + heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app)
        + heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app)
        + heat.rev_n_1(T1, c_n_surf, param, I_app)
        + heat.rev_p_1(T1, c_p_surf, param, I_app)
    )
    Q_loss_0 = -2 * param.h_prime * T0 / param.L
    Q_loss_1 = (
        -2 * param.h_prime * T1 / param.L
        - (
            (2 * (param.Ly + 1) - (param.L_tab_n + param.L_tab_p))
            * (param.h_prime * T0 / param.Ly)
        )
        - (
            ((param.L_tab_n + param.L_tab_p) / param.L)
            * (param.h_tab_prime * (param.L_cn + param.L_cp) + param.h_prime)
            * T0
            / param.Ly
        )
    )
    # Compute discretised dT/dt
    dT0_dt = (param.gamma_th / param.rho) * (param.B * Q_bar_0 + Q_loss_0) * np.ones(1)

    dT1_dt = (param.gamma_th / param.rho) * (param.B * Q_bar_1 + Q_loss_1) * np.ones(1)

    dT_dt = np.concatenate((dT0_dt, dT1_dt))

    return dT_dt


def rhs_temperature_spme(
    t,
    T0,
    T1,
    c_n_surf,
    c_p_surf,
    c_e_n_bar,
    c_e_p_bar,
    c_e_neg_sep,
    c_e_pos_sep,
    param,
    I_app,
):
    """
    Computes the rhs needed for the governing ODEs for the temperature
    for the 1D comparison with LIONSIMBA.
    (Only cooling at edges x = 1+L_cp and -L_cp, not from other sides.)
    Parameters
    ----------
    t: float
        Current time.
    T0: array_like
        Array of the leading-order temperature at the current time.
    T1: array_like
        Array of the first-order temperature at the current time.
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
    # Compute heat source terms
    # Just add in Ohmi_cc heating as I ** 2 R at leading-order
    Q_bar_0 = (
        heat.rxn_n_0(T0, c_n_surf, param, I_app)
        + heat.rxn_p_0(T0, c_p_surf, param, I_app)
        + heat.rev_n_0(T0, c_n_surf, param, I_app)
        + heat.rev_p_0(T0, c_p_surf, param, I_app)
        + (I_app / param.Ly) ** 2 / param.sigma_cn
        + (I_app / param.Ly) ** 2 / param.sigma_cp
    )
    Q_bar_1 = (
        heat.ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app)
        + heat.ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app)
        + heat.ohmic_p_1(c_e_p_bar, c_e_pos_sep, param, I_app)
        + heat.rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app)
        + heat.rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app)
        + heat.rev_n_1(T1, c_n_surf, param, I_app)
        + heat.rev_p_1(T1, c_p_surf, param, I_app)
    )
    Q_loss_0 = -2 * param.h_prime * T0 / param.L
    Q_loss_1 = -2 * param.h_prime * T1 / param.L

    # Compute discretised dT/dt
    dT0_dt = (param.gamma_th / param.rho) * (param.B * Q_bar_0 + Q_loss_0) * np.ones(1)

    dT1_dt = (param.gamma_th / param.rho) * (param.B * Q_bar_1 + Q_loss_1) * np.ones(1)

    dT_dt = np.concatenate((dT0_dt, dT1_dt))

    return dT_dt

def fast_pouch_cell(t, y, mesh, psi, W, R_CC, param):
    # Find I_app
    I_app = current(t, param)

    # Get variables
    # TODO: change this to the form it should be
    c_s_n, c_s_p, c_e_n, c_e_s, c_e_p = ut.get_fast_pouch_cell_vars(y, mesh)

    # Surface concentration for BV
    c_n_surf = c_n[-1] + (c_n[-1] - c_n[-2]) / 2
    c_p_surf = c_p[-1] + (c_p[-1] - c_p[-2]) / 2

    # Electrode averaged electrolyte concentrations and the values at the
    # electrode/separator interfaces needed for heat source terms

    # Find voltage distribution ----------------------------------

    # average ocv
    T0 = 1
    u_n =  ocp.U_n(c_n_surf, T0, param)
    u_p =  ocp.U_p(c_p_surf, T0, param)
    u_eq = u_p - u_n

    u_n_av = np.mean(u_n)
    u_p_av = np.mean(u_p)
    u_eq_av = u_p_av - u_n_av

    # average reaction overpotential
    j0_n = param.m_n * param.C_hat_n / param.L_n
               * c_n_surf ** 0.5 * (1 - c_n_surf) ** 0.5
               * (c_e_n) ** 0.5
    j0_p = param.m_p * param.C_hat_p / param.L_p
               * c_p_surf ** 0.5 * (1 - c_p_surf) ** 0.5
               * (c_e_p) ** 0.5
    
    j0_n_av = np.mean(j_n)
    j0_p_av = np.mean(j_p)

    eta_n_av = 2 * np.arcsinh(I_app / j0_n_av / param.L_n / param.Ly)
    eta_p_av = - 2 * np.arcsinh(I_app / j0_p_av / param.L_p / param.Ly)
    eta_r_av = eta_p_av - eta_n_av
 
    # average concentration overpotential
    c_e_n_av = np.trapz(c_e_n, dx=mesh.dx_n, axis=0) / param.L_n
    c_e_p_av = np.trapz(c_e_p, dx=mesh.dx_p, axis=0) / param.L_p
    eta_c_av = (2 * (1 - param.t_plus) * np.log(c_e_p_av/c_e_n_av))
 
    # average electrolyte ohmic losses
    Delta_Phi_elec_av = (- (param.delta * param.nu * I_app
                         / param.Lambda / param.Ly
                         / param.electrolyte_conductivity(1))
                      * (param.L_n / 3 / param.epsilon_n ** param.brug
                          + param.L_s / param.epsilon_s ** param.brug
                          + param.L_p / 3 / param.epsilon_p ** param.brug))

    # average solid phase ohmic losses
    Delta_Phi_solid_av = (- I_app / 3 / param.Ly
                       * (param.L_p / param.sigma_p
                          + param.L_n / param.sigma_n))

    # Average through cell-voltage
    V_through_cell_av = u_eq_av + eta_r_av + eta_c_av + Delta_Phi_elec_av + Delta_Phi_solid_av

    # Average current collector ohmic losses
    Delta_Phi_CC = - param.delta * I_app * R_CC

    # Terminal voltage 
    V = V_through_cell_av + Delta_Phi_CC


    # Find current distribution ----------------------------------------------

    # Through-cell voltage distribution
    alpha = (1 / (param.epsilon ** 2 * param.sigma_cp * param.L_cp) 
            + 1 / (param.epsilon ** 2 * param.sigma_cn * param.L_cn) )

    V_through_cell = V_through_cell_av - (I_app / param.Ly) * alpha * W

    # negative current collector potential
    sigma_cn_prime = param.epsilon * param.sigma_cn
    sigma_cp_prime = param.epsilon * param.sigma_cp
    phi_c_n = (I_app * psi - sigma_cp_prime * para.L_cp * V_through_cell) / 
                ( sigma_cn_prime * param.L_cn + sigma_cp_prime * param.L_cp)

    # positive current collector potential
    phi_c_p = (I_app * psi + sigma_cn_prime * para.L_cn * V_through_cell) / 
                ( sigma_cn_prime * param.L_cn + sigma_cp_prime * param.L_cp)

    # negative solid potential
    phi_s_n = phi_c_n +  I_app / (2 * param.sigma_n * param.L_n * param.Ly) * 
                mesh.x * (2 * param.L_n - x)

    # positive solid potential
    phi_s_p = phi_c_p + I_app * (mesh.x_p - 1) 
               * (1 - 2 param.L_p - mesh.x_p) / 
               ( 2 * param.sigma_p * param.L_p * param.Ly)
 
    # electrolyte constant of integration
    phi_e_prime = - np.mean(u_n) - eta_n_av 
                    - 2 * (1-param.t_plus) * np.log(np.mean(c_e_n)) 
                    + (param.delta * param.nu * I_app * param.L_n) / 
                      (param.Ly * param.electrolyte_conductivity(1)) 
                    * (1 / (3 * param.epsilon_n**param.brug) - 
                       1/ (3 * param.epsilon_s ** param.brug) )
                    - I_app * param.L_n / 2 / param.sigma_n / param.Ly
                    + phi_c_n 
    # note: need to add phi_c_n in pouch cell case (value on boundary)

    # negative electrolyte potential
    phi_e_n = phi_e_prime + 2 * (1 - param.t_plus) * np.log(c_e_n)
              - ( param.delta * I_app * param.nu / 
              ( param.electrolyte_conductivity(1) * param.Ly) ) 
              * ( (mesh.x_n**2- param.L_n**2)/(2*param.epsilon_n**param.brug * param.L_n) 
                  + param.L_n / param.epsilon_s**b )

    # separator electrolyte potential
    phi_e_s = phi_e_prime + 2 * (1 - param.t_plus) * np.log(c_e_s)
              - ( param.delta * I_app * param.nu / 
              ( param.electrolyte_conductivity(1) * param.Ly) ) * mesh.x_s

    # positive electrolyte potential
    phi_e_p = phi_e_prime + 2 * (1 - param.t_plus) * np.log(c_e_p)
              - ( param.delta * I_app * param.nu / 
              ( param.electrolyte_conductivity(1) * param.Ly) ) 
              * ( (mesh.x_p * (2 - mesh.x_p) + param.L_p**2 - 1 ) /                 (2*param.epsilon_p**param.brug * param.L_p) 
                  + (1 - param.L_p) / param.epsilon_s**b )

    # reaction overpotentials
    eta_n = phi_s_n - phi_e_n - u_n
    eta_p = phi_s_p - phi_e_p - u_p
    
    # interfacial current density (j)
    j_n = j0_n * np.sinh(eta_n / 2)
    j_p = j0_p * np.sinh(eta_p / 2)


    # Update concentrations --------------------------------------------------

    # Update particle concentrations
    dcdt_s_n = rhs_many_particle(t, c_p, mesh, param, j_n)
    dcdt_s_p = rhs_many_particle(t, c_n, mesh, param, j_p)

    # Update electrolyte concentration
    c_e = np.concatenate([c_e_n, c_e_s, c_e_p])
    dcdt_e = rhs_electrolye(t, c_e, mesh, param, I_app/param.Ly)

    # Concatenate RHS
    return = np.concatenate((dck_dt, dce_dt, dT_dt))

