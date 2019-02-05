import numpy as np

import open_circuit_potentials as ocp
from utilities import get_vars_time
from current_profile import current


class Voltage(object):

    def __init__(self, soln, mesh, R, param):
        """
        Computes terminal voltage components from SPMeCC model.
        Parameters
        ----------
        soln: array_like
            Object containing solution
        mesh: object
            Object containing information about the mesh.
        R: float
            The current collector effective resistance.
        param: object
            Object containing model parameters.
        """
        # Get variables
        c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars_time(soln.y, mesh)
        t = soln.t

        # Surface concentration for BV
        c_n_surf = c_n[-1, :] + (c_n[-1, :] - c_n[-2, :]) / 2
        c_p_surf = c_p[-1, :] + (c_p[-1, :] - c_p[-2, :]) / 2

        # Evaluate I_app
        I_app = current(t, param)

        self.U_eq_init = open_circuit(param.c_n_0, param.c_p_0,
                                      param.T_0, 0, param)
        self.U_eq = open_circuit(c_n_surf, c_p_surf, T0, T1, param)
        self.eta_r = reac_overpotential(c_n_surf, c_p_surf, c_e_n, c_e_p, T0,
                                        mesh, param, I_app)
        self.eta_c = conc_overpotential(c_e_n, c_e_p, mesh, param)
        self.Delta_Phi_elec = electrolyte_ohmic(param, I_app)
        self.Delta_Phi_solid = solid_ohmic(param, I_app)
        self.Delta_Phi_cc = cc_ohmic(R, param, I_app)
        self.v_term = (self.U_eq + self.eta_r + self.eta_c
                       + self.Delta_Phi_elec + self.Delta_Phi_solid
                       + self.Delta_Phi_cc)


def open_circuit(c_n_surf, c_p_surf, T0, T1, param):
    """
    Computes the open circuit voltage.
    Parameters
    ----------
    c_n_surf: array_like
        The value of the concetration at the surface of the negative electrode
        particle.
    c_n_surf: array_like
        The value of the concetration at the surface of the negative electrode
        particle.
    c_e: array_like
        Array of the electrolyte concetration.
    T0: array_like
        Array of the leading-order temperature.
    T1: array_like
        Array of the first-order temperature.
    param: object
        Object containing model parameters.
    Returns
    ----------
    float
        The open circuit voltage.
    """
    U_eq = (ocp.U_p(c_p_surf, T0, param) - ocp.U_n(c_n_surf, T0, param)
            + param.delta * (ocp.dUdT_p(c_p_surf, param) * T1
                             - ocp.dUdT_n(c_n_surf, param) * T1))
    return U_eq


def reac_overpotential(c_n_surf, c_p_surf, c_e_n, c_e_p, T0,
                       mesh, param, I_app):
    """
    Computes the reaction overpotential losses.
    Parameters
    ----------
    c_n_surf: array_like
        The value of the concetration at the surface of the negative electrode
        particle.
    c_n_surf: array_like
        The value of the concetration at the surface of the negative electrode
        particle.
    c_e_n: array_like
        Array of the electrolyte concetration in the negative electrode.
    c_e_p: array_like
        Array of the electrolyte concetration in the positive electrode.
    T0: array_like
        Array of the leading-order temperature.
    mesh: object
        Object containing information about the mesh.
    param: object
        Object containing model parameters.
    I_app: array_like
        The applied current.
    Returns
    ----------
    float
        The reaction overpotential losses.
    """
    g_n_bar = (param.m_n * param.C_hat_n / param.L_n
               * c_n_surf ** 0.5 * (1 - c_n_surf) ** 0.5
               * np.trapz((1 + param.delta * c_e_n) ** 0.5,
                          dx=mesh.dx_n, axis=0))
    g_p_bar = (param.m_p * param.C_hat_p / param.L_p
               * c_p_surf ** 0.5 * (1 - c_p_surf) ** 0.5
               * np.trapz((1 + param.delta * c_e_p) ** 0.5,
                          dx=mesh.dx_p, axis=0))
    eta_r = (- 2 * (1 + param.Theta * T0) / param.Lambda
             * np.arcsinh(I_app / g_p_bar / param.L_p / param.Ly)
             - 2 * (1 + param.Theta * T0) / param.Lambda
             * np.arcsinh(I_app / g_n_bar / param.L_n / param.Ly))
    return eta_r


def conc_overpotential(c_e_n, c_e_p, mesh, param):
    """
    Computes the concentration overpotential losses.
    Parameters
    ----------
    c_e_n: array_like
        Array of the electrolyte concetration in the negative electrode.
    c_e_p: array_like
        Array of the electrolyte concetration in the positive electrode.
    mesh: object
        Object containing information about the mesh.
    param: object
        Object containing model parameters.
    Returns
    ----------
    float
        The concentration overpotential losses.
    """
    c_e_n_bar = np.trapz(c_e_n, dx=mesh.dx_n, axis=0) / param.L_n
    c_e_p_bar = np.trapz(c_e_p, dx=mesh.dx_p, axis=0) / param.L_p
    eta_c = (2 * param.delta * (1 - param.t_plus) / param.Lambda
             * (c_e_p_bar - c_e_n_bar))
    return eta_c


def electrolyte_ohmic(param, I_app):
    """
    Computes the electrolyte Ohmic losses.
    Parameters
    ----------
    param: object
        Object containing model parameters.
    I_app: array_like
        The applied current.
    Returns
    ----------
    float
        The electrolyte Ohmic losses.
    """
    Delta_Phi_elec = (- (param.delta * param.nu * I_app
                         / param.Lambda / param.Ly
                         / param.electrolyte_conductivity(1))
                      * (param.L_n / 3 / param.epsilon_n ** param.brug
                          + param.L_s / param.epsilon_s ** param.brug
                          + param.L_p / 3 / param.epsilon_p ** param.brug))
    return Delta_Phi_elec


def solid_ohmic(param, I_app):
    """
    Computes the solid Ohmic losses.
    Parameters
    ----------
    param: object
        Object containing model parameters.
    I_app: array_like
        The applied current.
    Returns
    ----------
    float
        The solid Ohmic losses.
    """
    Delta_Phi_solid = (- I_app / 3 / param.Ly
                       * (param.L_p / param.sigma_p
                          + param.L_n / param.sigma_n))
    return Delta_Phi_solid


def cc_ohmic(R, param, I_app):
    """
    Computes the current collector Ohmic losses.
    Parameters
    ----------
    R: float
        The current collector effective resistance.
    param: object
        Object containing model parameters.
    I_app: array_like
        The applied current.
    Returns
    ----------
    float
        The current collector Ohmic losses.
    """
    Delta_Phi_cc = - param.delta * I_app * R
    return Delta_Phi_cc
