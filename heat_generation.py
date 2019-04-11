import numpy as np

import open_circuit_potentials as ocp


def rxn_n_0(T0, c_n_surf, param, I_app):
    """
    Computes leading-order heating due to electrochemical reactions in the
    negative electrode.
    Parameters
    ----------
    T0: float   
        The leading-order bulk temperature.
    c_n_surf: float
        The value of the concetration at the surface of the negative electrode
        particle.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The leading-order heating due to electrochemical reactions.
    """
    # Compute surface flux
    g_bar_n_0 = (param.m_n * param.C_hat_n
                 * c_n_surf ** (1/2) * (1 - c_n_surf) ** (1/2))

    # Compute heating
    result = ((I_app / param.Ly)
              * ((2 * (1 + param.Theta * T0) / param.Lambda)
                 * (np.arcsinh(I_app / (g_bar_n_0 * param.L_n * param.Ly)))))

    return result / param.L


def rxn_p_0(T0, c_p_surf, param, I_app):
    """
    Computes leading-order heating due to electrochemical reactions in the
    positive electrode.
    Parameters
    ----------
    T0: float
        The leading-order bulk temperature.
    c_p_surf: float
        The value of the concetration at the surface of the positive electrode
        particle.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The leading-order heating due to electrochemical reactions.
    """
    # Compute surface flux
    g_bar_p_0 = (param.m_p * param.C_hat_p
                 * c_p_surf ** (1/2) * (1 - c_p_surf) ** (1/2))

    # Compute heating
    result = ((I_app / param.Ly)
              * ((2 * (1 + param.Theta * T0) / param.Lambda)
                 * np.arcsinh(I_app / (g_bar_p_0 * param.L_p * param.Ly))))

    return result / param.L


def rev_n_0(T0, c_n_surf, param, I_app):
    """
    Computes leading-order reversible heating in the negative electrode.
    Parameters
    ----------
    T0: float
        The leading-order bulk temperature.
    c_n_surf: float
        The value of the concetration at the surface of the negative electrode
        particle.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The leading-order reversible heating.
    """
    result = ((I_app / param.Ly) * (1 / param.Theta + T0)
              * (ocp.dUdT_n(c_n_surf, param)))
    return result / param.L


def rev_p_0(T0, c_p_surf, param, I_app):
    """
    Computes leading-order reversible heating.
    Parameters
    ----------
    T0: float
        The leading-order bulk temperature.
    c_p_surf: float
        The value of the concetration at the surface of the positive electrode
        particle.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The leading-order reversible heating.
    """
    result = (-(I_app / param.Ly) * (1 / param.Theta + T0)
              * (ocp.dUdT_p(c_p_surf, param)))
    return result / param.L


def ohmic_cc_1(R, param, I_app):
    """
    Computes first-order Ohmic heating in the current collectors.
    Parameters
    ----------
    R: float
        The current collector effective resistance.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order Ohmic heating in the current collectors.
    """
    result = I_app ** 2 * R
    return result / param.L


def ohmic_n_1(c_e_n_bar, c_e_neg_sep, param, I_app):
    """
    Computes first-order Ohmic heating in the negative electrode.
    Parameters
    ----------
    c_e_n_bar: float
        The x-averaged electrolyte concetration in the negative electrode.
    c_e_neg_sep: float
        The value of the electrolyte concetration at the boundary of the
        negative electrode and separator.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order Ohmic heating in the negative electrode.
    """
    solid = (I_app * param.Ly) ** 2 * (param.L_n / 3 / param.sigma_n_prime)
    electrolyte = ((I_app * param.Ly) ** 2
                   * (param.nu / param.Lambda
                      / (param.epsilon_n ** param.brug)
                      / param.electrolyte_conductivity(1))
                   * (param.L_n / 3)
                   - (I_app * param.Ly)
                   * (2*(1 - param.t_plus) / param.Lambda)
                   * (c_e_neg_sep - c_e_n_bar))
    result = solid + electrolyte
    return result / param.L


def ohmic_s_1(c_e_neg_sep, c_e_pos_sep, param, I_app):
    """
    Computes first-order Ohmic heating in the separator.
    Parameters
    ----------
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
    float
        The first-order Ohmic heating in the separator.
    """
    result = ((I_app * param.Ly) ** 2
              * (param.nu / param.Lambda
              / (param.epsilon_s ** param.brug)
              / param.electrolyte_conductivity(1))
              * param.L_s
              - (I_app * param.Ly) * (2*(1 - param.t_plus) / param.Lambda)
              * (c_e_pos_sep - c_e_neg_sep))
    return result / param.L


def ohmic_p_1(c_e_p_bar, c_e_pos_sep, param, I_app):
    """
    Computes first-order Ohmic heating in the positive electrode.
    Parameters
    ----------
    c_e_p_bar: float
        The x-averaged electrolyte concetration in the positive electrode.
    c_e_pos_sep: float
        The value of the electrolyte concetration at the boundary of the
        positive electrode and separator.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order Ohmic heating in the positive electrode.
    """
    solid = (I_app * param.Ly) ** 2 * (param.L_p / 3 / param.sigma_p_prime)
    electrolyte = ((I_app * param.Ly) ** 2
                   * (param.nu / param.Lambda
                      / (param.epsilon_p ** param.brug)
                      / param.electrolyte_conductivity(1))
                   * (param.L_p / 3)
                   - (I_app * param.Ly)
                   * (2*(1 - param.t_plus) / param.Lambda)
                   * (-c_e_pos_sep + c_e_p_bar))
    result = solid + electrolyte
    return result / param.L


def rxn_n_1(T0, T1, c_n_surf, c_e_n_bar, param, I_app):
    """
    Computes first-order heating in the negative electrode due to
    electrochemical reactions.
    Parameters
    ----------
    T0: array_like
        Array of the leading-order temperature at the current time.
    T0: array_like
        Array of the first-order temperature at the current time.
    c_n_surf: float
        The value of the concetration at the surface of the negative electrode
        particle.
    c_e_n_bar: float
        The x-averaged electrolyte concetration in the negative electrode.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order heating in the negative electrode due to
        electrochemical reactions.
    """
    g_bar_n_0 = (param.m_n * param.C_hat_n
                 * c_n_surf ** (1/2) * (1 - c_n_surf) ** (1/2))

    eta_n_1 = ((- c_e_n_bar / param.Lambda / param.L_n)
               * np.tanh((np.arcsinh(I_app
                                     / (g_bar_n_0 * param.L_n * param.Ly))))
               + param.Theta * T1 / (1 + param.Theta * T0))
    result = (I_app / param.Ly)*eta_n_1
    return result / param.L


def rxn_p_1(T0, T1, c_p_surf, c_e_p_bar, param, I_app):
    """
    Computes first-order heating in the positive electrode due to
    electrochemical reactions.
    Parameters
    ----------
    T0: array_like
        Array of the leading-order temperature at the current time.
    T0: array_like
        Array of the first-order temperature at the current time.
    c_p_surf: float
        The value of the concetration at the surface of the positive electrode
        particle.
    c_e_p_bar: float
        The x-averaged electrolyte concetration in the positive electrode.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order heating in the positive electrode due to
        electrochemical reactions.
    """
    g_bar_p_0 = (param.m_p * param.C_hat_p
                 * c_p_surf ** (1/2) * (1 - c_p_surf) ** (1/2))

    eta_p_1 = ((- c_e_p_bar / param.Lambda / param.L_p)
               * np.tanh(-(np.arcsinh(I_app
                                      / (g_bar_p_0 * param.L_p * param.Ly))))
               + param.Theta * T1 / (1 + param.Theta * T0))
    result = -(I_app / param.Ly)*eta_p_1
    return result / param.L


def rev_n_1(T, c_n_surf, param, I_app):
    """
    Computes first-order reversible heating in the negative electrode.
    Parameters
    ----------
    T: float
        First-order temperature at the current time.
    c_n_surf: float
        The value of the concetration at the surface of the negative electrode
        particle.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order reversible heating in the negative electrode.
    """
    result = ((I_app / param.Ly) * T
              * ocp.dUdT_n(c_n_surf, param))
    return result / param.L


def rev_p_1(T, c_p_surf, param, I_app):
    """
    Computes first-order reversible heating in the positive electrode.
    Parameters
    ----------
    T: float
        First-order temperature at the current time.
    c_p_surf: float
        The value of the concetration at the surface of the positive electrode
        particle.
    param: object
        Object containing model parameters.
    I_app: float
        The applied current.
    Returns
    ----------
    float
        The first-order reversible heating in the positive electrode.
    """
    result = -((I_app / param.Ly) * T
               * ocp.dUdT_p(c_p_surf, param))
    return result / param.L
