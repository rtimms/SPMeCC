#  import numpy as np


def solid_diffusivity_n(c, param):
    """
    Calculates the solid diffusivity in the negative electrode particle as
    a function of concentration.

    Parameters
    ----------
    c: array_like
        Array of concentration in each volume.
    param: object
        Object containing model parameters.

    Returns
    -------
    array_like
        The the value of the diffusivity at each given concentration.
    """
    return 1


def solid_diffusivity_p(c, param):
    """
    Calculates the solid diffusivity in the positive electrode particle as
    a function of concentration.

    Parameters
    ----------
    c: array_like
        Array of concentration in each volume.
    param: object
        Object containing model parameters.

    Returns
    -------
    array_like
        The the value of the diffusivity at each given concentration.
    """
    return 1


def electrolyte_diffusivity(c, param):
    """
    Calculates the electrolyte diffusivity as a function of concentration.

    Parameters
    ----------
    c: array_like
        Array of concentration in each volume.
    param: object
        Object containing model parameters.

    Returns
    -------
    array_like
        The the value of the diffusivity at each given concentration.
    """
    # From LIONSIMBA at ambient temperature
    # Make c dimensional
    c = c * param.c_e_typ_star
    exponent = (-4.43 - (54 / (param.T_inf_star - 229 - 5 * 1E-3 * c))
                - 0.22 * 1E-3 * c)
    D_e = 1E-4 * 10 ** exponent

    # Make D_e dimensionless
    D_e = D_e / param.D_e_typ_star  # Make dimensionless
    return D_e


def electrolyte_conductivity(c, param):
    """
    Calculates the electrolyte conductivity as a function of concentration.

    Parameters
    ----------
    c: array_like
        Array of concentration in each volume.
    param: object
        Object containing model parameters.

    Returns
    -------
    array_like
        The the value of the conductivity at each given concentration.
    """
    # From LIONSIMBA at ambient temperature
    # Make c dimensional
    c = c * param.c_e_typ_star

    temp = (-10.5 + 0.668 * 1E-3 * c + 0.494 * 1E-6 * c ** 2
            + (0.074 - 1.78 * 1E-5 * c - 8.86 * 1E-10 * c ** 2)
            * param.T_inf_star
            + (-6.96 * 1E-5 + 2.8 * 1E-8 * c) * param.T_inf_star ** 2)
    kappa_e = 1E-4 * c * temp ** 2

    # Make kappa_e dimensionless
    kappa_e = (kappa_e * param.Rg_star * param.T_inf_star
               / param.F_star ** 2 / param.D_e_typ_star / param.c_e_typ_star)
    return kappa_e
