import numpy as np


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
    D_e = np.exp(- 0.65 * c)
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
    kappa_e = 0.0911 + 1.9101 * c - 1.052 * c ** 2 + 0.1554 * c ** 3
    return kappa_e
