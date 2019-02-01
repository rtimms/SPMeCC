import numpy as np

# Take from LIONSIMBA - not consistent with parameters chosen, need
# to look in literature for correct OCP for our system


def U_n(c, T, param):
    """
    Computes the open circuit potential in the negative electrode.
    Parameters
    ----------
    c: array_like
        Array of the discretised concentration at the current time.
    T: float
        The (leading-order) bulk temperature.
    param: object
        Object containing model parameters.
    Returns
    ----------
    array_like
        The non-dimensional open circuit potential.
    """
    U_ref = (0.7222 + 0.1387 * c + 0.029 * c ** 0.5 - 0.0172 / c
             + 0.0019 / (c ** 1.5) + 0.2808 * np.exp(0.9 - 15 * c)
             - 0.7984 * np.exp(0.4465 * c - 0.4108))
    result = (U_ref / param.Phi_star) + T * dUdT_n(c, param)
    return result


def U_p(c, T, param):
    """
    Computes the open circuit potential in the positive electrode.
    Parameters
    ----------
    c: array_like
        Array of the discretised concentration at the current time.
    T: float
        The (leading-order) bulk temperature.
    param: object
        Object containing model parameters.
    Returns
    ----------
    array_like
        The non-dimensional open circuit potential
    """
    U_ref = ((-4.656 + 88.669 * c ** 2 - 401.119 * c ** 4 + 342.909 * c ** 6
              - 462.471 * c ** 8 + 433.434 * c ** 10)
             / (-1 + 18.933 * c ** 2 - 79.532 * c ** 4 + 37.311 * c ** 6
                - 73.083 * c ** 8 + 95.96 * c ** 10))
    result = (U_ref / param.Phi_star) + T * dUdT_p(c, param)
    return result


def dUdT_n(c, param):
    """
    Computes the entropic heat coefficient in the negative electrode.
    Parameters
    ----------
    c: array_like
        Array of the discretised concentration at the current time.
    param: object
        Object containing model parameters.
    Returns
    ----------
    array_like
        The non-dimensional entropic heat coefficient.
    """
    # There is a typo in this eqn. in the original paper
    result = (0.001 * (0.005269056 + 3.299265709 * c - 91.79325798 * c ** 2
                       + 1004.911008 * c ** 3 - 5812.278127 * c ** 4
                       + 19329.7549 * c ** 5 - 37147.8947 * c ** 6
                       + 38379.18127 * c ** 7 - 16515.05308 * c ** 8))
    result = (result / (1 - 48.09287227 * c + 1017.234804 * c ** 2
                        - 10481.80419 * c ** 3 + 59431.3 * c ** 4
                        - 195881.6488 * c ** 5 + 374577.3152 * c ** 6
                        - 385821.1607 * c ** 7 + 165705.8597 * c ** 8))
    return result * (param.Delta_T_star / param.Phi_star)


def dUdT_p(c, param):
    """
    Computes the entropic heat coefficient in the positive electrode.
    Parameters
    ----------
    c: array_like
        Array of the discretised concentration at the current time.
    param: object
        Object containing model parameters.
    Returns
    ----------
    array_like
        The non-dimensional entropic heat coefficient.
    """
    result = (-0.001 * (0.199521039 - 0.928373822 * c
                        + 1.364550689000003 * c ** 2
                        - 0.6115448939999998 * c ** 3))
    result = (result / (1 - 5.661479886999997 * c
                        + 11.47636191 * c ** 2
                        - 9.82431213599998 * c ** 3
                        + 3.048755063 * c ** 4))
    return result * (param.Delta_T_star / param.Phi_star)
