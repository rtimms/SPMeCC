import numpy as np


def current(t, param, current_type='constant'):
        """
        Returns the applied current as a function of time (and possibly
        model parameters).

        Parameters
        ----------
        t: float or array_like
            Times at which the current is to be calculated.
        param: object
            Object containing model parameters.
        current_type: string
            The name of the desired current profile.

        Returns
        -------
        array_like
            The applied current.
        """
        if current_type == 'constant':
            current = np.ones(np.size(t))
        else:
            raise ValueError('Given current profile not recognised!')
        return current
