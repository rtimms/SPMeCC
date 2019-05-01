import numpy as np


class FiniteVolumeMesh(object):
    def __init__(
        self, param, Nr=101, N_pts=51, t_steps=100, t_final=3600, N_y=4, N_z=4
    ):
        """
        Generates the meshes used for the Finite Volume discretisation, as well
        as the array of times at which the solution is to be calculated.
        Parameters
        ----------
        param: object
            Object containing model parameters.
        Nr: int
            Number of meshpoints in the radial grid.
        N_pts: int
            Target number of meshpoints in the x-direction component grids.
        t_steps: int
            Number of timesteps at which the solution is to be computed.
        t_final: int
            Simulation stopping time in seconds.
        """
        # Set spherical grid
        self.Nr = Nr
        self.r = np.linspace(0, 1, Nr)
        self.dr = self.r[1] - self.r[0]
        self.r = np.reshape(self.r, [1, 1, 1, len(self.r)])

        # Aim to make grid as uniform as possible
        targetmeshsize = min(param.L_n, param.L_s, param.L_p) / N_pts

        # Negative electrode grid
        self.Nx_n = round(param.L_n / targetmeshsize) + 1
        self.x_n = np.linspace(0.0, param.L_n, self.Nx_n)
        self.dx_n = self.x_n[1] - self.x_n[0]
        self.x_n = np.reshape(self.x_n, [len(self.x_n), 1, 1])

        # Separator grid
        self.Nx_s = round(param.L_s / targetmeshsize) + 1
        self.x_s = np.linspace(param.L_n, 1 - param.L_p, self.Nx_s)
        self.dx_s = self.x_s[1] - self.x_s[0]
        self.x_s = np.reshape(self.x_s, [len(self.x_s), 1, 1])

        # Positive electrode grid
        self.Nx_p = round(param.L_p / targetmeshsize) + 1
        self.x_p = np.linspace(1 - param.L_p, 1, self.Nx_p)
        self.dx_p = self.x_p[1] - self.x_p[0]
        self.x_p = np.reshape(self.x_p, [len(self.x_p), 1, 1])

        # Times to compute solution at
        self.t_final = t_final / param.tau_d_star
        self.t_steps = t_steps
        self.t = np.linspace(0.0, self.t_final, self.t_steps)

        # add on the y-z directions for fast-pouch-cell
        self.N_y = N_y
        self.y = np.linspace(0, param.Ly)
        self.dy = self.y[1] - self.y[0]

        self.N_z = N_z
        self.z = np.linspace(0, 1)
        self.dz = self.z[1] - self.z[0]
