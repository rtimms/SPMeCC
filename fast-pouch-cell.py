import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from make_parameters
from effective_cc_resistance import solve_psi_W
import make_mesh
import make_rhs
import make_plots as myplot
import utilities as ut
from current_profile import current


# Load parameters -------------------------------------------------------------
C_rate = 2
param = make_parameters.Parameters(C_rate, "mypouch")

# Solve psi, W problems and compute effective resitance -----------------------
Ny, Nz = 64, 64  # Number of gridpoints
degree = 2  # Degree of polynomial
psi, W, R_CC, R_cn, R_cp = solve_psi_W(param, Ny, Nz, degree)

# Make grids ------------------------------------------------------------------
mesh = make_mesh.FiniteVolumeMesh(param, Nr=7, N_pts=30)

# Initial conditions ----------------------------------------------------------
c_n_0 = param.c_n_0 * np.ones(mesh.Nr - 1)
c_p_0 = param.c_p_0 * np.ones(mesh.Nr - 1)
c_e_0 = np.zeros((mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1))


# Termination conditions
# TODO: change this to account for the many particles
def voltage_cutoff_wrapper(t, y):
    return ut.voltage_cutoff(t, y, mesh, param, current(t, param))

voltage_cutoff_wrapper.terminal = True

# Solve IVP
print("Solving Fast Pouch Cell.")
soln = solve_ivp(
    lambda t, y: make_rhs.rhs_spmecc(t, y, mesh, R_cn, R_cp, param),
    [mesh.t[0], mesh.t[-1]],
    y_0,
    t_eval=mesh.t,
    rtol=1e-8,
    atol=1e-8,
    method="BDF",
    events=[voltage_cutoff_wrapper, full_particle_wrapper, empty_particle_wrapper],
)

