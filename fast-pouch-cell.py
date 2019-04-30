import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import make_parameters
from effective_cc_resistance import solve_psi_W
import make_mesh
import make_rhs
import make_plots as myplot
import utilities as ut
from current_profile import current


# Load parameters -------------------------------------------------------------
C_rate = 1
param = make_parameters.Parameters(C_rate, "mypouch")

# Solve psi, W problems and compute effective resistance -----------------------
Ny, Nz = 64, 64  # Number of gridpoints
degree = 2  # Degree of polynomial
psi, W, R_CC, R_cn, R_cp = solve_psi_W(param, Ny, Nz, degree)

# Make grids ------------------------------------------------------------------
mesh = make_mesh.FiniteVolumeMesh(param, Nr=3, N_pts=16)

# number of points required by each variable
x_neg_pts = mesh.Nx_n
x_sep_pts = mesh.Nx_s
x_pos_pts = mesh.Nx_p
x_pts = x_neg_pts + x_sep_pts + x_pos_pts

y_pts = mesh.N_y
z_pts = mesh.N_z

r_neg_pts = mesh.Nr
r_pos_pts = mesh.Nr

macro_neg_pts = x_neg_pts * y_pts * z_pts
macro_pos_pts = x_pos_pts * y_pts * z_pts

total_neg_pts = macro_neg_pts * r_neg_pts
total_pos_pts = macro_pos_pts * r_pos_pts

# number of points for negative electrode particles
c_s_n_pts = total_neg_pts

# number of points for positive electrode particles
c_s_p_pts = total_pos_pts

# number of points for the electrolyte
c_e_n_pts = x_neg_pts
c_e_s_pts = x_sep_pts
c_e_p_pts = x_pos_pts
c_e_pts = x_pts

# Initial conditions ----------------------------------------------------------
c_n_0 = param.c_n_0 * np.ones(c_s_n_pts - 1)
c_p_0 = param.c_p_0 * np.ones(c_s_p_pts - 1)
c_e_0 = np.zeros(c_e_pts - 3)
y_0 = np.concatenate((c_n_0, c_p_0, c_e_0))


# Termination conditions --------------------------------------
def voltage_cutoff_wrapper(t, y):
    return ut.voltage_cutoff(t, y, psi, W, R_CC, param, mesh)


voltage_cutoff_wrapper.terminal = True

# Solve IVP --------------------------------------------
print("Solving Fast Pouch Cell.")
soln = solve_ivp(
    lambda t, y: make_rhs.fast_pouch_cell(t, y, psi, W, R_CC, param, mesh),
    [mesh.t[0], mesh.t[-1]],
    y_0,
    t_eval=mesh.t,
    rtol=1e-8,
    atol=1e-8,
    method="BDF",
    events=[voltage_cutoff_wrapper],
)

