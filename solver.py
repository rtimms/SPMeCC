import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from make_parameters import Parameters
from effective_cc_resistance import solve_psi_W
from make_mesh import FiniteVolumeMesh
from make_rhs import rhs_spmecc
import make_plots as myplot

# Load parameters -------------------------------------------------------------
C_rate = 1
# param = Parameters(C_rate)
param = Parameters(C_rate, 'mypouch')

# Solve psi, W problems and compute effective resitance -----------------------
Ny, Nz = 64, 64  # Number of gridpoints
degree = 2  # Degree of polynomial
psi, W, R_CC, R_cn, R_cp = solve_psi_W(param, Ny, Nz, degree)

# Make grids ------------------------------------------------------------------
mesh = FiniteVolumeMesh(param, 101, 51, 240, 3600)

# Initial conditions ----------------------------------------------------------
c_n_0 = param.c_n_0*np.ones(mesh.Nr - 1)
c_p_0 = param.c_p_0*np.ones(mesh.Nr - 1)
c_e_0 = np.zeros((mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1))
T0_0 = param.T_0*np.ones(1)
T1_0 = np.zeros(1)
soln_0 = np.concatenate((c_n_0, c_p_0, c_e_0, T0_0, T1_0))

# Integrate in time -----------------------------------------------------------
print('Solving SPMeCC.')
soln = odeint(rhs_spmecc, soln_0, mesh.t,
              args=(mesh, R_cn, R_cp, param))

# Plot solution ---------------------------------------------------------------
makeplots = 'True'

if makeplots == 'True':
    # Static plots
    myplot.plot_psi_W(psi, W, R_CC, param)
    myplot.plot_surface_concentration(soln, mesh, param)
    myplot.plot_temperature(soln, mesh, param)
    myplot.plot_heat_generation(soln, mesh, R_cn, R_cp, param)
    plt.show()
    # Plot as function of time
    myplot.plot_electrolyte_concentration(soln, mesh, param)
