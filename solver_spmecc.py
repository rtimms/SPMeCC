import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from make_parameters import Parameters
from effective_cc_resistance import solve_psi_W
from make_mesh import FiniteVolumeMesh
from make_rhs import rhs_spmecc
import make_plots as myplot
import utilities as ut
from current_profile import current


# Load parameters -------------------------------------------------------------
C_rate = 2
# param = Parameters(C_rate)
param = Parameters(C_rate, 'mypouch')

# Solve psi, W problems and compute effective resitance -----------------------
Ny, Nz = 64, 64  # Number of gridpoints
degree = 2  # Degree of polynomial
psi, W, R_CC, R_cn, R_cp = solve_psi_W(param, Ny, Nz, degree)

# Make grids ------------------------------------------------------------------
mesh = FiniteVolumeMesh(param, 11, 51, 3600, 3600)

# Initial conditions ----------------------------------------------------------
c_n_0 = param.c_n_0*np.ones(mesh.Nr - 1)
c_p_0 = param.c_p_0*np.ones(mesh.Nr - 1)
c_e_0 = np.zeros((mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1))
T0_0 = param.T_0*np.ones(1)
T1_0 = np.zeros(1)
y_0 = np.concatenate((c_n_0, c_p_0, c_e_0, T0_0, T1_0))

# Integrate in time -----------------------------------------------------------


# Termination events
def empty_particle_wrapper(t, y):
    return ut.empty_particle(t, y, mesh)


empty_particle_wrapper.terminal = True


def full_particle_wrapper(t, y):
    return ut.full_particle(t, y, mesh)


full_particle_wrapper.terminal = True


def voltage_cutoff_wrapper(t, y):
    return ut.voltage_cutoff(t, y, mesh, param, current(t, param))


voltage_cutoff_wrapper.terminal = True

# Solve IVP
print('Solving SPMeCC.')
soln = solve_ivp(
     lambda t, y: rhs_spmecc(t, y, mesh, R_cn, R_cp, param),
     [mesh.t[0], mesh.t[-1]],
     y_0,
     t_eval=mesh.t,
     rtol=1e-8,
     atol=1e-8,
     method='BDF',
     events=[voltage_cutoff_wrapper, full_particle_wrapper, empty_particle_wrapper]
     )

# Plot solution ---------------------------------------------------------------
makeplots = False

myplot.plot_voltage_breakdown(soln, mesh, R_CC, param)
plt.show()

if makeplots is True:
    # Static plots
    myplot.plot_terminal_voltage(soln, mesh, R_CC, param)
    myplot.plot_OCP(np.linspace(0, 1, 100), 0, param)
    myplot.plot_psi_W(psi, W, R_CC, param)
    myplot.plot_surface_concentration(soln, mesh, param)
    myplot.plot_temperature(soln, mesh, param)
    myplot.plot_heat_generation(soln, mesh, R_cn, R_cp, param)
    plt.show()
    # Plot as function of time
    myplot.plot_electrolyte_concentration(soln, mesh, param)
