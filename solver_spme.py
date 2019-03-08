import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from make_parameters import Parameters
from make_mesh import FiniteVolumeMesh
from make_rhs import rhs_spme
import make_plots_LIONSIMBA as myplot
import utilities as ut

# Load parameters -------------------------------------------------------------
C_rate = 1.0
# param = Parameters(C_rate)
param = Parameters(C_rate, 'mypouch')

# Make grids ------------------------------------------------------------------
mesh = FiniteVolumeMesh(param, 101, 49,
                        (3600 / 15 / C_rate) + 1, 3600 / C_rate)

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
    return ut.voltage_cutoff(t, y, mesh, param)


voltage_cutoff_wrapper.terminal = True

# Solve IVP
print('Solving SPMe.')
soln = solve_ivp(
     lambda t, y: rhs_spme(t, y, mesh, param),
     [mesh.t[0], mesh.t[-1]],
     y_0,
     t_eval=mesh.t,
     rtol=1e-8,
     atol=1e-8,
     method='BDF',
     events=[voltage_cutoff_wrapper, empty_particle_wrapper, full_particle_wrapper]
     )

# Plot solution ---------------------------------------------------------------
makeplots = 'True'

if makeplots == 'True':
    # Static plots
    myplot.plot_voltage(soln, mesh, 0, param)
    myplot.plot_temperature(soln, mesh, param)
    myplot.plot_surface_concentration(soln, mesh, param)
    #  myplot.plot_electrolyte_concentration(soln, mesh, param, 1800)
    myplot. plot_heat_generation(soln, mesh, param)
    myplot.plot_OCP(soln, mesh, param)
    plt.show()
