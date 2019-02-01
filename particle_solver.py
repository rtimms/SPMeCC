import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# from parameters import Parameters
from parameters import ParametersLIONSIMBA_pouch


from make_rhs import rhs_particle
from make_mesh import FiniteVolumeMesh

# Load parameters -------------------------------------------------------------
C_rate = 1
#  param = Parameters(C_rate)
param = ParametersLIONSIMBA_pouch(C_rate)

# Make grids ------------------------------------------------------------------
mesh = FiniteVolumeMesh(param, 101, 51, 100, 3600)

# Initial conditions
c_n_0 = param.c_n_0*np.ones(mesh.Nr - 1)
c_p_0 = param.c_p_0*np.ones(mesh.Nr - 1)
soln_0 = np.concatenate((c_n_0, c_p_0))

# Integrate in time for constant current --------------------------------------
I_app = 1*np.ones(1)
print('Solving SPMeCC.')
soln = odeint(rhs_particle, soln_0, mesh.t,
              args=(mesh, param, I_app))

# Extract variables -----------------------------------------------------------
c_n_idx = mesh.Nr - 1
c_p_idx = 2 * (mesh.Nr - 1)
c_n = soln[:, 0:c_n_idx]
c_p = soln[:, c_n_idx:c_p_idx]

# Plot surface concetrations --------------------------------------------------
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
fig = plt.figure(1)
plt.subplot(1, 2, 1)
p1 = plt.plot(mesh.t * param.tau_d_star, c_n[:, -1])
plt.xlabel(r'$t$ [s]', fontsize=11)
plt.ylabel('Surface 'r'$c_{{\mathrm{{n}}}}$', fontsize=11)
plt.subplot(1, 2, 2)
p2 = plt.plot(mesh.t * param.tau_d_star, c_p[:, -1])
plt.xlabel(r'$t$ [s]', fontsize=11)
plt.ylabel('Surface 'r'$c_{{\mathrm{{p}}}}$', fontsize=11)
fig.tight_layout()
plt.show()
