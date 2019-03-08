import autograd.numpy as np
from autograd import jacobian
from scikits.odes import dae
import matplotlib.pyplot as plt

from make_parameters import Parameters
from make_stiffness_matrix import fem_matrices
from make_mesh import FiniteVolumeMesh
import open_circuit_potentials as ocp
from make_rhs import rhs_particle
import make_plots as myplot
import utilities as ut

# Load parameters -------------------------------------------------------------
C_rate = 1
# param = Parameters(C_rate)
param = Parameters(C_rate, 'mypouch')


# Get stifnness matrix --------------------------------------------------------
Ny, Nz = 10, 10  # Number of gridpoints
degree = 1  # Degree of polynomial
K, load_tab_n, load_tab_p, Nyz_dofs = fem_matrices(param, Ny, Nz, degree)

param.Ny = Ny  # Add to param
param.Nz = Nz  # Add to param
param.Nyz_dofs = Nyz_dofs  # Add to param


# Make grids ------------------------------------------------------------------
mesh = FiniteVolumeMesh(param, 11, 2, 240, 3600)


# Initial conditions ----------------------------------------------------------
V_0 = (ocp.U_p(param.c_p_0, param.T_0, param)
       - ocp.U_p(param.c_n_0, param.T_0, param)) * np.ones(Nyz_dofs)
I_0 = np.zeros(Nyz_dofs)
c_n_0 = param.c_n_0*np.ones(Nyz_dofs * (mesh.Nr - 1))
c_p_0 = param.c_p_0*np.ones(Nyz_dofs * (mesh.Nr - 1))

# Concatenate variables
y_0 = np.concatenate([V_0, I_0, c_n_0, c_p_0])


# Make matrices for DAE system ------------------------------------------------
# NOTE: variables arranged as V, I, c_n, c_p, with c_n order by radial
# coordinate, then y,z coordinate (labelling as decided by dolfin)
# NOTE: should probably work with sparse matrices

# Total number of dofs
param.N_dofs = 2 * Nyz_dofs + 2 * Nyz_dofs * (mesh.Nr - 1)   # Add to param

# Mass matrix
M = np.zeros([param.N_dofs, param.N_dofs])
M[2*Nyz_dofs:, 2*Nyz_dofs:] = np.eye(2 * Nyz_dofs * (mesh.Nr - 1))

# Linear part
A = np.zeros([param.N_dofs, param.N_dofs])
A[0:Nyz_dofs, 0:Nyz_dofs] = K
A[0:Nyz_dofs, Nyz_dofs:2*Nyz_dofs] = param.alpha*np.eye(Nyz_dofs)
A[Nyz_dofs:2*Nyz_dofs, 0:Nyz_dofs] = np.eye(Nyz_dofs)

# Load vector (RHS)
b = np.zeros(param.N_dofs)
b[0:Nyz_dofs] = -(load_tab_n + load_tab_p)
# Remining entries depend on time and are computed during solve


def update_load(t, y, mesh, param):

    # Create load vector
    load = np.zeros(param.N_dofs)

    # Start indices for variables
    I_idx = param.Nyz_dofs
    c_n_idx = 2*param.Nyz_dofs
    c_p_idx = 2*param.Nyz_dofs + param.Nyz_dofs * (mesh.Nr - 1)

    # Get variables
    I_current = y[I_idx:c_n_idx]
    c_n = y[c_n_idx:c_p_idx]
    c_p = y[c_p_idx:]

    # Surface concentration for BV
    ind = np.arange(mesh.Nr - 2, np.size(c_n), mesh.Nr - 1)
    c_n_surf = c_n[ind] + (c_n[ind] - c_n[ind-1]) / 2
    c_p_surf = c_p[ind] + (c_p[ind] - c_p[ind-1]) / 2

    # SPM relation
    load[I_idx:c_n_idx] = SPM_fun(I_current, c_n_surf, c_p_surf, param)

    # FVM for concentration
    # NOTE: slow loop for now but eventually will be implemented using PyBaMM
    for i in range(param.Nyz_dofs):
        idx = (mesh.Nr - 1)*i
        c_n_slice = c_n[idx:idx+mesh.Nr-1]
        c_p_slice = c_p[idx:idx+mesh.Nr-1]
        dc_dt = rhs_particle(t, c_n_slice, c_p_slice, mesh, param, I_current[idx])

        load[c_n_idx + idx:c_n_idx + idx + mesh.Nr-1] = dc_dt[0:mesh.Nr-1]
        load[c_p_idx + idx:c_p_idx + idx + mesh.Nr-1] = dc_dt[mesh.Nr-1:]
    return load


def SPM_fun(I, c_n, c_p, param):
    g_n = param.m_n*param.C_hat_n*np.sqrt(c_n)*np.sqrt(1 - c_n)
    g_p = param.m_p*param.C_hat_p*np.sqrt(c_p)*np.sqrt(1 - c_p)
    result = (
        ocp.U_p(c_p, param.T_0, param) - ocp.U_n(param.c_n, param.T_0, param)
        - (2/param.Lambda)*np.arcsinh(I / (g_p * param.L_p))
        - (2/param.Lambda)*np.arcsinh(I / (g_n * param.L_n))
    )
    return result


# Define residual and Jacobian for DAE system ---------------------------------
def my_rhs(t, y):
        # Load vector
        new_load = update_load(t, y, mesh, param)
        b[param.Nyz_dofs:] = new_load[param.Nyz_dofs:]
        # dy/dt = rhs
        rhs = np.dot(A, y) + b
        return rhs


def my_residual(t, y, ydot, result):
    # Residual F(t, y, y_dot) = 0
    result[:] = (np.dot(M, ydot) - my_rhs(t, y))


# Calculate Jacobian of RHS
J = jacobian(my_rhs, 1)


def my_jac(self, t, y, ydot, cj, jac):
    # Need to write function to approximate jacobian of nonlinear part
    jac[:][:] = -J(t, y) + cj*M


# TO DO: Get consistant initial conditions ------------------------------------
y_dot_0 = np.zeros(np.size(y_0))


# Solve IVP -------------------------------------------------------------------
print("Solving ODE.")
ODE_start = mesh.t[0]
solver = dae("ida", my_residual, jacfn=my_jac, atol=1e-4, rtol=1e-4, old_api=False)
soln = solver.solve(mesh.t, y_0, y_dot_0)
ODE_finish = mesh.t[-1]
t_out = soln.values.t
y_out = np.transpose(soln.values.y)
