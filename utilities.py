import numpy as np
import open_circuit_potentials as ocp


def get_vars_time(soln, mesh):
    c_n_idx = mesh.Nr - 1
    c_p_idx = 2 * (mesh.Nr - 1)
    c_e_n_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1))
    c_e_s_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1) + (mesh.Nx_s - 1))
    c_e_p_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1))

    c_n = soln[0:c_n_idx, :]
    c_p = soln[c_n_idx:c_p_idx, :]
    c_e_n = soln[c_p_idx:c_e_n_idx, :]
    c_e_s = soln[c_e_n_idx:c_e_s_idx, :]
    c_e_p = soln[c_e_s_idx:c_e_p_idx, :]
    T0 = soln[-2, :]
    T1 = soln[-1, :]

    return c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1


def get_vars(soln, mesh):
    c_n_idx = mesh.Nr - 1
    c_p_idx = 2 * (mesh.Nr - 1)
    c_e_n_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1))
    c_e_s_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1) + (mesh.Nx_s - 1))
    c_e_p_idx = (2 * (mesh.Nr - 1)
                 + (mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1))

    c_n = soln[0:c_n_idx]
    c_p = soln[c_n_idx:c_p_idx]
    c_e_n = soln[c_p_idx:c_e_n_idx]
    c_e_s = soln[c_e_n_idx:c_e_s_idx]
    c_e_p = soln[c_e_s_idx:c_e_p_idx]
    T0 = soln[-2]
    T1 = soln[-1]

    return c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1


def voltage_cutoff(t, y, mesh, param):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(y, mesh)

    # Surface concentration for BV
    c_n_surf = c_n[-1] + (c_n[-1] - c_n[-2]) / 2
    c_p_surf = c_p[-1] + (c_p[-1] - c_p[-2]) / 2

    # OCV
    OCV = ocp.U_p(c_p_surf, T0, param) - ocp.U_n(c_n_surf, T0, param)

    TOL = 1e-3
    return OCV - param.V_min - TOL


def empty_particle(t, y, mesh):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(y, mesh)
    TOL = 1E-3
    return np.min([np.min(c_n), np.min(c_p)]) - 0.1 - TOL


def full_particle(t, y, mesh):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(y, mesh)
    TOL = 1E-3
    return 0.99 - (np.max([np.max(c_n), np.max(c_p)]) + TOL)
