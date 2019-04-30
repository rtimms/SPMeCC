import numpy as np
import open_circuit_potentials as ocp
from current_profile import current


def get_vars_time(soln, mesh):
    c_n_idx = mesh.Nr - 1
    c_p_idx = 2 * (mesh.Nr - 1)
    c_e_n_idx = 2 * (mesh.Nr - 1) + (mesh.Nx_n - 1)
    c_e_s_idx = 2 * (mesh.Nr - 1) + (mesh.Nx_n - 1) + (mesh.Nx_s - 1)
    c_e_p_idx = 2 * (mesh.Nr - 1) + (mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1)

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
    c_e_n_idx = 2 * (mesh.Nr - 1) + (mesh.Nx_n - 1)
    c_e_s_idx = 2 * (mesh.Nr - 1) + (mesh.Nx_n - 1) + (mesh.Nx_s - 1)
    c_e_p_idx = 2 * (mesh.Nr - 1) + (mesh.Nx_n - 1) + (mesh.Nx_s - 1) + (mesh.Nx_p - 1)

    c_n = soln[0:c_n_idx]
    c_p = soln[c_n_idx:c_p_idx]
    c_e_n = soln[c_p_idx:c_e_n_idx]
    c_e_s = soln[c_e_n_idx:c_e_s_idx]
    c_e_p = soln[c_e_s_idx:c_e_p_idx]
    T0 = soln[-2]
    T1 = soln[-1]

    return c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1


def get_fast_pouch_cell_vars(soln, mesh):

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

    # indexes
    c_s_n_idx = c_s_n_pts - 1
    c_s_p_idx = c_s_n_idx + c_s_p_pts - 1
    c_e_n_idx = c_s_p_idx + c_e_n_pts - 1
    c_e_s_idx = c_e_n_idx + c_e_s_pts - 1
    c_e_p_idx = c_e_s_idx + c_e_p_pts - 1

    # extract entries
    c_s_n = soln[0:c_s_n_idx]
    c_s_p = soln[c_s_n_idx:c_s_p_idx]
    c_e_n = soln[c_s_p_idx:c_e_n_idx]
    c_e_s = soln[c_e_n_idx:c_e_s_idx]
    c_e_p = soln[c_e_s_idx:c_e_p_idx]

    # reshape particles
    neg_shape = [x_neg_pts, y_pts, z_pts, r_neg_pts]
    c_s_n = np.reshape(c_s_n, neg_shape)

    pos_shape = [x_pos_pts, y_pts, z_pts, r_pos_pts]
    c_s_p = np.reshape(c_s_p, pos_shape)

    return c_s_n, c_s_p, c_e_n, c_e_s, c_e_p


def voltage_cutoff(t, y, psi, W, R_CC, param, mesh):
    # Find I_app
    I_app = current(t, param)

    # Get variables
    c_s_n, c_s_p, c_e_n, c_e_s, c_e_p = get_fast_pouch_cell_vars(y, mesh)

    # Surface concentration for BV
    c_s_n_surf = c_s_n[:, :, -1] + (c_s_n[:, :, -1] - c_s_n[:, :, -2]) / 2
    c_s_p_surf = c_s_p[:, :, -1] + (c_s_p[:, :, -1] - c_s_p[:, :, -2]) / 2

    # average ocv
    T0 = 1
    u_n = ocp.U_n(c_s_n_surf, T0, param)
    u_p = ocp.U_p(c_s_p_surf, T0, param)
    u_eq = u_p - u_n

    u_n_av = np.mean(u_n)
    u_p_av = np.mean(u_p)
    u_eq_av = u_p_av - u_n_av

    # average reaction overpotential
    j0_n = (
        param.m_n
        * param.C_hat_n
        / param.L_n
        * c_s_n_surf ** 0.5
        * (1 - c_s_n_surf) ** 0.5
        * (c_e_n) ** 0.5
    )
    j0_p = (
        param.m_p
        * param.C_hat_p
        / param.L_p
        * c_s_p_surf ** 0.5
        * (1 - c_s_p_surf) ** 0.5
        * (c_e_p) ** 0.5
    )

    j0_n_av = np.mean(j0_n)
    j0_p_av = np.mean(j0_p)

    eta_n_av = 2 * np.arcsinh(I_app / j0_n_av / param.L_n / param.Ly)
    eta_p_av = -2 * np.arcsinh(I_app / j0_p_av / param.L_p / param.Ly)
    eta_r_av = eta_p_av - eta_n_av

    # average concentration overpotential
    c_e_n_av = np.mean(c_e_n) / param.L_n
    c_e_p_av = np.mean(c_e_p) / param.L_p
    eta_c_av = 2 * (1 - param.t_plus) * np.log(c_e_p_av / c_e_n_av)

    # average electrolyte ohmic losses
    Delta_Phi_elec_av = -(
        param.delta
        * param.nu
        * I_app
        / param.Lambda
        / param.Ly
        / param.electrolyte_conductivity(1)
    ) * (
        param.L_n / 3 / param.epsilon_n ** param.brug
        + param.L_s / param.epsilon_s ** param.brug
        + param.L_p / 3 / param.epsilon_p ** param.brug
    )

    # average solid phase ohmic losses
    Delta_Phi_solid_av = (
        -I_app / 3 / param.Ly * (param.L_p / param.sigma_p + param.L_n / param.sigma_n)
    )

    # Average through cell-voltage
    V_through_cell_av = (
        u_eq_av + eta_r_av + eta_c_av + Delta_Phi_elec_av + Delta_Phi_solid_av
    )

    # Average current collector ohmic losses
    Delta_Phi_CC = -param.delta * I_app * R_CC

    # Terminal voltagt
    V = V_through_cell_av + Delta_Phi_CC

    return voltage - param.V_min


def empty_particle(t, y, mesh):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(y, mesh)
    c_n_surf = c_n[-1] + (c_n[-1] - c_n[-2]) / 2
    c_p_surf = c_p[-1] + (c_p[-1] - c_p[-2]) / 2
    TOL = 1e-3
    return np.min([c_n_surf, c_p_surf]) - TOL


def full_particle(t, y, mesh):
    # Get variables
    c_n, c_p, c_e_n, c_e_s, c_e_p, T0, T1 = get_vars(y, mesh)
    c_n_surf = c_n[-1] + (c_n[-1] - c_n[-2]) / 2
    c_p_surf = c_p[-1] + (c_p[-1] - c_p[-2]) / 2
    TOL = 1e-3
    return 1.0 - (np.max([c_n_surf, c_p_surf]) + TOL)
