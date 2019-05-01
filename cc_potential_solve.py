# This is an alternative formulation which is easier to understand for the Fast-Pouch-Cell model. It also has the added benefit of not requiring an additional element for the average and therefore degrees of freedom of the solution correspond to the values on the nodes when using polynomials of order 1 as the basis functions.

# This formulation is derived by setting
# phi_c_n = I_app * R_c_n and phi_c_p - V = I_app * R_c_p
# in
# Nabla^2 phi_c_k == (+-) I_app / (L_ck * Ly * sigma_c_k * delta^2).
# and then solving for R_c_k.

# Note: I_app>0 => phi_c_p>=V => R_c_p>=0

# this should all collapse down onto the same equations if we rescale the R_c_ks by the rhs of their respective problems. The only difference is the tab position.

import dolfin as df
import numpy as np


def solve_cc_potentials(param, Ny, Nz, degree=1):

    # Create mesh
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(param.Ly, 1), Ny, Nz)

    # Create classes for defining tabs
    class NegativeTab(df.SubDomain):
        def inside(self, x, on_boundary):
            if param.tab_n_location == "t":
                return df.near(x[1], 1.0) and df.between(
                    x[0],
                    (
                        param.tab_n_centre - param.L_tab_n / 2,
                        param.tab_n_centre + param.L_tab_n / 2,
                    ),
                )
            elif param.tab_n_location == "b":
                return df.near(x[1], 0.0) and df.between(
                    x[0],
                    (
                        param.tab_n_centre - param.L_tab_n / 2,
                        param.tab_n_centre + param.L_tab_n / 2,
                    ),
                )
            elif param.tab_n_location == "l":
                return df.near(x[0], 0.0) and df.between(
                    x[1],
                    (
                        param.tab_n_centre - param.L_tab_n / 2,
                        param.tab_n_centre + param.L_tab_n / 2,
                    ),
                )
            elif param.tab_n_location == "r":
                return df.near(x[0], param.Ly) and df.between(
                    x[1],
                    (
                        param.tab_n_centre - param.L_tab_n / 2,
                        param.tab_n_centre + param.L_tab_n / 2,
                    ),
                )
            else:
                raise ValueError("Neg. tab location must be one of " "t, b, l, r!")

    class PositiveTab(df.SubDomain):
        def inside(self, x, on_boundary):
            if param.tab_p_location == "t":
                return df.near(x[1], 1.0) and df.between(
                    x[0],
                    (
                        param.tab_p_centre - param.L_tab_p / 2,
                        param.tab_p_centre + param.L_tab_p / 2,
                    ),
                )
            elif param.tab_p_location == "b":
                return df.near(x[1], 0.0) and df.between(
                    x[0],
                    (
                        param.tab_p_centre - param.L_tab_p / 2,
                        param.tab_p_centre + param.L_tab_p / 2,
                    ),
                )
            elif param.tab_p_location == "l":
                return df.near(x[0], 0.0) and df.between(
                    x[1],
                    (
                        param.tab_p_centre - param.L_tab_p / 2,
                        param.tab_p_centre + param.L_tab_p / 2,
                    ),
                )
            elif param.tab_p_location == "r":
                return df.near(x[0], param.Ly) and df.between(
                    x[1],
                    (
                        param.tab_p_centre - param.L_tab_p / 2,
                        param.tab_p_centre + param.L_tab_p / 2,
                    ),
                )
            else:
                raise ValueError("Pos. tab location must be one of " "t, b, l, r!")

    # Initialize sub-domain instances fot tabs
    # negative_tab = NegativeTab()
    # positive_tab = PositiveTab()

    # # Initialize mesh function for boundary domains
    # boundary_markers = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # boundary_markers.set_all(0)
    # negative_tab.mark(boundary_markers, 1)
    # positive_tab.mark(boundary_markers, 2)

    # Create measure of parts of the boundary
    # ds = df.Measure("ds", domain=mesh, subdomain_data=boundary_markers)

    # Define functions space to solve R_c_n and R_c_p on
    V = df.FunctionSpace(mesh, "CG", degree)

    # Trial functions
    R_c_n = df.TrialFunction(V)
    R_c_p = df.TrialFunction(V)

    # Test functions
    R_c_n_test = df.TestFunction(V)
    R_c_p_test = df.TestFunction(V)

    # boundary conditions
    bc_negative_tab = df.DirichletBC(V, df.Constant(0), NegativeTab())
    bc_positive_tab = df.DirichletBC(V, df.Constant(0), PositiveTab())

    # Weak form
    a_n = (df.inner(df.grad(R_c_n), df.grad(R_c_n_test))) * df.dx
    a_p = (df.inner(df.grad(R_c_p), df.grad(R_c_p_test))) * df.dx

    L_n = (
        df.Constant(1 / (param.L_cn * param.Ly * param.sigma_cn * param.epsilon ** 2))
        * R_c_n_test
        * df.dx
    )
    L_p = (
        df.Constant(1 / (param.L_cp * param.Ly * param.sigma_cp * param.epsilon ** 2))
        * R_c_p_test
        * df.dx
    )

    # Create output variables
    R_c_n = df.Function(V)
    R_c_p = df.Function(V)

    # Solve negative and positive problems separately
    df.solve(a_n == L_n, R_c_n, bc_negative_tab)
    df.solve(a_p == L_p, R_c_p, bc_positive_tab)

    # Compute R_CC
    # note assume uniform mesh here
    R_c_n_av = np.mean(R_c_n.vector()[:])
    R_c_p_av = np.mean(R_c_p.vector()[:])

    R_CC = R_c_n_av + R_c_p_av

    # try create a list of the nodes

    return R_cn, R_cp, R_CC
