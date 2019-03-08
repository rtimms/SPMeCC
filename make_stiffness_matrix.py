import dolfin as df
import numpy as np


def fem_matrices(param, Ny, Nz, degree=1):
    """
    Returns the stiffness matrix and contribution to the load vector due to
    boundary conditions.

    Parameters
    ----------
    param: object
        Object containing model parameters.
    Ny: int
        Number of mesh points in the y direction.
    Nz: int
        Number of mesh points in the z direction.
    degree: int
        Degree of polynomial used in FEM.
    Returns
    -------
    K: array_like
        Stiffness matrix.
    load_tab_n: array_like
        Contribution to the load vector due to the boundary condition at
        the negative tab.
    load_tab_p: array_like
        Contribution to the load vector due to the boundary condition at
        the positive tab.
    """
    # Optimization options for the form compiler
    df.parameters["form_compiler"]["cpp_optimize"] = True
    df.ffc_options = {
        "optimize": True,
        "eliminate_zeros": True,
        "precompute_basis_const": True,
        "precompute_ip_const": True,
        "quadrature_degree": 2,
    }

    # Create mesh and function space
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(param.Ly, 1), Ny, Nz)
    element = df.FunctionSpace(mesh, "Lagrange", degree)

    V = df.TrialFunction(element)
    V_test = df.TestFunction(element)

    # Create classes for defining tabs
    class NegativeTab(df.SubDomain):
        def inside(self, x, on_boundary):
            if param.tab_n_location == 't':
                return df.near(x[1], 1.0) and \
                    df.between(x[0], (param.tab_n_centre - param.L_tab_n / 2,
                                      param.tab_n_centre + param.L_tab_n / 2)
                               )
            elif param.tab_n_location == 'b':
                return df.near(x[1], 0.0) and \
                    df.between(x[0], (param.tab_n_centre - param.L_tab_n / 2,
                                      param.tab_n_centre + param.L_tab_n / 2)
                               )
            elif param.tab_n_location == 'l':
                return df.near(x[0], 0.0) and \
                    df.between(x[1], (param.tab_n_centre - param.L_tab_n / 2,
                                      param.tab_n_centre + param.L_tab_n / 2)
                               )
            elif param.tab_n_location == 'r':
                return df.near(x[0], param.Ly) and \
                    df.between(x[1], (param.tab_n_centre - param.L_tab_n / 2,
                                      param.tab_n_centre + param.L_tab_n / 2)
                               )
            else:
                raise ValueError("Neg. tab location must be one of "
                                 "t, b, l, r!")

    class PositiveTab(df.SubDomain):
        def inside(self, x, on_boundary):
            if param.tab_p_location == 't':
                return df.near(x[1], 1.0) and \
                    df.between(x[0], (param.tab_p_centre - param.L_tab_p / 2,
                                      param.tab_p_centre + param.L_tab_p / 2)
                               )
            elif param.tab_p_location == 'b':
                return df.near(x[1], 0.0) and \
                    df.between(x[0], (param.tab_p_centre - param.L_tab_p / 2,
                                      param.tab_p_centre + param.L_tab_p / 2)
                               )
            elif param.tab_p_location == 'l':
                return df.near(x[0], 0.0) and \
                    df.between(x[1], (param.tab_p_centre - param.L_tab_p / 2,
                                      param.tab_p_centre + param.L_tab_p / 2)
                               )
            elif param.tab_p_location == 'r':
                return df.near(x[0], param.Ly) and \
                    df.between(x[1], (param.tab_p_centre - param.L_tab_p / 2,
                                      param.tab_p_centre + param.L_tab_p / 2)
                               )
            else:
                raise ValueError("Pos. tab location must be one of "
                                 "t, b, l, r!")

    # Initialize sub-domain instances fot tabs
    negativetab = NegativeTab()
    positivetab = PositiveTab()

    # Initialize mesh function for boundary domains
    boundary_markers = df.MeshFunction("size_t", mesh,
                                       mesh.topology().dim() - 1)
    boundary_markers.set_all(0)
    negativetab.mark(boundary_markers, 1)
    positivetab.mark(boundary_markers, 2)

    # Create measure of parts of the boundary
    ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)

    # Create load vectors for tabs
    dVdn_negativetab = df.Constant(-param.C_rate/(param.sigma_cn_prime*param.A_tab_n))  # dV/dn at -ve tab
    dVdn_positivetab = df.Constant(-param.C_rate/(param.sigma_cp_prime*param.A_tab_p))  # dV/dn at +ve tab
    neg_tab_form = dVdn_negativetab * V_test * ds(1)
    pos_tab_form = dVdn_positivetab * V_test * ds(2)
    load_tab_n = df.assemble(neg_tab_form).get_local()[:]
    load_tab_p = df.assemble(pos_tab_form).get_local()[:]

    # Create stifnness matrix
    K_form = df.inner(df.grad(V), df.grad(V_test)) * df.dx
    K = df.assemble(K_form).array()

    # Number of degrees of freedom
    V = df.Function(element)
    N_dofs = np.size(V.vector()[:])

    return K, load_tab_n, load_tab_p, N_dofs
