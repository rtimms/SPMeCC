# Script to compute effective resistance of current collectors for a user
# specified geometry. The problems for W and psi are solved using the finite
# element method, implemented using fenics. A unique solutions is ensured by
# adding an additional zero mean constraint to both problems (this is necessary
# for W, but any other means of obtaining a unique solution for psi is
# permitted, since the effective resistance only depends on the difference in
# psi between the tabs). The user must specify the geometry and the values of
# the electrical conductivities.
# DISCLAIMER: This is very rough and ready and could likely be written in
# a better way, but think it does the job. I haven't done any rigourous
# testing, but I have tried various geometries and messed about with grid
# points etc. and it seems ok.

import dolfin as df


def solve_psi_W(param, Ny, Nz, degree=1):
    """
    Returns the applied current as a function of time (and possibly
    model parameters).

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
    psi: fenics solution
        The weighted sum of current collector potentials.
    W: fenics solution
        The spatially dependent part of the local potential difference.
    R_CC: float
        The effective current collector resistance.
    R_cn: float
        The effective negative current collector resistance calculated for
        Ohmic heating.
    R_cp: float
        The effective positive current collector resistance calculated for
        Ohmic heating.
    """
    # Boundary conditions
    dpsidn_negativetab = df.Constant(param.L_cn / param.A_tab_n)
    dpsidn_positivetab = df.Constant(-param.L_cp / param.A_tab_p)

    dWdn_negativetab = df.Constant(param.Ly /
                                   (param.alpha_prime
                                    * param.sigma_cn_dblprime
                                    * param.A_tab_n))
    dWdn_positivetab = df.Constant(param.Ly /
                                   (param.alpha_prime
                                    * param.sigma_cp_dblprime
                                    * param.A_tab_p))

    # Create mesh
    mesh = df.RectangleMesh(df.Point(0, 0), df.Point(param.Ly, 1), Ny, Nz)

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

    # Define and solve problems for psi and W
    lagrange_element = df.FiniteElement("Lagrange", mesh.ufl_cell(), degree)
    real_element = df.FiniteElement("Real", mesh.ufl_cell(), 0)
    mixed_functionspace = df.FunctionSpace(mesh,
                                           lagrange_element * real_element)

    # Trial and test functions
    psi, c1 = df.TrialFunction(mixed_functionspace)
    psi_test, c2 = df.TestFunction(mixed_functionspace)

    W, c3 = df.TrialFunction(mixed_functionspace)
    W_test, c4 = df.TestFunction(mixed_functionspace)

    # Define variational form for psi and W problems
    a_psi = (df.inner(df.grad(psi), df.grad(psi_test))
             + c1 * psi_test + psi * c2) * df.dx
    L_psi = (dpsidn_negativetab * psi_test * ds(1)
             + dpsidn_positivetab * psi_test * ds(2))

    a_W = (df.inner(df.grad(W), df.grad(W_test))
           + c3 * W_test + W * c4) * df.dx
    L_W = (-W_test * df.dx
           + dWdn_negativetab * W_test * ds(1)
           + dWdn_positivetab * W_test * ds(2))

    # Solve psi and W problems
    psi = df.Function(mixed_functionspace)
    df.solve(a_psi == L_psi, psi)
    (psi, c_psi) = psi.split()  # Split psi and c from mixed solution

    W = df.Function(mixed_functionspace)
    df.solve(a_W == L_W, W)
    (W, c_W) = W.split()  # Split psi and c from mixed solution

    # Check avg W is zero
    if abs(df.assemble(W * df.dx)) > 1E-6:
        raise ValueError("Average of W outside of tolerance!")

    # Compute R_CC
    psi_tab_n = df.assemble(psi * ds(1)) / param.L_tab_n
    psi_tab_p = df.assemble(psi * ds(2)) / param.L_tab_p
    W_tab_n = df.assemble(W * ds(1)) / param.L_tab_n
    W_tab_p = df.assemble(W * ds(2)) / param.L_tab_p

    R_CC = (((param.alpha_prime / param.Ly)
            * (param.sigma_cn_dblprime * param.L_cn * W_tab_p
               + param.sigma_cp_dblprime * param.L_cp * W_tab_n)
            - (psi_tab_p - psi_tab_n))
            / (param.sigma_cn_dblprime * param.L_cn
               + param.sigma_cp_dblprime * param.L_cp))

    # Compute effective resitance for current collector heating R_cn and R_cn
    grad_phi_cn = ((df.grad(psi)
                   + (param.alpha_prime / param.Ly)
                   * param.sigma_cp_dblprime * param.L_cp
                   * df.grad(W))
                   / (param.sigma_cn_dblprime * param.L_cn
                      + param.sigma_cp_dblprime * param.L_cp))

    R_cn = ((param.sigma_cn_dblprime * param.L_cn / param.Ly)
            * df.assemble(df.inner(grad_phi_cn, grad_phi_cn) * df.dx))

    grad_phi_cp = ((df.grad(psi)
                    - (param.alpha_prime / param.Ly)
                    * param.sigma_cn_dblprime * param.L_cn
                    * df.grad(W))
                   / (param.sigma_cn_dblprime * param.L_cn
                      + param.sigma_cp_dblprime * param.L_cp))

    R_cp = ((param.sigma_cp_dblprime * param.L_cp / param.Ly)
            * df.assemble(df.inner(grad_phi_cp, grad_phi_cp) * df.dx))

    return psi, W, R_CC, R_cn, R_cp
