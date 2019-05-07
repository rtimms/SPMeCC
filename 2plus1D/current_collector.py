import dolfin as df
import numpy as np


class CCVoltage(object):

    def __init__(self, param, Ny=32, Nz=32, degree=1):
        """
        Sets up the mesh, function space etc. for the voltage problem in the
        current collectors.

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
        """
        self.Ny = Ny
        self.Nz = Nz
        self.degree = degree
        self.alpha = param.alpha

        # Create mesh and function space
        self.mesh = df.RectangleMesh(df.Point(0, 0), df.Point(param.Ly, 1), self.Ny, self.Nz)
        self.element = df.FunctionSpace(self.mesh, "Lagrange", self.degree)

        self.TrialFunction = df.TrialFunction(self.element)
        self.TestFunction = df.TestFunction(self.element)

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
        boundary_markers = df.MeshFunction("size_t", self.mesh,
                                           self.mesh.topology().dim() - 1)
        boundary_markers.set_all(0)
        negativetab.mark(boundary_markers, 1)
        positivetab.mark(boundary_markers, 2)

        # Create measure of parts of the boundary
        self.ds = df.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        # Boundary values
        self.dVdn_negativetab = df.Constant(-param.C_rate/(param.sigma_cn_prime*param.A_tab_n))  # dV/dn at -ve tab
        self.dVdn_positivetab = df.Constant(-param.C_rate/(param.sigma_cp_prime*param.A_tab_p))  # dV/dn at +ve tab

    def assemble(self):
        " Assemble mass and stiffness matrices, and boundary load vector."
        # Create mass matrix
        M_form = self.TrialFunction * self.TestFunction * df.dx
        self.mass = df.assemble(M_form).array()

        # Create stifnness matrix
        K_form = df.inner(df.grad(self.TrialFunction), df.grad(self.TestFunction)) * df.dx
        self.stiffness = df.assemble(K_form).array()

        # Create load vectors for tabs
        neg_tab_form = self.dVdn_negativetab * self.TestFunction * self.ds(1)
        pos_tab_form = self.dVdn_positivetab * self.TestFunction * self.ds(2)
        self.load_tab_n = df.assemble(neg_tab_form).get_local()[:]
        self.load_tab_p = df.assemble(pos_tab_form).get_local()[:]

        # Set functions for V and I
        self.voltage = df.Function(self.element)
        self.current = df.Function(self.element)

        # Number of degrees of freedom
        self.N_dofs = np.size(self.voltage.vector()[:])

        # Placeholder for voltage difference
        self.voltage_difference = 1

    def update_current_values(self, current):
        "Update the entries of the through-cell current density."
        self.current.vector()[:] = current

    def solve(self):
        "Solve the linear system K*V = b"
        # Right hand side
        b = (self.load_tab_n + self.load_tab_p
             + np.dot(self.mass, self.alpha * self.current.vector()))

        # Store old values for error computation
        voltage_prev = self.voltage.vector()[:]

        # Solve
        df.solve(self.stiffness, self.voltage.vector(), b)

        # Update difference in solution
        self.voltage_difference = np.linalg.norm(voltage_prev - self.voltage.vector()[:])
