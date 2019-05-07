from make_parameters import Parameters as myparams
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Optimization options for the form compiler ----------------------------------
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True,
    "quadrature_degree": 2,
}


# Visualisation ---------------------------------------------------------------
timeplots = False  # If true plots at each time are displayed
file_V = File("output/output_1C/V.pvd", "compressed")  # File to save output to
file_I = File("output/output_1C/I.pvd", "compressed")  # File to save output to
file_cn = File("output/output_1C/cn.pvd", "compressed")  # File to save output to
file_cp = File("output/output_1C/cp.pvd", "compressed")  # File to save output to
file_T = File("output/output_1C/T.pvd", "compressed")  # File to save output to


# Load parameters -------------------------------------------------------------
C_rate = 1.0
# param = Parameters(C_rate)
param = myparams(C_rate, "mypouch")

# Initial and boundary conditions ---------------------------------------------
I_app = 1.0  # Applied current (TO DO: make function of time)

c_n0 = param.c_n_0  # Initial (uniform) negative electrode surface concentration
c_p0 = param.c_p_0  # Initial (uniform) positive electrode surface concentration
T_initial = param.T_0  # Initial (uniform) temperature

dpsidn_negativetab = Constant(param.L_cn / param.A_tab_n)  # dpsi/dn at -ve tab
dpsidn_positivetab = Constant(-param.L_cp / param.A_tab_p)  # dpsi/dn at +ve tab
dVdn_negativetab = Constant(
    -I_app / (param.sigma_cn_prime * param.A_tab_n)
)  # dV/dn at -ve tab
dVdn_positivetab = Constant(
    -I_app / (param.sigma_cp_prime * param.A_tab_p)
)  # dV/dn at +ve tab


# Timestepping ----------------------------------------------------------------
t = 0.0  # initial time
t_final = (3600) / param.tau_d_star  # final time
dt = 15 / param.tau_d_star  # step size


# Open circuit potentials -----------------------------------------------------
def mytanh(x):
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x))


def U_n(c, T, param):
    U_ref = (
        0.194
        + 1.5 * exp(-120.0 * c)
        + 0.0351 * mytanh((c - 0.286) / 0.083)
        - 0.0045 * mytanh((c - 0.849) / 0.119)
        - 0.035 * mytanh((c - 0.9233) / 0.05)
        - 0.0147 * mytanh((c - 0.5) / 0.034)
        - 0.102 * mytanh((c - 0.194) / 0.142)
        - 0.022 * mytanh((c - 0.9) / 0.0164)
        - 0.011 * mytanh((c - 0.124) / 0.0226)
        + 0.0155 * mytanh((c - 0.105) / 0.029)
    )
    return (U_ref / param.Phi_star) + T * dUdT_n(c, param)


def U_p(c, T, param):
    stretch = 1.062
    sto = stretch * c
    U_ref = (
        2.16216
        + 0.07645 * mytanh(30.834 - 54.4806 * sto)
        + 2.1581 * mytanh(52.294 - 50.294 * sto)
        - 0.14169 * mytanh(11.0923 - 19.8543 * sto)
        + 0.2051 * mytanh(1.4684 - 5.4888 * sto)
        + 0.2531 * mytanh((-sto + 0.56478) / 0.1316)
        - 0.02167 * mytanh((sto - 0.525) / 0.006)
    )
    return (U_ref / param.Phi_star) + T * dUdT_n(c, param)


def dUdT_n(c, param):
    result = 0.001 * (
        0.005269056
        + 3.299265709 * c
        - 91.79325798 * c ** 2
        + 1004.911008 * c ** 3
        - 5812.278127 * c ** 4
        + 19329.7549 * c ** 5
        - 37147.8947 * c ** 6
        + 38379.18127 * c ** 7
        - 16515.05308 * c ** 8
    )
    result = result / (
        1
        - 48.09287227 * c
        + 1017.234804 * c ** 2
        - 10481.80419 * c ** 3
        + 59431.3 * c ** 4
        - 195881.6488 * c ** 5
        + 374577.3152 * c ** 6
        - 385821.1607 * c ** 7
        + 165705.8597 * c ** 8
    )
    return result * (param.Delta_T_star / param.Phi_star)


def dUdT_p(c, param):
    result = -0.001 * (
        0.199521039
        - 0.928373822 * c
        + 1.364550689000003 * c ** 2
        - 0.6115448939999998 * c ** 3
    )
    result = result / (
        1
        - 5.661479886999997 * c
        + 11.47636191 * c ** 2
        - 9.82431213599998 * c ** 3
        + 3.048755063 * c ** 4
    )
    return result * (param.Delta_T_star / param.Phi_star)


# Butler-Volmer Coefficient fucntion ------------------------------------------
def g_n(c, param):
    return param.m_n * param.C_hat_n * sqrt(c) * sqrt(1 - c)


def g_p(c, param):
    return param.m_p * param.C_hat_p * sqrt(c) * sqrt(1 - c)


# Reaction overpotentials ----------------------------------------------------
def eta_n(I, c, T, param):
    result = (
        2
        * ((1 + param.Theta * T) / param.Lambda)
        * (
            ln(
                I / (g_n(c, param) * param.L_n)
                + sqrt(1 + (I / (g_n(c, param) * param.L_n)) ** 2)
            )
        )
    )
    return result


def eta_p(I, c, T, param):
    result = -(
        2
        * ((1 + param.Theta * T) / param.Lambda)
        * (
            ln(
                I / (g_p(c, param) * param.L_p)
                + sqrt(1 + (I / (g_p(c, param) * param.L_p)) ** 2)
            )
        )
    )
    return result


# x-averaged heat source term -------------------------------------------------
def Q_bar(psi, V, I, c_n, c_p, T, param):
    phi_cn = (I_app * psi - param.sigma_cp_prime * param.L_cp * V) / (
        param.sigma_cn_prime * param.L_cn + param.sigma_cp_prime * param.L_cp
    )
    phi_cp = (I_app * psi + param.sigma_cn_prime * param.L_cn * V) / (
        param.sigma_cn_prime * param.L_cn + param.sigma_cp_prime * param.L_cp
    )

    Q_cn = param.L_cn * param.sigma_cn_prime * inner(grad(phi_cn), grad(phi_cn))
    Q_cp = param.L_cp * param.sigma_cp_prime * inner(grad(phi_cp), grad(phi_cp))
    Q_rxn = -I * (V - (U_p(c_p, T, param) - U_n(c_n, T, param)))
    Q_rev = -I * (1 / param.Theta + T) * (dUdT_p(c_p, param) - dUdT_n(c_n, param))
    return (Q_cn + Q_cp + Q_rxn + Q_rev) / param.L


# Meshing ---------------------------------------------------------------------
# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(param.Ly, 1), 32, 32)


# Create classes for defining tabs
class NegativeTab(SubDomain):
    def inside(self, x, on_boundary):
        if param.tab_n_location == "t":
            return near(x[1], 1.0) and between(
                x[0],
                (
                    param.tab_n_centre - param.L_tab_n / 2,
                    param.tab_n_centre + param.L_tab_n / 2,
                ),
            )
        elif param.tab_n_location == "b":
            return near(x[1], 0.0) and between(
                x[0],
                (
                    param.tab_n_centre - param.L_tab_n / 2,
                    param.tab_n_centre + param.L_tab_n / 2,
                ),
            )
        elif param.tab_n_location == "l":
            return near(x[0], 0.0) and between(
                x[1],
                (
                    param.tab_n_centre - param.L_tab_n / 2,
                    param.tab_n_centre + param.L_tab_n / 2,
                ),
            )
        elif param.tab_n_location == "r":
            return near(x[0], param.Ly) and between(
                x[1],
                (
                    param.tab_n_centre - param.L_tab_n / 2,
                    param.tab_n_centre + param.L_tab_n / 2,
                ),
            )
        else:
            raise ValueError("Neg. tab location must be one of " "t, b, l, r!")


class PositiveTab(SubDomain):
    def inside(self, x, on_boundary):
        if param.tab_p_location == "t":
            return near(x[1], 1.0) and between(
                x[0],
                (
                    param.tab_p_centre - param.L_tab_p / 2,
                    param.tab_p_centre + param.L_tab_p / 2,
                ),
            )
        elif param.tab_p_location == "b":
            return near(x[1], 0.0) and between(
                x[0],
                (
                    param.tab_p_centre - param.L_tab_p / 2,
                    param.tab_p_centre + param.L_tab_p / 2,
                ),
            )
        elif param.tab_p_location == "l":
            return near(x[0], 0.0) and between(
                x[1],
                (
                    param.tab_p_centre - param.L_tab_p / 2,
                    param.tab_p_centre + param.L_tab_p / 2,
                ),
            )
        elif param.tab_p_location == "r":
            return near(x[0], param.Ly) and between(
                x[1],
                (
                    param.tab_p_centre - param.L_tab_p / 2,
                    param.tab_p_centre + param.L_tab_p / 2,
                ),
            )
        else:
            raise ValueError("Pos. tab location must be one of " "t, b, l, r!")


# Initialize sub-domain instances fot tabs
negativetab = NegativeTab()
positivetab = PositiveTab()


# Initialize mesh function for boundary domains
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0)
negativetab.mark(boundary_markers, 1)
positivetab.mark(boundary_markers, 2)

# Create measure of parts of the boundary
ds = Measure("ds", domain=mesh, subdomain_data=boundary_markers)


# Define and solve problem for psi --------------------------------------------
# Define function space and basis functions for psi. Need c1, c2 to impose
# addtional constraint since we have pure Neumann BCs. Can reconstruct phi_cn
# and phi_cp later using solution for V and then add constant to fix
# phi = 0 somewhere
psi_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
real_element = FiniteElement("Real", mesh.ufl_cell(), 0)
psi_functionspace = FunctionSpace(mesh, psi_element * real_element)

# Trial and test functions
psi, c1 = TrialFunction(psi_functionspace)
psi_test, c2 = TestFunction(psi_functionspace)

# Define variational form for psi problem
a = (inner(grad(psi), grad(psi_test)) + c1 * psi_test + psi * c2) * dx
L = dpsidn_negativetab * psi_test * ds(1) + dpsidn_positivetab * psi_test * ds(2)

# Solve psi problem
psi = Function(psi_functionspace)
solve(a == L, psi)
(psi, c) = psi.split()  # Split psi and c from mixed solution


# Define and solve time dependent problem for u = [V, I, c_n, c_p, T] ---------
# Function space and mixed elements
u_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
u_mixedelement = MixedElement([u_element, u_element, u_element, u_element, u_element])
u_functionspace = FunctionSpace(mesh, u_mixedelement)

# Define test functions
V_test, I_test, c_n_test, c_p_test, T_test = TestFunctions(u_functionspace)

# Define functions u (current solution) and u_ (previous solution)
u = Function(u_functionspace)
u_prev = Function(u_functionspace)

# Split system functions to access components
V, I, c_n, c_p, T = split(u)
V_prev, I_prev, c_n_prev, c_p_prev, T_prev = split(u_prev)


# Class representing the intial conditions
class InitialConditions(UserExpression):
    def eval(self, values, x):
        values[0] = U_p(c_p0, T_initial, param) - U_n(c_n0, T_initial, param)
        values[1] = 0.0
        values[2] = c_n0
        values[3] = c_p0
        values[4] = T_initial

    def value_shape(self):
        return (5,)


# Create initial conditions and interpolate
u_init = InitialConditions(degree=0)
u.interpolate(u_init)
u_prev.interpolate(u_init)


# Step in time
counter = 0
while t < t_final:

    # Increase time
    t += dt
    print("t = {:.0f} seconds".format(t * param.tau_d_star))

    # Write down weak form F == 0
    F1 = (
        (inner(grad(V), grad(V_test)) - param.alpha * I * V_test) * dx
        - dVdn_negativetab * V_test * ds(1)
        - dVdn_positivetab * V_test * ds(2)
    )
    F2 = (
        (
            V
            - (U_p(c_p, T, param) - U_n(c_n, T, param))
            - (eta_p(I, c_p, T, param) - eta_n(I, c_n, T, param))
        )
        * I_test
        * dx
    )
    F3 = (
        (c_n - c_n_prev + I * 3 * dt / (param.beta_n * param.C_hat_n * param.L_n))
        * c_n_test
        * dx
    )
    F4 = (
        (c_p - c_p_prev - I * 3 * dt / (param.beta_p * param.C_hat_p * param.L_p))
        * c_p_test
        * dx
    )
    F5 = (
        (param.rho / param.gamma_th) * (T - T_prev) * T_test * dx
        + dt * param.lambda_x * inner(grad(T), grad(T_test)) * dx
        - dt * param.B * Q_bar(psi, V, I, c_n, c_p, T, param) * T_test * dx
        + dt * (2 * param.h_prime / param.L) * T * T_test * dx
        + dt * param.epsilon * param.h_prime * T * T_test * ds(0)
        + dt
        * (param.epsilon / param.L)
        * (
            (param.h_tab_prime * (param.L_cn + param.L_cp))
            + param.h_prime
            - param.h_prime * param.L
        )
        * T
        * T_test
        * ds(1)
        + dt
        * (param.epsilon / param.L)
        * (
            (param.h_tab_prime * (param.L_cn + param.L_cp))
            + param.h_prime
            - param.h_prime * param.L
        )
        * T
        * T_test
        * ds(2)
    )
    F = F1 + F2 + F3 + F4 + F5

    # Jacobian
    J = derivative(F, u)

    # Nonlinear solve
    solve(
        F == 0,
        u,
        J=J,
        solver_parameters={"newton_solver": {"relative_tolerance": 1e-6}},
        form_compiler_parameters=ffc_options,
    )

    # Assign u to previous solution
    u_prev.assign(u)
    (V_split, I_split, c_n_split, c_p_split, T_split) = u.split()

    # Save to file
    file_V << (V_split, t)
    file_I << (I_split, t)
    file_cn << (c_n_split, t)
    file_cp << (c_p_split, t)
    file_T << (T_split, t)

    # Print information about conservation of current
    I_dx = assemble(I_split * dx)  # integrate I over the domain
    I_app_num = (
        I_app
        * assemble(
            (1 / (param.sigma_cn_prime * param.A_tab_n)) * ds(1)
            + (1 / (param.sigma_cp_prime * param.A_tab_p)) * ds(2)
        )
        / param.alpha
    )
    I_error = np.linalg.norm(I_app_num - I_dx)
    print(
        "Check current conservation: I_app = {:.2f}, I_app_num = {:.2f}, "
        "I*dx = {:.2f}, "
        "error = {:.2E}".format(I_app, I_app_num, I_dx, I_error)
    )
    if abs(I_error) > 1e-8:
        print("Total current not conserved! Check tabs and gridsize!")
        break

    # Plotting
    counter += 1
    if timeplots == True and counter % 5 == 1:
        plt.figure(1, figsize=(15, 9))
        plt.subplot(2, 3, 1)
        p1 = plot(V_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("V")
        plt.colorbar(p1)
        plt.subplot(2, 3, 2)
        p2 = plot(I_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("I")
        plt.colorbar(p2)
        plt.subplot(2, 3, 3)
        p3 = plot(param.T_0_star + param.Delta_T_star * T_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("T /K")
        plt.colorbar(p3)
        plt.subplot(2, 3, 4)
        p4 = plot(c_n_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("c_n")
        plt.colorbar(p4)
        plt.subplot(2, 3, 5)
        p5 = plot(c_p_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("c_p")
        plt.colorbar(p5)
        plt.subplot(2, 3, 6)
        p6 = plot(Q_bar(psi, V_split, I_split, c_n_split, c_p_split, T_split, param))
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("Q")
        plt.colorbar(p6)
        plt.suptitle(
            "Solution at t = {:.0f} minutes "
            "(alpha = {:.2f})".format(t * param.tau_d_star / 60, param.alpha)
        )
        plt.show()

    # Break if cutoff voltage exceeded
    V_check = V_split.compute_vertex_values(mesh)
    if min(V_check) < param.V_min or max(V_check) > param.V_max:
        print("Voltage cutoff! Simulation stopping!")
        break


# Plot solution ---------------------------------------------------------------
# plt.figure(1, figsize=(15, 9))
# plt.subplot(2, 3, 1)
# p1 = plot(V_split)
# plt.xlabel('y')
# plt.ylabel('z')
# plt.title("V")
# plt.colorbar(p1)
# plt.subplot(2, 3, 2)
# p2 = plot(I_split)
# plt.xlabel('y')
# plt.ylabel('z')
# plt.title("I")
# plt.colorbar(p2)
# plt.subplot(2, 3, 3)
# p3 = plot(param.T_0_star + param.Delta_T_star*T_split)
# plt.xlabel('y')
# plt.ylabel('z')
# plt.title("T /K")
# plt.colorbar(p3)
# plt.subplot(2, 3, 4)
# p4 = plot(c_n_split)
# plt.xlabel('y')
# plt.ylabel('z')
# plt.title("c_n")
# plt.colorbar(p4)
# plt.subplot(2, 3, 5)
# p5 = plot(c_p_split)
# plt.xlabel('y')
# plt.ylabel('z')
# plt.title("c_p")
# plt.colorbar(p5)
# plt.suptitle('Solution at t = {:.0f} minutes '
#             '(alpha = {:.2f})'.format(t*param.tau_d_star/60, param.alpha))
# plt.show()


# Font stuff
plt.rc("text", usetex=True)
plt.rc("font", family="sans-serif")
plt.rc("mathtext", fontset="stixsans")
plt.rc("text.latex", preamble=r"\usepackage{sfmath}")
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=18)
plt.rc("axes", titlepad=10)

# Make plots
# fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
# ax = plt.gca()
# p1 = plot(V_split)
# p1.set_cmap("viridis")
# plt.xlabel(r"$y$", fontsize=22)
# plt.ylabel(r"$z$", fontsize=22)
# plt.title(r"\textbf{Potential (V)}", fontsize=24)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(p1, cax=cax)
# fig.tight_layout()
# plt.savefig("V_2D.eps", format="eps", dpi=1000, bbox_inches="tight")

fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
ax = plt.gca()
p2 = plot(param.T_0_star + param.Delta_T_star * T_split, vmin=301, vmax=302.5)
p2.set_cmap("plasma")
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"$T$ /K", fontsize=24)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks = [301, 301.25, 301.5, 301.75, 302, 302.25, 302.5]
cbar = plt.colorbar(p2, cax=cax, extend='both')
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(ticks)
fig.tight_layout()
plt.savefig("T.eps", format="eps", dpi=1000, bbox_inches="tight")

fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
ax = plt.gca()
p3 = plot(I_split, vmin=1.05, vmax=1.4)
p3.set_cmap("viridis")
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"$\mathcal{I}$", fontsize=24)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks = [1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
cbar = plt.colorbar(p3, cax=cax, extend='both')
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(ticks)
fig.tight_layout()
plt.savefig("I.eps", format="eps", dpi=1000, bbox_inches="tight")

fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
ax = plt.gca()
p4 = plot(c_n_split, vmin=0.6205, vmax=0.6705)
p4.set_cmap("BuGn")
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"$c_n$", fontsize=24)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks = [0.62, 0.63, 0.64, 0.65, 0.66, 0.67]
cbar = plt.colorbar(p4, cax=cax)
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(ticks)
fig.tight_layout()
plt.savefig("cn.eps", format="eps", dpi=1000, bbox_inches="tight")

fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
ax = plt.gca()
p5 = plot(c_p_split, vmin=0.87, vmax=0.9)
p5.set_cmap("RdPu")
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"$c_p$", fontsize=24)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
ticks = [0.87, 0.88, 0.89, 0.90]
cbar = plt.colorbar(p5, cax=cax, extend='both')
cbar.set_ticks(ticks)
cbar.ax.set_yticklabels(ticks)
fig.tight_layout()
plt.savefig("cp.eps", format="eps", dpi=1000, bbox_inches="tight")

plt.show()
