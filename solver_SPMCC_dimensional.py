from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal


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
timeplots = True  # If true plots at each time are displayed
file_V = File("output_dim/V.pvd", "compressed")  # File to save output to
file_I = File("output_dim/I.pvd", "compressed")  # File to save output to
file_cn = File("output_dim/cn.pvd", "compressed")  # File to save output to
file_cp = File("output_dim/cp.pvd", "compressed")  # File to save output to
file_T = File("output_dim/T.pvd", "compressed")  # File to save output to


# Parameters -------------------------------------------------------------
C_rate = 1.0
capacity = 20
I_app_1C = 20
I_app_period = 50

# Geometry
L_cn = 0.25 * 1e-3
L_n = 2.7 * 1e-3
L_s = 0.0
L_p = 2.7 * 1e-3
L_cp = 0.25 * 1e-3

L_x = L_cn + L_n + L_p + L_cp
L_y = 150 * 1e-3
L_z = 200 * 1e-3

L = L_cn + L_n + L_s + L_p + L_cp

tab_offset = 10 * 1e-3
tab_width = 48 * 1e-3

A_tab_n = tab_width * L_cn
A_tab_p = tab_width * L_cp

# Conductivty
sigma_cn = 5.96 * 1e7
sigma_cp = 3.55 * 1e7

# Particle radius
R_n = 2 * 1e-6
R_p = 2 * 1e-6

# Particle surface area density
a_n = 723600
a_p = 885000

# Constants
F = 96487
R = 8.314
T_infty = 298.15

# Electrochemical
k_U = 0.36
Delta_S = 6.8
n = 1
j_0_ref = 14.31
E_j0 = 31.79*1E3

# Concentration
c_n_max = 31507.
c_p_max = 21190.
c_n_SOC_min = 0.
c_n_SOC_max = 0.98
c_p_SOC_min = 0.01
c_p_SOC_max = 0.9

# Thermal
rho_eff = 2.32 * 1e6
lambda_eff = 1
h = 12.5
h_tab = 50

# Derived
alpha = 1 / (sigma_cp * L_cp) + 1 / (sigma_cn * L_cn)

# Initial conditions
SOC_init = 0.5
OCV_init = 3.25
c_n_init = (SOC_init * (c_n_SOC_max - c_n_SOC_min) + c_n_SOC_min) * c_n_max
c_p_init = (SOC_init * (c_p_SOC_max - c_p_SOC_min) + c_p_SOC_min) * c_p_max
T_init = T_infty


# Applied current
def I_app(t):
    return C_rate * I_app_1C * signal.square(2*np.pi*t / I_app_period)


# Timestepping ----------------------------------------------------------------
t = 0.0  # initial time
t_final = 3600  # final time
dt = 1  # step size


# Exchange current density ----------------------------------------------------
def j_0_n(c, T):
    Arrhenius = exp((-E_j0 / R)*(1/T - 1/T_infty))
    return j_0_ref * Arrhenius * sqrt(c/c_n_max) * sqrt((c_n_max - c)/c_n_max)


def j_0_p(c, T):
    Arrhenius = exp((-E_j0 / R)*(1/T - 1/T_infty))
    return j_0_ref * Arrhenius * sqrt(c/c_p_max) * sqrt((c_p_max - c)/c_p_max)


# Reaction overpotentials ----------------------------------------------------
def eta_n(I, c, T):
    result = (2 * R * T / F) * (
        ln(
            I / (a_n * L_n * j_0_n(c, T))
            + sqrt(1 + (I / (a_n * L_n * j_0_n(c, T)) ** 2))
        )
    )
    return result


def eta_p(I, c, T):
    result = (2 * R * T / F) * (
        ln(
            I / (a_p * L_p * j_0_p(c, T))
            + sqrt(1 + (I / (a_p * L_p * j_0_p(c, T)) ** 2))
        )
    )
    return result


# Open circuit potentials -----------------------------------------------------
def OCV(c_p, T):
    return OCV_init + k_U * SOC(c_p)


def SOC(c_p):
    return (c_p / c_p_max - c_p_SOC_min) / (c_p_SOC_max - c_p_SOC_min)


# x-averaged heat source term -------------------------------------------------
def Q_bar(psi, V, I, c_n, c_p, T):
    phi_cn = (I_app * psi - sigma_cp * L_cp * V) / (
        sigma_cn * L_cn + sigma_cp * L_cp
    )
    phi_cp = (I_app * psi + sigma_cn * L_cn * V) / (
        sigma_cn * L_cn + sigma_cp * L_cp
    )

    Q_cn = L_cn * sigma_cn * inner(grad(phi_cn), grad(phi_cn))
    Q_cp = L_cp * sigma_cp * inner(grad(phi_cp), grad(phi_cp))
    Q_rxn = -I * (eta_p(I, c_p, T) - eta_n(I, c_n, T))
    Q_rev = I * T * Delta_S / n / F
    return (Q_cn + Q_cp + Q_rxn + Q_rev) / L


# Meshing ---------------------------------------------------------------------
# Create mesh
mesh = RectangleMesh(Point(0, 0), Point(L_y, L_z), 64, 64)


# Create classes for defining tabs
class NegativeTab(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], L_z) and between(x[0], (tab_offset, tab_offset + tab_width))


class PositiveTab(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], L_z) and between(
            x[0], (L_y - tab_offset - tab_width, L_y - tab_offset)
        )


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
dpsidn_negativetab = Constant(L_cn / A_tab_n)
dpsidn_positivetab = Constant(-L_cp / A_tab_p)
a_psi = (inner(grad(psi), grad(psi_test)) + c1 * psi_test + psi * c2) * dx
L_psi = dpsidn_negativetab * psi_test * ds(1) + dpsidn_positivetab * psi_test * ds(2)

# Solve psi problem
psi = Function(psi_functionspace)
solve(a_psi == L_psi, psi)
(psi, c) = psi.split()  # Split psi and c from mixed solution
p = plot(psi)
plt.xlabel("y")
plt.ylabel("z")
plt.title("psi")
plt.colorbar(p)
plt.show()


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
        values[0] = OCV_init
        values[1] = 0.0
        values[2] = c_n_init
        values[3] = c_p_init
        values[4] = T_init

    def value_shape(self):
        return (5,)


# Create initial conditions and interpolate
u_init = InitialConditions(degree=0)
u.interpolate(u_init)
u_prev.interpolate(u_init)

# Boundary conditions
dVdn_negativetab = Constant(-1 / (sigma_cn * A_tab_n))
dVdn_positivetab = Constant(-1 / (sigma_cp * A_tab_p))

# Step in time
counter = 0
while t < t_final:

    # Increase time
    t += dt
    print("t = {:.0f} seconds".format(t))
    I_app = Constant(I_app(t))

    # Write down weak form F == 0
    F1 = (
        (inner(grad(V), grad(V_test)) - alpha * I * V_test) * dx
        - I_app * dVdn_negativetab * V_test * ds(1)
        - I_app * dVdn_positivetab * V_test * ds(2)
    )
    F2 = (
        (
            V
            - (OCV(c_p, T))
            - (eta_p(I, c_p, T) - eta_n(I, c_n, T))
        )
        * I_test
        * dx
    )
    F3 = (
        (c_n - c_n_prev + I * 3 * dt / (R_n * a_n * L_n * F))
        * c_n_test
        * dx
    )
    F4 = (
        (c_p - c_p_prev - I * 3 * dt / (R_p * a_p * L_p * F))
        * c_p_test
        * dx
    )
    F5 = (
        rho_eff * (T - T_prev) * T_test * dx
        + dt * lambda_eff * inner(grad(T), grad(T_test)) * dx
        - dt * Q_bar(psi, V, I, c_n, c_p, T) * T_test * dx
        + dt * (2 * h / L) * (T - T_infty) * T_test * dx
        + dt * (2 * h * (L_y + L_z)) * (T - T_infty) * T_test * ds(0)
        + dt * (((h_tab * (L_cn+L_cp) + h * (L_n+L_s+L_p))/L - h) * tab_width) * (T - T_infty) * T_test * ds(1)
        + dt * (((h_tab * (L_cn+L_cp) + h * (L_n+L_s+L_p))/L - h) * tab_width) * (T - T_infty) * T_test * ds(2)
    )
    F_SPMCC = F1 + F2 + F3 + F4 + F5

    # Jacobian
    J = derivative(F_SPMCC, u)

    # Nonlinear solve
    solve(
        F_SPMCC == 0,
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

    # Plotting
    counter += 1
    if timeplots == True and counter % 5 == 1:
        plt.figure(1, figsize=(15, 9))
        plt.subplot(2, 3, 1)
        p1 = plot(V_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("V (V)")
        plt.colorbar(p1)
        plt.subplot(2, 3, 2)
        p2 = plot(I_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("I (A/m^2)")
        plt.colorbar(p2)
        plt.subplot(2, 3, 3)
        p3 = plot(T_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("T (K)")
        plt.colorbar(p3)
        plt.subplot(2, 3, 4)
        p4 = plot(c_n_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("c_n (mol/m^3)")
        plt.colorbar(p4)
        plt.subplot(2, 3, 5)
        p5 = plot(c_p_split)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("c_p (mol/m^3)")
        plt.colorbar(p5)
        plt.subplot(2, 3, 6)
        p6 = plot(Q_bar(psi, V_split, I_split, c_n_split, c_p_split, T_split))
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title("Q (W/m^3)")
        plt.colorbar(p6)
        plt.suptitle("Solution at t = {:.0f} seconds ".format(t))
        plt.show()


# Plot solution ---------------------------------------------------------------

# Font stuff
plt.rc("text", usetex=True)
plt.rc("font", family="sans-serif")
plt.rc("mathtext", fontset="stixsans")
plt.rc("text.latex", preamble=r"\usepackage{sfmath}")
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=18)
plt.rc("axes", titlepad=10)

# Make plots
fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
ax = plt.gca()
p1 = plot(V_split)
p1.set_cmap("viridis")
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"\textbf{Potential (V)}", fontsize=24)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(p1, cax=cax)
fig.tight_layout()
plt.savefig("V_2D.eps", format="eps", dpi=1000, bbox_inches="tight")

fig = plt.figure(figsize=(12 / 2.54, 18 / 2.54))
ax = plt.gca()
p2 = plot(T_split)
p2.set_cmap("plasma")
plt.xlabel(r"$y$", fontsize=22)
plt.ylabel(r"$z$", fontsize=22)
plt.title(r"\textbf{Temperature (K)}", fontsize=24)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(p2, cax=cax)
fig.tight_layout()
plt.savefig("T_2D.eps", format="eps", dpi=1000, bbox_inches="tight")
plt.show()
