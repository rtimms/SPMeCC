from make_parameters import Parameters
from current_collector import CCVoltage


# Load parameters -------------------------------------------------------------
C_rate = 1
# param = Parameters(C_rate)
param = Parameters(C_rate, 'mypouch')

# Create class for current collector problem ----------------------------------
Ny, Nz = 10, 10  # Number of gridpoints
degree = 1  # Degree of polynomial
CC_model = CCVoltage(param, Ny, Nz, degree)

# Assemble matrices for current collector problem -----------------------------
CCVoltage.assemble()

# Get initial through-cell current from external model ------------------------
# GET CONSISTANT ICs

# Timestepping ----------------------------------------------------------------
t = 0.0  # initial time
t_final = (3600) / param.tau_d_star  # final time
dt = 15 / param.tau_d_star  # Coarse step size
tol = 1E-3

while t < t_final:

    # Increase time
    t += dt

    while CCVoltage.voltage_difference > tol:

        # Update voltage
        CCVoltage.solve()

        # Compute new through-cell current
        # DO SOMETHING
        CCVoltage.update_current_values()
