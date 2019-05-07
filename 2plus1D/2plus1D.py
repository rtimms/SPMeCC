from make_parameters import Parameters
from current_collector import CCVoltage
import numpy as np


# Load parameters -------------------------------------------------------------
C_rate = 1
# param = Parameters(C_rate)
param = Parameters(C_rate, 'mypouch')

# Create class for current collector problem ----------------------------------
Ny, Nz = 10, 10  # Number of gridpoints
degree = 1  # Degree of polynomial
current_collector_model = CCVoltage(param, Ny, Nz, degree)

# Assemble matrices for current collector problem -----------------------------
current_collector_model.assemble()

# Get initial through-cell current from external model ------------------------
# GET CONSISTANT ICs

# test e.g. current 1 everywhere
current = np.ones(np.size(current_collector_model.get_voltage()))
current_collector_model.update_current_values(current)

# Timestepping ----------------------------------------------------------------
t = 0.0  # initial time
t_final = (3600) / param.tau_d_star  # final time
dt = 15 / param.tau_d_star  # Coarse step size
tol = 1E-3

while t < t_final:

    # Increase time
    t += dt

    while current_collector_model.voltage_difference > tol:

        # Update voltage
        current_collector_model.solve()

        # Compute new through-cell current
        # DO SOMETHING
        current_collector_model.update_current_values(current)

    # test plotting
    current_collector_model.plot()
