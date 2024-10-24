import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Set up the battery model (DFN model)
model = pybamm.lithium_ion.DFN()
param = model.default_parameter_values

# Define attack start and end times
attack_start_time = 150  # Start attack at 150 seconds
attack_end_time = 170    # End attack at 170 seconds
tau = 10                 # Time constant for smoothing transitions

# Define the base ambient temperature and the spike temperature
T_base = 298.15  # Nominal ambient temperature in Kelvin

# Define the attack parameters: temperature manipulation
def adversarial_temperature_pyBAMM(t, T_spike):
    gamma = 0.5 * (1 + pybamm.tanh((t - attack_start_time) / tau)) * \
            0.5 * (1 + pybamm.tanh((attack_end_time - t) / tau))
    delta_T = gamma * (T_spike - T_base)
    return T_base + delta_T

# Define the plotting-friendly version for adversarial temperature (uses NumPy)
def adversarial_temperature_plot(t, T_spike):
    gamma = 0.5 * (1 + np.tanh((t - attack_start_time) / tau)) * \
            0.5 * (1 + np.tanh((attack_end_time - t) / tau))
    delta_T = gamma * (T_spike - T_base)
    return T_base + delta_T

# Solve the model without adversarial attack (normal scenario)
t_eval = np.linspace(0, 1000, 500)

solver = pybamm.CasadiSolver()
solver.max_steps = 50000  # Increase the max number of steps to avoid early termination
solver.rtol = 1e-6  # Relax relative tolerance
solver.atol = 1e-8  # Relax absolute tolerance
solver.dt_max = 1e-2  # Limit max time step to avoid excessively small steps

sim_normal = pybamm.Simulation(model, parameter_values=param, solver=solver)
solution_normal = sim_normal.solve(t_eval)
voltage_normal = solution_normal["Terminal voltage [V]"].entries
current_normal = np.full_like(t_eval, 3)  # Nominal current is constant at 3A

# Define an objective function to maximize the voltage deviation
def objective(params):
    T_spike = params[0]
   
    # Update parameters with adversarial temperature attack function
    param_attack = param.copy()
    param_attack.update({
        "Ambient temperature [K]": adversarial_temperature_pyBAMM(pybamm.t, T_spike),
    }, check_already_exists=False)

    # Solve the model with adversarial attack
    sim = pybamm.Simulation(model, parameter_values=param_attack, solver=solver)
   
    try:
        solution = sim.solve(t_eval)
    except pybamm.SolverError:
        return np.inf  # Penalize if the solver fails
   
    # Extract relevant data from the solution
    voltage = solution["Terminal voltage [V]"].entries
    voltage_interp = interp1d(solution["Time [s]"].entries, voltage, kind="linear", fill_value="extrapolate")
   
    # Interpolate normal solution to the adversarial time grid
    voltage_normal_interp = interp1d(solution_normal["Time [s]"].entries, voltage_normal, kind="linear", fill_value="extrapolate")
   
    common_time = np.linspace(0, 1000, 500)  # Define a common time grid for comparison
    voltage_adversarial = voltage_interp(common_time)
    voltage_normal_resampled = voltage_normal_interp(common_time)
   
    # Calculate deviation
    deviation = np.max(np.abs(voltage_adversarial - voltage_normal_resampled))
    return -deviation  # Minimize the negative of deviation (i.e., maximize the deviation)

# Set initial guesses for the parameters and run the optimization
initial_guess = [310]  # Initial guess for T_spike in Kelvin
bounds = [(T_base, T_base + 20)]  # Limit T_spike to a reasonable range, e.g., up to 318.15 K
result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

# Extract optimal parameters and plot the results
optimal_params = result.x
print(f"Optimal Parameters: {optimal_params}")

# Now plot the results with the optimal parameters
param_attack = param.copy()
param_attack.update({
    "Ambient temperature [K]": adversarial_temperature_pyBAMM(pybamm.t, optimal_params[0]),
})

# Solve with optimal parameters
sim = pybamm.Simulation(model, parameter_values=param_attack, solver=solver)
solution_opt = sim.solve(t_eval)

# Extract adversarial voltage
voltage_opt = solution_opt["Terminal voltage [V]"].entries
time_temp = solution_opt["Time [s]"].entries

# Interpolate the adversarial solution to match the time grid of t_eval
voltage_opt_interp = interp1d(time_temp, voltage_opt, kind="linear", fill_value="extrapolate")(t_eval)

# Configure plot for Times New Roman and font size 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# Plot the terminal voltage
plt.figure(figsize=(10, 6))
plt.plot(t_eval, voltage_opt_interp, label="Adversarial Temperature Attack", color='red')
plt.plot(t_eval, voltage_normal, label="Normal Condition", color='blue')
plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
plt.xlabel("Time [s]")
plt.ylabel("Terminal Voltage [V]")
plt.title("Terminal Voltage: Temperature-Only Adversarial Attack")
plt.legend()
plt.grid(True)
plt.savefig("terminal_voltage_temperature_attack.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot temperature
plt.figure(figsize=(10, 6))
temperature_adversarial = adversarial_temperature_plot(t_eval, optimal_params[0])
plt.plot(t_eval, temperature_adversarial - 273.15, label="Adversarial Temperature", color='red')
plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
plt.xlabel("Time [s]")
plt.ylabel("Ambient Temperature [°C]")
plt.title("Ambient Temperature: Adversarial Temperature Attack")
plt.legend()
plt.grid(True)
plt.savefig("ambient_temperature_attack.pdf", format="pdf", bbox_inches="tight")
plt.show()
