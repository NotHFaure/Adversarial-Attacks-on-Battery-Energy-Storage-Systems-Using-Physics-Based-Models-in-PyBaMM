import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Set up the battery model (DFN model)
model = pybamm.lithium_ion.DFN()
param = model.default_parameter_values

# Define attack start and end times
attack_start_time = 100  # Start attack at 100 seconds
attack_end_time = 200    # End attack at 200 seconds

# Perturbation scaling for a feasible attack
perturbation_amplitude = 0.005  # Start with a very low perturbation amplitude
perturbation_increase = 0.02  # Gradually increase perturbation over time

# Define the attack parameters: current, SEI resistance
def adversarial_current(t, perturbation_amplitude, perturbation_increase, for_plot=False):
    base_current = 3  # Constant 3A discharge
    transition_width = 10  # Smooth transition width for attack

    # If for plotting, use numpy.tanh; otherwise use pybamm.tanh for PyBaMM simulations
    tanh_func = np.tanh if for_plot else pybamm.tanh

    # Smooth transition for attack window
    attack_window = 0.5 * (1 + tanh_func((t - attack_start_time) / transition_width)) * \
                    0.5 * (1 + tanh_func((attack_end_time - t) / transition_width))

    # Apply dynamic perturbation that increases over time
    perturbation = attack_window * (perturbation_amplitude + perturbation_increase * (t - attack_start_time))
   
    return base_current + perturbation

# Manipulate SEI resistance as part of the attack (Simulate internal resistance increase)
def adversarial_sei_resistance(t, sei_increase):
    base_sei_resistance = 1e-3  # Base SEI resistance in Ohm.m2
    transition_width = 10  # Smooth transition width for attack

    # Smooth transition for attack window using PyBaMM's tanh
    attack_window = 0.5 * (1 + pybamm.tanh((t - attack_start_time) / transition_width)) * \
                    0.5 * (1 + pybamm.tanh((attack_end_time - t) / transition_width))

    # Slowly increase SEI resistance
    resistance_spike = attack_window * sei_increase + base_sei_resistance  # Increase resistance
    return resistance_spike

# Solve the model without adversarial attack (normal scenario)
t_eval = np.linspace(0, 1000, 500)
sim_normal = pybamm.Simulation(model, parameter_values=param)
solution_normal = sim_normal.solve(t_eval)
voltage_normal = solution_normal["Terminal voltage [V]"].entries
current_normal = np.full_like(t_eval, 3)  # Nominal current is constant at 3A

# Define an objective function to maximize the voltage deviation
def objective(params):
    perturbation_amplitude, perturbation_increase, sei_increase = params
   
    # Update parameters with adversarial attack functions
    param.update({
        "Current function [A]": adversarial_current(pybamm.t, perturbation_amplitude, perturbation_increase),
        "Negative electrode SEI resistance [Ohm.m2]": adversarial_sei_resistance(pybamm.t, sei_increase)
    }, check_already_exists=False)
   
    # Solve the model with adversarial attack
    sim = pybamm.Simulation(model, parameter_values=param)
   
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
initial_guess = [0.005, 0.02, 1e-3]  # Perturbation amplitude, rate, SEI resistance
bounds = [(0, 0.1), (0, 0.05), (1e-3, 2e-3)]  # Tight bounds for more controlled attacks
result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

# Extract optimal parameters and plot the results
optimal_params = result.x
print(f"Optimal Parameters: {optimal_params}")

# Now plot the results with the optimal parameters
param.update({
    "Current function [A]": adversarial_current(pybamm.t, optimal_params[0], optimal_params[1]),
    "Negative electrode SEI resistance [Ohm.m2]": adversarial_sei_resistance(pybamm.t, optimal_params[2])
})

# Solve with optimal parameters
sim = pybamm.Simulation(model, parameter_values=param)
solution_opt = sim.solve(t_eval)

# Extract adversarial voltage
voltage_opt = solution_opt["Terminal voltage [V]"].entries
time_voltage = solution_opt["Time [s]"].entries

# Interpolate the adversarial solution to match the time grid of t_eval
voltage_opt_interp = interp1d(time_voltage, voltage_opt, kind="linear", fill_value="extrapolate")(t_eval)

# Configure plot for Times New Roman and font size 20
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 20

# Plot the terminal voltage
plt.figure(figsize=(10, 6))
plt.plot(t_eval, voltage_opt_interp, label="Combined Adversarial Attack", color='red')
plt.plot(t_eval, voltage_normal, label="Normal Condition", color='blue')
plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
plt.xlabel("Time [s]")
plt.ylabel("Terminal Voltage [V]")
plt.title("Terminal Voltage: Optimal Adversarial Attack")
plt.legend()
plt.grid(True)
plt.savefig("terminal_voltage_attack.pdf", format="pdf", bbox_inches="tight")
plt.show()

# Plot current
plt.figure(figsize=(10, 6))
plt.plot(t_eval, adversarial_current(t_eval, optimal_params[0], optimal_params[1], for_plot=True), label="Adversarial Current", color='red')
plt.plot(t_eval, current_normal, label="Normal Current", color='blue')
plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
plt.xlabel("Time [s]")
plt.ylabel("Current [A]")
plt.title("Current: Adversarial Attack")
plt.legend()
plt.grid(True)
plt.savefig("current_attack.pdf", format="pdf", bbox_inches="tight")
plt.show()