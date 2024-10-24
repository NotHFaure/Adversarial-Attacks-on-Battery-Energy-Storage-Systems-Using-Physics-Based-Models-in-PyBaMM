import pybamm
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Initialize the battery model with SEI included
options = {"SEI": "constant"}
model = pybamm.lithium_ion.DFN(options)
param = model.default_parameter_values  # Use the default parameter values from the model

# Adjust solver settings for faster computation
solver = pybamm.CasadiSolver()
solver.rtol = 1e-3  # Increase relative tolerance
solver.atol = 1e-6  # Increase absolute tolerance
solver.max_steps = 10000  # Limit the maximum number of steps
solver.dt_max = 1  # Set maximum time step size

# Define the time range for the simulation (from 0 to 1000 seconds, with fewer points)
t_eval = np.linspace(0, 1000, 200)

# Run the normal simulation (no adversarial attacks)
sim_normal = pybamm.Simulation(model, parameter_values=param, solver=solver)
solution_normal = sim_normal.solve(t_eval)
voltage_normal = solution_normal["Terminal voltage [V]"].entries
current_normal = np.full_like(t_eval, 3)  # Nominal current is constant at 3A

# Define the attack window
attack_start_time = 100
attack_end_time = 200

# Adversarial current attack function
def adversarial_current(t, perturbation_amplitude, perturbation_increase, for_plot=False):
    base_current = 3  # Constant 3A discharge
    transition_width = 10  # Smooth transition width for attack
    tanh_func = np.tanh if for_plot else pybamm.tanh

    # Smooth transition for attack window
    attack_window = 0.5 * (1 + tanh_func((t - attack_start_time) / transition_width)) * \
                    0.5 * (1 + tanh_func((attack_end_time - t) / transition_width))

    # Apply dynamic perturbation that increases over time
    perturbation = attack_window * (perturbation_amplitude + perturbation_increase * (t - attack_start_time))
   
    return base_current + perturbation

# Adversarial temperature attack with lasting effect
def adversarial_temperature(t, temperature_spike, for_plot=False):
    base_temperature = 298.15  # Base temperature in K (25°C)
    transition_width = 10  # Smooth transition width for attack
    tanh_func = np.tanh if for_plot else pybamm.tanh

    # Smooth transition for attack window
    attack_window = 0.5 * (1 + tanh_func((t - attack_start_time) / transition_width)) * \
                    0.5 * (1 + tanh_func((attack_end_time - t) / transition_width))

    # After attack, maintain a lasting effect using another smooth transition
    after_attack = 0.5 * (1 + tanh_func((t - attack_end_time) / transition_width))

    # Combine both windows to keep the temperature elevated after the attack window ends
    temp_perturbation = attack_window * temperature_spike + after_attack * temperature_spike + base_temperature
    
    return temp_perturbation

# Corrected adversarial SEI resistivity attack function
def adversarial_SEI_resistivity_constant(t, resistivity_spike, for_plot=False):
    base_SEI_resistivity = param["SEI resistivity [Ohm.m]"]
    sign_func = np.sign if for_plot else pybamm.sign

    # Define the attack window using the sign function
    attack_window = 0.5 * (sign_func(t - attack_start_time) - sign_func(t - attack_end_time))

    # SEI resistivity during attack
    sei_resistivity = base_SEI_resistivity + attack_window * resistivity_spike
    return sei_resistivity

# Define the objective function to maximize the voltage deviation
def objective(params):
    perturbation_amplitude, perturbation_increase, temperature_spike, resistivity_spike = params
   
    # Update parameters with adversarial attack functions
    param_attack = param.copy()
    param_attack.update({
        "Current function [A]": adversarial_current(pybamm.t, perturbation_amplitude, perturbation_increase),
        "Ambient temperature [K]": adversarial_temperature(pybamm.t, temperature_spike),
        "SEI resistivity [Ohm.m]": adversarial_SEI_resistivity_constant(pybamm.t, resistivity_spike)
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
   
    common_time = t_eval  # Use the same time grid
    voltage_adversarial = voltage_interp(common_time)
    voltage_normal_resampled = voltage_normal_interp(common_time)
   
    # Calculate deviation
    deviation = np.max(np.abs(voltage_adversarial - voltage_normal_resampled))
    return -deviation  # Minimize the negative of deviation (i.e., maximize the deviation)

# Function to plot the results of the attacks
def plot_results(optimal_params, t_eval, voltage_normal, current_normal):
    # Solve with optimal parameters for current, temperature, and SEI resistivity attacks
    param_attack = param.copy()
    param_attack.update({
        "Current function [A]": adversarial_current(pybamm.t, optimal_params[0], optimal_params[1]),
        "Ambient temperature [K]": adversarial_temperature(pybamm.t, optimal_params[2]),
        "SEI resistivity [Ohm.m]": adversarial_SEI_resistivity_constant(pybamm.t, optimal_params[3])
    })
    
    # Run simulation
    sim = pybamm.Simulation(model, parameter_values=param_attack, solver=solver)
    solution = sim.solve(t_eval)
    
    # Extract adversarial voltage and time data
    voltage = solution["Terminal voltage [V]"].entries
    time_voltage = solution["Time [s]"].entries
    
    # Interpolate adversarial solution to match time grid of t_eval
    voltage_interp = interp1d(time_voltage, voltage, kind="linear", fill_value="extrapolate")(t_eval)
    
    # Plot terminal voltage
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, voltage_interp, label="Combined Adversarial Attack", color='red')
    plt.plot(t_eval, voltage_normal, label="Normal Condition", color='blue')
    plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
    plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
    plt.xlabel("Time [s]")
    plt.ylabel("Terminal Voltage [V]")
    plt.title("Terminal Voltage: Combined Current, Temperature, and SEI Resistivity Attack")
    plt.legend()
    plt.grid(True)
    plt.savefig("terminal_voltage_combined_attack.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    # Plot current profile
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
    plt.savefig("current_attack_profile.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    # Plot temperature profile
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, adversarial_temperature(t_eval, optimal_params[2], for_plot=True), label="Adversarial Temperature", color='red')
    plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
    plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [K]")
    plt.title("Temperature: Adversarial Attack")
    plt.legend()
    plt.grid(True)
    plt.savefig("temperature_attack_profile.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    
    # Plot SEI resistivity profile
    plt.figure(figsize=(10, 6))
    sei_resistivity = adversarial_SEI_resistivity_constant(t_eval, optimal_params[3], for_plot=True)
    plt.plot(t_eval, sei_resistivity, label="Adversarial SEI Resistivity", color='red')
    plt.axvline(x=attack_start_time, color='green', linestyle='--', label="Attack Start")
    plt.axvline(x=attack_end_time, color='orange', linestyle='--', label="Attack End")
    plt.xlabel("Time [s]")
    plt.ylabel("SEI Resistivity [Ohm·m]")
    plt.title("SEI Resistivity: Adversarial Attack")
    plt.legend()
    plt.grid(True)
    plt.savefig("sei_resistivity_attack_profile.pdf", format="pdf", bbox_inches="tight")
    plt.show()

# Set initial guesses for the parameters and run the optimization
initial_guess = [0.005, 0.02, 10, 1e-3]  # Adjust initial guesses if necessary
bounds = [(0, 0.1), (0, 0.05), (0, 15), (0, 1e-2)]  # Bounds for parameters

# Run the optimization
result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")

# Extract optimal parameters
optimal_params = result.x
print(f"Optimal Parameters: {optimal_params}")

# Plot the results
plot_results(optimal_params, t_eval, voltage_normal, current_normal)
