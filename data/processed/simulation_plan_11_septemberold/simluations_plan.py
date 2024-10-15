import itertools
import numpy as np

# Define the parameters as lists
S = ["1E7", "2.9E7", "1.5E8"]
Udc = ["15", "50", "150", "334"]
mup = ["1.36E-4", "3.6E-5", "4.8E-6"]
mun = ["1.87E-4", "3.6E-5", "4.8E-6"]
erEpoxy = ["4.5", "5", "5.5"]
kEpoxy = ["3.33E-19", "3.33E-18"]
kGas = ["1E-16", "1E-17", "1E-20"]

# Generate the cartesian product (combinations) of all parameters
combinations = list(itertools.product(S, Udc, mup, mun, erEpoxy, kEpoxy, kGas))

# Transpose the list of combinations to get lists for each parameter
# This will group all values for 'S', 'Udc', 'mup', etc.
transposed_combinations = list(zip(*combinations))

# Print the results in the desired format
parameter_names = ["S", "Udc", "mup", "mun", "erEpoxy", "kEpoxy", "kGas"]
parameters_dict = {}
for i, param in enumerate(parameter_names):
    values = ", ".join(transposed_combinations[i])
    parameters_dict[param] = values

# Define the units or additional text for each parameter
units = {
    "S": "[1/(m^3*s)]",
    "Udc": "[kV]",
    "mup": "[m^2/(V*s)]",
    "mun": "[m^2/(V*s)]",
    "erEpoxy": "[]",
    "kEpoxy": "[S/m]",
    "kGas": "[S/m]",
}

batch_size = 200
num_values = len(list(parameters_dict["S"].split(",")))
print(f"{num_values = }")
for simulation_run in range(int(np.ceil(num_values / batch_size))):
    # Create a dictionary to hold the current batch of parameters
    parameters_dict_run = {}

    # Get the start and end indices for the current batch
    start_idx = simulation_run * batch_size
    end_idx = min(start_idx + batch_size, num_values)

    print(f"{start_idx = }")
    print(f"{end_idx = }")
    # Slice each parameter's values to get the current batch
    for param, values in parameters_dict.items():
        values_list = values.split(", ")
        parameters_dict_run[param] = ", ".join(values_list[start_idx:end_idx])

    # Save the dictionary to a text file with proper formatting
    with open(
        f"./simulation_plan/parameters_simulation_grid_run{simulation_run}.txt", "w"
    ) as file:
        for param, values in parameters_dict_run.items():
            unit = units.get(param, "")  # Fetch the unit if available, else empty
            file.write(f'{param} "{values}" {unit}\n')
