def compute_s_value(
    phi, pressure, temperature, gas_molar_mass, gas_constant=8.314, w_ev=34
):
    # Constants
    e = 1.602e-19  # Elementary charge in Coulombs
    w_joules = w_ev * e  # Convert eV to Joules

    # Compute density using ideal gas law
    rho = pressure / (gas_constant * temperature)
    rho = rho * gas_molar_mass
    print(f"rho: {rho}")

    # Compute S value
    s_value = (phi * rho) / (w_joules)

    return s_value


# Example usage
phi = 0.02e-6  # Example radiation flux Gy/s
pressure = 0.5e6  # Pressure in Pascals (1 atm)
temperature = 273  # Temperature in Kelvin
gas_molar_mass = 146.0554e-3  # kg/mol
s_value = compute_s_value(phi, pressure, temperature, gas_molar_mass)
print(f"S value: {s_value}")
