def compute_s_value(phi, pressure, temperature, gas_molar_mass, w_ev=34):
    """
    Compute the S alue based on Lopes and Lucchini's paper
    """
    # Constants
    gas_constant = 8.314  # J/molK
    e = 1.602e-19  # Elementary charge in Coulombs
    w_joules = w_ev * e  # Convert eV to Joules

    # Compute density using ideal gas law
    rho = pressure / (gas_constant * temperature)
    rho = rho * gas_molar_mass
    print(f"rho: {rho}")

    # Compute S value
    s_value = (phi * rho) / (w_joules)

    return s_value


def compute_s_value_lopes(dose, pressure, temperature, G_ion):
    """
    - Dose should be in Gy/s
    - density in kg/m3
    - G_ion in ion pairs/100eV
    """
    # Compute density using ideal gas law
    gas_constant = 8.314  # J/molK
    rho = pressure / (gas_constant * temperature)
    return dose * rho * G_ion * 6.25e16


# Example usage
phi = 1  # Example radiation flux Gy/s
pressure = 5e5  # Pressure in Pascals (5 bar)
temperature = 273  # Temperature in Kelvin
gas_molar_mass = 146.0554e-3  # kg/mol
s_value = compute_s_value(
    phi=phi, pressure=pressure, temperature=temperature, gas_molar_mass=gas_molar_mass
)
print(f"S value: {s_value}")

g_ion = 2.95
dose = 0.8 / 365 / 24 / 3600  # Gy/s
s_value_lopes = compute_s_value_lopes(
    dose=dose, pressure=pressure, temperature=temperature, G_ion=g_ion
)
print(f"S value: {s_value_lopes}")
