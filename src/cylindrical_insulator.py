def load_cylindrical_comsol_parameters():
    """
    Data related to the study [[dark_currents_TDS]], check obsidian for more context
    """
    # Volume of the gas chamber computed on COMSOL m^3
    volume_of_chamber = 0.024946
    # Area of the electrodes that the gas ions are integrated m^2
    area_dark_currents = 0.46954
    # Area of the interface [m^2]
    area_interface_insulator_gas = 0.045247
    # length interface [m]
    length_interface = 0.15129
    # Area of the insulated wall [m^2]
    area_insulated_wall = 0.090478
    # Length insulated wall [m]
    length_insulated_wall = 0.12
    # Elementary charge C
    e = 1.602176634e-19
    # Relation of saturation levels and S
    L_char = 1.86088545e-21 / e / area_dark_currents

    return {
        "volume_of_chamber": volume_of_chamber,
        "area_dark_currents": area_dark_currents,
        "area_interface_insulator_gas": area_interface_insulator_gas,
        "length_interface": length_interface,
        "area_insulated_wall": area_insulated_wall,
        "length_insulated_wall": length_insulated_wall,
        "e": e,
        "L_char": L_char,
    }
