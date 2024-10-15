from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from src.comsolComponent import ComsolComponent
from src.darkCurrents import DarkCurrents
from src.utils.plotting import FlexiblePlotter, InteractivePlot
from src.utils.transform import normalize_variables, compute_t90  # {{ edit_2 }}
from src.linear_fits import (
    PieceWiseFit,
    LinearFit,
    _wrapper_compute_fit,
    piecewise_fit_currents,
)


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


def main():
    folder_path = "C:/Users/kenji/OneDrive/2024/Research DII006/experimentDarkCurrent/databases_darkcurrent/"
    files_names = {
        "dark_current": ["darkcurrent_lineintegration.txt", 4],
        "volumetric_charge": ["volumetric_charge.txt", 4],
        "tds_stats": ["tds_stats.txt", 4],
        "E_current_avg": ["E_current_avg.txt", 4],
        "E_current_min": ["E_current_min.txt", 4],
        "E_current_max": ["E_current_max.txt", 4],
        "interface_flow": ["interface_flow.txt", 4],
        "top_boundary_flow": ["top_boundary_flow.txt", 4],
        "surface_charge_density": ["surface_charge_density_interface.txt", 7],
        "surface_electric_potential": ["surface_electric_potential_interface.txt", 7],
    }
    data = DarkCurrents(
        folder_path, files_names, retreat=False
    )  # Use retreat true if any changes in the files
    df = data.dataframes["dark_current"]
    df["Time_s"] = df["Time_s"].astype("int")
    df.sort_values(by="S", inplace=True)

    # Normalize values
    df = normalize_variables(df, ["S"], ["Current (A)", "Charge"])

    # Parameters studied
    print(f"{df[['S', 'Udc_kV']].drop_duplicates()}")

    # Ion generation charge
    faraday = 96485.3321  # C / mol
    df_ion = df[["S", "ion_generation", "ion_recombination"]].drop_duplicates("S")
    df_ion["charge_generation"] = df_ion["ion_generation"] * faraday

    df["negative_charge"] = df["concentration_negative"] * faraday
    df["positive_charge"] = df["concentration_positive"] * faraday

    # Exponential increse of current with Udc - dependance on S?
    df["Current (A)_log"] = np.log(df["Current (A)"])
    piecewise_fit_currents(df, S=210000000, number_of_segments=2, plot=True)

    fittings = df.groupby(["S"])[["S", "Current (A)_log", "Time_s", "Udc_kV"]].apply(
        _wrapper_compute_fit
    )
    fittings = fittings.droplevel(level=0).reset_index(drop=True)

    # Linear regression of Current saturation vs ion-pair generation
    linear_fit = LinearFit(
        df.loc[
            (df["Udc_kV"] == df["Udc_kV"].max()) & (df["Time_s"] == df["Time_s"].max()),
            "S",
        ],
        df.loc[
            (df["Udc_kV"] == df["Udc_kV"].max()) & (df["Time_s"] == df["Time_s"].max()),
            "Current (A)",
        ],
    )
    linear_fit.fit_model()
    linear_fit.graph_fit()


if __name__ == "__main__":
    main()
