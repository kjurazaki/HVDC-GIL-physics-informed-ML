from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
from src.comsolComponent import ComsolComponent
from src.darkCurrents import DarkCurrents
from src.utils.plotting import FlexiblePlotter, InteractivePlot
from src.utils.transform import normalize_variables, compute_t90
from src.cylindrical_insulator import load_cylindrical_comsol_parameters
import seaborn as sns


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
    df = data.dataframes["surface_charge_density"]

    df_grouped = (
        df.groupby(["S", "Udc_kV", "Time_s"])["surface_charge_density"]
        .agg(
            [
                ("mean", "mean"),
                ("median", "median"),
                ("min", "min"),
                ("max", "max"),
                ("abs_sum", lambda x: x.abs().sum()),
                ("list_values", lambda x: list(x)),
            ]
        )
        .reset_index()
    )
    df_plot = df_grouped[
        (df_grouped["S"] == 2900000)
        & (df_grouped["Udc_kV"] == 335)
        & (df_grouped["Time_s"] > 0)
    ]
    fig, axs = plt.subplots(2, 3, figsize=(10, 8))

    axs[0, 0].scatter(df_plot["Time_s"], df_plot["mean"], label="mean", c="blue")
    axs[0, 0].set_title("Mean")
    axs[0, 0].legend()

    axs[0, 1].scatter(df_plot["Time_s"], df_plot["median"], label="median", c="red")
    axs[0, 1].set_title("Median")
    axs[0, 1].legend()

    axs[0, 2].scatter(df_plot["Time_s"], df_plot["max"], label="max", c="green")
    axs[0, 2].set_title("Max")
    axs[0, 2].legend()

    axs[1, 0].scatter(df_plot["Time_s"], df_plot["min"], label="min", c="orange")
    axs[1, 0].set_title("Min")
    axs[1, 0].legend()

    axs[1, 1].scatter(
        df_plot["Time_s"], df_plot["abs_sum"], label="abs sum", c="purple"
    )
    axs[1, 1].set_title("Absolute Sum")
    axs[1, 1].legend()

    for ax in axs.flat:
        ax.set(xlabel="Time_s", ylabel="Value")

    plt.tight_layout()
    plt.show()

    # Add violin plot for each Time_s of the surface_charge_density
    # Sample df_plot in uniform timesteps, so the graph has only 10 points of Time_s
    df_plot = df_plot.iloc[:: max(1, len(df_plot) // 10)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=df_plot.explode("list_values"), x="Time_s", y="list_values", ax=ax
    )
    plt.title("Violin Plot")
    plt.xlabel("Time_s")
    plt.ylabel("Surface Charge Density")
    plt.show()
    print(df_grouped)


if __name__ == "__main__":
    main()
