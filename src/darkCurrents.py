from src.comsolComponent import ComsolComponent
import os
import pandas as pd
from src.utils.transform import normalize_variables
from src.linear_fits import (
    PieceWiseFit,
    LinearFit,
    _wrapper_compute_fit,
    piecewise_fit_currents,
)
import numpy as np


class DarkCurrents(ComsolComponent):
    """
    Class used to the studies in dark currents
    """

    def __init__(self, folder_path, files_names, retreat=False):
        self.faraday = 96485.3321  # C / mol
        self.folder_path = folder_path
        self.retreat = retreat

        self.dataframe_dict = {
            key: [folder_path + value[0], value[1]]
            for key, value in files_names.items()
        }

        # Save paths
        self.save_paths = {
            "dark_current": os.path.join(
                folder_path, "treated/", "data_derived_values.csv"
            ),
            "surface_charge_density": os.path.join(
                folder_path, "treated/", "data_surface_charge_density.csv"
            ),
            "surface_electric_potential": os.path.join(
                folder_path, "treated/", "data_surface_electric_potential.csv"
            ),
        }

        # Dataframes to merge at the end as dark_current
        self.list_merge_dataframes = [
            "dark_current",
            "volumetric_charge",
            "tds_stats",
            "E_current_avg",
            "E_current_min",
            "E_current_max",
            "interface_flow",
            "top_boundary_flow",
            "surface_charge_density_min_max",
            "surface_electric_potential_min_max",
        ]

        # Load or treat the DataFrames
        if self.retreat:
            print("Processing raw data...")
            self.process_dataframes()
        else:
            print("Loading treated data...")
            self.dataframes = {}
            self.load_dataframes()

    def process_dataframes(self):
        super().__init__(self.dataframe_dict)
        self.correct_names()

        self.treat_interface_dataframes("surface_charge_density")
        self.treat_interface_dataframes("surface_electric_potential")
        self.melt_interface_dataframes("surface_charge_density")
        self.melt_interface_dataframes("surface_electric_potential")
        self.arc_electric_features_max_min(
            "surface_charge_density", "surface_charge_density"
        )
        self.arc_electric_features_max_min(
            "surface_electric_potential", "surface_electric_potential"
        )
        self.max_min_dataframe("surface_charge_density", "surface_charge_density")
        self.max_min_dataframe(
            "surface_electric_potential", "surface_electric_potential"
        )

        self.treat_merging_columns(["S", "Udc_kV", "Time_s"])

        self.dataframes["dark_current"] = self.merge_dataframes(
            ["S", "Udc_kV", "Time_s"],
            self.list_merge_dataframes,
        )
        self.dataframes["dark_current"]["Time_s"] = self.dataframes["dark_current"][
            "Time_s"
        ].astype("int")
        self.dataframes["dark_current"].sort_values(by="S", inplace=True)
        # Normalize values
        self.dataframes["dark_current"] = normalize_variables(
            self.dataframes["dark_current"], ["S"], ["Current (A)", "Charge"]
        )

        # Save treated dataframes
        self.save_dataframes()

    def save_dataframes(self):
        """Save treated dataframes to CSV files"""
        for df_name, path in self.save_paths.items():
            self.dataframes[df_name].to_csv(path, index=False)

    def load_dataframes(self):
        """Load saved dataframes if available"""
        try:
            self.dataframes["dark_current"] = pd.read_csv(
                self.save_paths["dark_current"]
            )
            self.dataframes["surface_charge_density"] = pd.read_csv(
                self.save_paths["surface_charge_density"]
            )
            self.dataframes["surface_electric_potential"] = pd.read_csv(
                self.save_paths["surface_electric_potential"]
            )
            print("Dataframes loaded successfully from saved files.")
        except FileNotFoundError:
            print("Saved dataframes not found, processing dataframes...")
            self.process_dataframes()

    def treat_merging_columns(self, merging_columns):
        for dataframe in self.list_merge_dataframes:
            self.dataframes[dataframe][merging_columns] = self.dataframes[dataframe][
                merging_columns
            ].astype("int")

    def max_min_dataframe(self, dataframe_name, column_name):
        self.dataframes[f"{dataframe_name}_min_max"] = self.dataframes[
            dataframe_name
        ].drop_duplicates(subset=["Udc_kV", "S", "Time_s"])[
            [
                "Udc_kV",
                "S",
                "Time_s",
                f"{column_name}_min",
                f"{column_name}_max",
            ]
        ]

    def melt_interface_dataframes(self, dataframe_name):
        self.dataframes[dataframe_name] = self.dataframes[dataframe_name].melt(
            id_vars="Arc",
        )
        self.dataframes[dataframe_name][["S", "Udc_kV", "Time_s"]] = self.dataframes[
            dataframe_name
        ]["variable"].str.split(";", expand=True)
        self.dataframes[dataframe_name][["S", "Udc_kV", "Time_s"]] = self.dataframes[
            dataframe_name
        ][["S", "Udc_kV", "Time_s"]].map(lambda a: int(float(a)))

        self.dataframes[dataframe_name].drop(columns="variable", inplace=True)
        self.dataframes[dataframe_name].rename(
            columns={"value": dataframe_name}, inplace=True
        )

    def treat_interface_dataframes(self, dataframe_name):
        df_condition = self.dataframes["dark_current"][
            ["S", "Udc_kV", "Time_s"]
        ].drop_duplicates(["S", "Udc_kV", "Time_s"])
        df_condition["column_name"] = df_condition.apply(
            lambda row: str(row["S"])
            + ";"
            + str(row["Udc_kV"])
            + ";"
            + str(row["Time_s"]),
            axis=1,
        )

        self.dataframes[dataframe_name].columns = [
            self.dataframes[dataframe_name].columns[0]
        ] + df_condition["column_name"].tolist()

    def correct_names(self):
        # Correct columns of eletric
        self.dataframes["dark_current"] = self.dataframes["dark_current"].rename(
            columns={"(1/(m^3*s))": "Udc_kV", "Udc": "Time_s", "(kV)": "Current (A)"}
        )

        self.dataframes["volumetric_charge"] = self.dataframes[
            "volumetric_charge"
        ].rename(columns={"(1/(m^3*s))": "Udc_kV", "Udc": "Time_s", "(kV)": "Charge"})

        self.dataframes["tds_stats"] = self.dataframes["tds_stats"].rename(
            columns={
                "(1/(m^3*s))": "Udc_kV",
                "Udc": "Time_s",
                "(kV)": "concentration_positive",
                "Time": "concentration_negative",
                "(s)": "ion_generation",
                "Concentration": "ion_recombination",
            }
        )

        self.dataframes["E_current_avg"] = self.dataframes["E_current_avg"].rename(
            columns={
                "(1/(m^3*s))": "Udc_kV",
                "Udc": "Time_s",
                "(kV)": "electric_field_norm_avg",
                "Time": "current_density_norm_avg",
                "(s)": "gas_current_density_norm_avg",
                "Electric": "peclet_avg",
            }
        )

        self.dataframes["E_current_min"] = self.dataframes["E_current_min"].rename(
            columns={
                "(1/(m^3*s))": "Udc_kV",
                "Udc": "Time_s",
                "(kV)": "electric_field_norm_min",
                "Time": "current_density_norm_min",
                "(s)": "gas_current_density_norm_min",
                "ec3.normE": "peclet_min",
            }
        )

        self.dataframes["E_current_max"] = self.dataframes["E_current_max"].rename(
            columns={
                "(1/(m^3*s))": "Udc_kV",
                "Udc": "Time_s",
                "(kV)": "electric_field_norm_max",
                "Time": "current_density_norm_max",
                "(s)": "gas_current_density_norm_max",
                "Electric": "peclet_max",
            }
        )

        self.dataframes["interface_flow"] = self.dataframes["interface_flow"].rename(
            columns={
                "(1/(m^3*s))": "Udc_kV",
                "Udc": "Time_s",
                "(kV)": "Current (A)_interface",
            }
        )

        self.dataframes["top_boundary_flow"] = self.dataframes[
            "top_boundary_flow"
        ].rename(
            columns={
                "(1/(m^3*s))": "Udc_kV",
                "Udc": "Time_s",
                "(kV)": "Current (A)_top",
            }
        )

        # Remove empty columns due to columns name splitting when reading file
        for key, values in self.dataframes.items():
            self.dataframes[key] = values.dropna(axis=1, how="all")

    def arc_electric_features_max_min(self, dataframe, transform_columns):
        self.dataframes[dataframe] = self.dataframes[dataframe].merge(
            self.dataframes[dataframe]
            .groupby(["S", "Udc_kV", "Time_s"])[transform_columns]
            .agg(["max", "min"])
            .reset_index(),
            on=["S", "Udc_kV", "Time_s"],
        )
        self.dataframes[dataframe].rename(
            columns={
                "max": f"{transform_columns}_max",
                "min": f"{transform_columns}_min",
            },
            inplace=True,
        )

    def compute_charge(self):
        """
        Compute charge based on the concentration of ions
        """
        self.dataframes["dark_current"]["negative_charge"] = (
            self.dataframes["dark_current"]["concentration_negative"] * self.faraday
        )
        self.dataframes["dark_current"]["positive_charge"] = (
            self.dataframes["dark_current"]["concentration_positive"] * self.faraday
        )

    def compute_ion_generation_comsol(self):
        """
        Receives the darkcurrent dataframe and computes the ion generation and recombination
        Doesn't insert in the darkcurrent dataframe as the ion_generation is repeated
        """
        df_ion = self.dataframes["dark_current"][
            ["S", "ion_generation", "ion_recombination"]
        ].drop_duplicates("S")
        df_ion["charge_generation"] = df_ion["ion_generation"] * self.faraday
        return df_ion


def piecewise_analysis(df):
    """
    Analysis of the current vs voltage with a piece wise fit - identify the log-like curve
    """
    # Exponential increase of current with Udc - dependance on S?
    df["Current (A)_log"] = np.log(df["Current (A)"])
    piecewise_fit_currents(df, S=210000000, number_of_segments=2, plot=True)

    fittings = df.groupby(["S"])[["S", "Current (A)_log", "Time_s", "Udc_kV"]].apply(
        _wrapper_compute_fit
    )
    fittings = fittings.droplevel(level=0).reset_index(drop=True)
    return fittings


def saturation_current_analysis(df):
    # Parameters studied
    print(f"{df[['S', 'Udc_kV']].drop_duplicates()}")

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
