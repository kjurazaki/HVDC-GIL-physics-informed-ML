from src.comsolComponent import ComsolComponent
import os
from itertools import chain

import pandas as pd
from src.utils.transform import normalize_variables
from src.linearFit import (
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

    def __init__(
        self,
        folder_path,
        files_names,
        retreat=False,
        dataframes_to_process=None,
        save=False,
        list_merge_dataframes=None,
        list_merge_1D_dataframes=None,
        save_paths=None,
        parameters_variables=None,
        parameters_variables_1D=None,
        modifier="",
    ):
        self.faraday = 96485.3321  # C / mol
        self.folder_path = folder_path
        self.retreat = retreat
        self.save = save
        self.parameters_variables = parameters_variables
        self.parameters_variables_1D = parameters_variables_1D
        # Save paths
        self.save_paths = save_paths
        # Dataframes to merge at the end as dark_current
        self.list_merge_dataframes = list_merge_dataframes
        # Dataframes to merge 1D related to the surface charge density
        self.list_merge_1D_dataframes = list_merge_1D_dataframes
        self.modifier = modifier
        # Determine which dataframes to process
        if dataframes_to_process is None:
            self.dataframes_to_process = files_names.keys()
        else:
            self.dataframes_to_process = list(dataframes_to_process)

        self.dataframe_dict = {
            key: [folder_path + value[0], value[1]]
            for key, value in files_names.items()
            if key in self.dataframes_to_process
        }

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
        self.treat_merging_columns(self.parameters_variables)

        for df_name in self.dataframes_to_process:
            # 1D integration dataframes
            if df_name in [
                "surface_charge_density",
                "surface_electric_potential",
                "surface_charge_density_down",
                "surface_charge_density_up",
                "surface_Ez_up",
                "surface_Ez_down",
                "surface_Er_up",
                "surface_Er_down",
                "HV_Er",
                "HV_Ez",
                "Grd_Er",
                "Grd_Ez",
            ]:
                # This should be done only if the dataframe has many columns - happens when save as many columns in COMSOL
                # Columns has the same name as the dataframe
                transform_column = df_name
                if self.dataframes[df_name].shape[1] > 3:
                    if "Time_s" in self.dataframes[df_name].columns:
                        self.dataframes[df_name] = self.dataframes[df_name].drop(
                            columns="Time_s"
                        )
                    self.treat_interface_dataframes(df_name)
                    self.melt_interface_dataframes(df_name)
                else:
                    self.dataframes[df_name].rename(
                        columns={"length": transform_column}, inplace=True
                    )
                self.arc_electric_features_max_min(df_name, transform_column)
                self.max_min_dataframe(df_name, transform_column)

        self.treat_merging_columns(self.parameters_variables)
        self.dataframes["dark_current"] = self.merge_dataframes(
            self.parameters_variables,
            [
                df
                for df in self.list_merge_dataframes
                if df in self.dataframes_to_process
            ],
        )
        self.dataframes["dark_current"]["Time_s"] = self.dataframes["dark_current"][
            "Time_s"
        ].astype("int")
        if "S" in self.parameters_variables:
            self.dataframes["dark_current"].sort_values(by="S", inplace=True)
            # Normalize values
            self.dataframes["dark_current"] = normalize_variables(
                self.dataframes["dark_current"], ["S"], ["Current (A)", "Charge"]
            )

        # Merge 1D dataframes
        self.merge_1D_dataframes(self.parameters_variables_1D, "down")
        self.merge_1D_dataframes(self.parameters_variables_1D, "up")

        # Merge 2D steady surface dataframes
        self.merge_2D_dataframes()

        # Remove tail of surface data
        if self.modifier == "_no_tail":
            self.remove_arc_tail()

        # Save treated dataframes
        if self.save:
            self.save_dataframes()

    def save_dataframes(self):
        """Save treated dataframes to parquet files"""
        for df_name, path in self.save_paths.items():
            if df_name in self.dataframes:
                self.dataframes[df_name].to_parquet(path, index=False)

    def load_dataframes(self):
        """Load saved dataframes if available"""
        try:
            for df_name, path in self.save_paths.items():
                print(f"Loading {df_name}...")
                self.dataframes[df_name] = pd.read_parquet(path)
            print("Dataframes loaded successfully from saved files.")
        except FileNotFoundError:
            print("Saved dataframes not found, processing dataframes...")
            self.process_dataframes()

    def remove_arc_tail(self):
        """
        Removes the tail of surface data
        """
        for i_surface in [
            "surface_charge_density_down",
            "surface_charge_density_up",
        ]:
            print(
                f"Removing tail from {i_surface}: initial shape {self.dataframes[i_surface].shape}"
            )
            self.dataframes[i_surface]["Arc_norm"] = (
                self.dataframes[i_surface]["Arc"]
                / self.dataframes[i_surface]["Arc"].max()
            )
            self.dataframes[i_surface] = self.dataframes[i_surface][
                self.dataframes[i_surface]["Arc_norm"] <= 0.95
            ]
            self.dataframes[i_surface].reset_index(drop=True, inplace=True)
            self.dataframes[i_surface].drop(columns=["Arc_norm"], inplace=True)
            print(
                f"Removing tail from {i_surface}: final shape {self.dataframes[i_surface].shape}"
            )

    def merge_1D_dataframes(self, merging_columns, surface_side):
        """
        Merge dataframes of surface into surface charge density dataframe
        """
        for dataframe in self.list_merge_1D_dataframes[surface_side]:
            if dataframe in self.dataframes:
                if f"surface_charge_density_{surface_side}" in self.dataframes:
                    self.dataframes[
                        f"surface_charge_density_{surface_side}"
                    ] = self.dataframes[f"surface_charge_density_{surface_side}"].merge(
                        self.dataframes[f"{dataframe}"], on=merging_columns, how="left"
                    )
                else:
                    self.dataframes[f"surface_charge_density_{surface_side}"] = (
                        self.dataframes[f"{dataframe}"]
                    )

    def merge_2D_dataframes(self):
        """
        Merge all 2D dataframes "gas_current_JGr_steady", "gas_current_JGz_steady",
        "gas_volume_Ez_steady" and "gas_volume_Er_steady"
        """
        for dataframe in [
            "gas_current_JGz_steady",
            "gas_volume_Er_steady",
            "gas_volume_Ez_steady",
        ]:
            if dataframe in self.dataframes:
                self.dataframes["gas_current_JGr_steady"] = self.dataframes[
                    "gas_current_JGr_steady"
                ].merge(
                    self.dataframes[dataframe],
                    on=["r", "z"],
                    how="left",
                )

    def treat_merging_columns(self, merging_columns):
        for dataframe in self.list_merge_dataframes + list(
            chain.from_iterable(self.list_merge_1D_dataframes.values())
        ):
            if dataframe in self.dataframes:
                for column in merging_columns:
                    if column not in self.dataframes[dataframe].columns:
                        print(
                            f"Adding column '{column}' to dataframe '{dataframe}' with default value 0"
                        )
                        self.dataframes[dataframe][column] = 0
                self.dataframes[dataframe][merging_columns] = self.dataframes[
                    dataframe
                ][merging_columns].astype("int")

    def max_min_dataframe(self, dataframe_name, column_name):
        self.dataframes[f"{dataframe_name}_min_max"] = self.dataframes[
            dataframe_name
        ].drop_duplicates(subset=self.parameters_variables)[
            self.parameters_variables
            + [
                f"{column_name}_min",
                f"{column_name}_max",
                f"{column_name}_mean",
                f"{column_name}_median",
            ]
        ]
        if f"{dataframe_name}_min_max" not in self.list_merge_dataframes:
            self.list_merge_dataframes.append(f"{dataframe_name}_min_max")
            if dataframe_name in self.list_merge_dataframes:
                self.list_merge_dataframes.remove(dataframe_name)
        self.dataframes_to_process.append(f"{dataframe_name}_min_max")

    def melt_interface_dataframes(self, dataframe_name):
        """
        Melt the dataframe to have a long format, with the conditions of the simulation as
        columns
        """
        self.dataframes[dataframe_name] = self.dataframes[dataframe_name].melt(
            id_vars="Arc",
        )
        self.dataframes[dataframe_name][self.parameters_variables] = self.dataframes[
            dataframe_name
        ]["variable"].str.split(";", expand=True)
        self.dataframes[dataframe_name][self.parameters_variables] = self.dataframes[
            dataframe_name
        ][self.parameters_variables].map(lambda a: int(float(a)))

        self.dataframes[dataframe_name].drop(columns="variable", inplace=True)
        self.dataframes[dataframe_name].rename(
            columns={"value": dataframe_name}, inplace=True
        )

    def treat_interface_dataframes(self, dataframe_name):
        """
        Interface dataframes have many columns, as the results for each condition
        simulated is appended as a column. This function treats the dataframe, so the name
        of the columns are the conditions of the simulation
        """
        df_condition = self.dataframes["dark_current"][
            self.parameters_variables
        ].drop_duplicates(self.parameters_variables)
        df_condition["column_name"] = df_condition.apply(
            lambda row: ";".join(
                str(row[param]) for param in self.parameters_variables
            ),
            axis=1,
        )

        self.dataframes[dataframe_name].columns = [
            self.dataframes[dataframe_name].columns[0]
        ] + df_condition["column_name"].tolist()

    def correct_names(self):
        # Correct columns of electric
        for dataframe_name in [
            "dark_current",
            "dark_currents_hv",
            "dark_currents_sld_hv",
            "dark_currents_sld_grd",
        ]:
            current_column_name = {
                "dark_current": "Current (A)",
                "dark_currents_hv": "Current (A) HV",
                "dark_currents_sld_hv": "Current (A) SLD HV",
                "dark_currents_sld_grd": "Current (A) SLD GRD",
            }
            if dataframe_name in self.dataframes:
                self.dataframes[dataframe_name].dropna(axis=1, inplace=True)
                if self.dataframes[dataframe_name].shape[1] == 2:
                    self.dataframes[dataframe_name] = self.dataframes[
                        dataframe_name
                    ].rename(
                        columns={
                            "Time": "Time_s",
                            "(s)": current_column_name[dataframe_name],
                        }
                    )
                else:
                    self.dataframes[dataframe_name] = self.dataframes[
                        dataframe_name
                    ].rename(
                        columns={
                            "(1/(m^3*s))": "Udc_kV",
                            "Udc": "Time_s",
                            "(kV)": current_column_name[dataframe_name],
                        }
                    )

        if "volumetric_charge" in self.dataframes:
            self.dataframes["volumetric_charge"].dropna(axis=1, inplace=True)
            if self.dataframes["volumetric_charge"].shape[1] == 2:
                self.dataframes["volumetric_charge"] = self.dataframes[
                    "volumetric_charge"
                ].rename(columns={"Time": "Time_s", "(s)": "Charge"})
            else:
                self.dataframes["volumetric_charge"] = self.dataframes[
                    "volumetric_charge"
                ].rename(
                    columns={"(1/(m^3*s))": "Udc_kV", "Udc": "Time_s", "(kV)": "Charge"}
                )

        if "tds_stats" in self.dataframes:
            self.dataframes["tds_stats"].dropna(axis=1, inplace=True)
            if self.dataframes["tds_stats"].shape[1] == 5:
                self.dataframes["tds_stats"] = self.dataframes["tds_stats"].rename(
                    columns={
                        "Time": "Time_s",
                        "(s)": "concentration_positive",
                        "Concentration": "concentration_negative",
                        "positive": "ion_generation",
                        "(mol)": "ion_recombination",
                    }
                )
            else:
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

        for e_direction in ["Ez", "Er"]:
            key = f"gas_volume_{e_direction}_steady"
            if key in self.dataframes:
                self.dataframes[key].dropna(axis=1, inplace=True)
                self.dataframes[key] = self.dataframes[key].rename(
                    columns={
                        "Electric": f"{e_direction}",
                    }
                )
        for current_direction in ["JGz", "JGr"]:
            key = f"gas_current_{current_direction}_steady"
            key_dict = {"JGz": "Color", "JGr": "Gas"}
            if key in self.dataframes:
                self.dataframes[key].dropna(axis=1, inplace=True)
                self.dataframes[key] = self.dataframes[key].rename(
                    columns={
                        key_dict[current_direction]: f"{current_direction}",
                    }
                )
        for current_type in ["avg", "min", "max"]:
            # This is volume information in the gas
            key = f"E_current_{current_type}"
            if key in self.dataframes:
                self.dataframes[key].dropna(axis=1, inplace=True)
                if self.dataframes[key].shape[1] == 5:
                    self.dataframes[key] = self.dataframes[key].rename(
                        columns={
                            "Time": "Time_s",
                            "(s)": f"electric_field_norm_{current_type}",
                            (
                                "Electric" if current_type != "min" else "ec.normE"
                            ): f"current_density_norm_{current_type}",
                            (
                                "field" if current_type != "min" else "(V/m)"
                            ): f"gas_current_density_norm_{current_type}",
                            (
                                "norm" if current_type != "min" else "ec.normJ"
                            ): f"peclet_{current_type}",
                        }
                    )
                if self.dataframes[key].shape[1] == 3:
                    self.dataframes[key] = self.dataframes[key].rename(
                        columns={
                            "Electric": f"electric_field_norm_{current_type}",
                            "field": f"electric_field_z_{current_type}",
                            "norm": f"electric_field_r_{current_type}",
                        }
                    )
                else:
                    self.dataframes[key] = self.dataframes[key].rename(
                        columns={
                            "(1/(m^3*s))": "Udc_kV",
                            "Udc": "Time_s",
                            "(kV)": f"electric_field_norm_{current_type}",
                            "Time": f"current_density_norm_{current_type}",
                            "(s)": f"gas_current_density_norm_{current_type}",
                            (
                                "Electric" if current_type != "min" else "ec3.normE"
                            ): f"peclet_{current_type}",
                        }
                    )
        for interface_loc in ["cathode", "anode", "downinterface", "upinterface"]:
            key = f"max_electric_field_{interface_loc}"
            if key in self.dataframes:
                self.dataframes[key].dropna(axis=1, inplace=True)
                if self.dataframes[key].shape[1] == 3:
                    self.dataframes[key] = self.dataframes[key].rename(
                        columns={
                            "Electric": f"max_electric_field_norm_{interface_loc}",
                            "field": f"max_Er_{interface_loc}",
                            "norm": f"max_Ez_{interface_loc}",
                        }
                    )
                else:
                    self.dataframes[key] = self.dataframes[key].rename(
                        columns={
                            "Time": "Time_s",
                            "(s)": f"max_electric_field_norm_{interface_loc}",
                            "Electric": f"max_Er_{interface_loc}",
                            "field": f"max_Ez_{interface_loc}",
                        }
                    )

        if "interface_flow" in self.dataframes:
            self.dataframes["interface_flow"] = self.dataframes[
                "interface_flow"
            ].rename(
                columns={
                    "(1/(m^3*s))": "Udc_kV",
                    "Udc": "Time_s",
                    "(kV)": "Current (A)_interface",
                }
            )

        if "top_boundary_flow" in self.dataframes:
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
            .groupby(self.parameters_variables)[transform_columns]
            .agg(["max", "min", "mean", "median"])
            .reset_index(),
            on=self.parameters_variables,
        )
        self.dataframes[dataframe].rename(
            columns={
                "max": f"{transform_columns}_max",
                "min": f"{transform_columns}_min",
                "mean": f"{transform_columns}_mean",
                "median": f"{transform_columns}_median",
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
