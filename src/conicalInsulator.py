from src.darkCurrents import DarkCurrents
from src.steadyState import SteadyState
import pandas as pd
import os
import copy


class ConicalInsulator:
    def __init__(self, voltage, S, run_config, modifier):
        print(run_config["save_paths"])
        self.run_config = run_config
        self.voltage = voltage
        self.S = S
        self.modifier = modifier
        self.S_scientific = f"{1000 * S:.0e}".replace("+0", "")
        self.files_names = run_config["file_names"]
        self.load_data()

    def load_data(
        self,
    ):
        self.data = DarkCurrents(
            self.run_config["folder_path"],
            self.files_names,
            retreat=self.run_config["retreat"],
            save=self.run_config["save"],
            save_paths=self.run_config["save_paths"],
            dataframes_to_process=self.files_names.keys(),
            list_merge_dataframes=self.run_config["list_merge_dataframes"],
            # List dataframes to merge on surface_charge_density_down and up
            list_merge_1D_dataframes=self.run_config["list_merge_1D_dataframes"],
            parameters_variables=["Time_s"],
            parameters_variables_1D=["Arc", "Time_s"],
            modifier=self.modifier,
        )  # Use retreat true if any changes in the files

    def final_dataframes(self):
        if self.modifier == "_no_tail":
            """
            Remove tail (Arc > 0.95) from dataframes
            """

        # Plot surface charges
        if "surface_charge_density_down" in self.files_names:
            self.df_surface_down = self.data.dataframes["surface_charge_density_down"]
            self.df_surface_down["Udc_kV"] = self.voltage
            self.df_surface_down["S"] = self.S
            steady_state_down = SteadyState(
                self.df_surface_down, "Time_s", "surface_charge_density_down", ["Arc"]
            )
            steady_state_down.get_steady_state_times()
            self.df_steady_state_down = pd.DataFrame.from_dict(
                steady_state_down.steady_state_bygroup
            )
        if "surface_charge_density_up" in self.files_names:
            self.df_surface_up = self.data.dataframes["surface_charge_density_up"]
            self.df_surface_up["Udc_kV"] = self.voltage
            self.df_surface_up["S"] = self.S
            steady_state_up = SteadyState(
                self.df_surface_up, "Time_s", "surface_charge_density_up", ["Arc"]
            )
            steady_state_up.get_steady_state_times()
            self.df_steady_state_up = pd.DataFrame.from_dict(
                steady_state_up.steady_state_bygroup
            )

        # Dark currents
        self.df_dark_current = self.data.dataframes["dark_current"]
        self.df_dark_current["Udc_kV"] = self.voltage
        self.df_dark_current["S"] = self.S

        if (
            "surface_charge_density_up" in self.files_names
            and "surface_charge_density_down" in self.files_names
        ):
            for key, value in {
                "up": steady_state_up,
                "down": steady_state_down,
            }.items():
                df_steady_state = pd.DataFrame.from_dict(value.steady_state_bygroup)
                self.df_dark_current[f"steady_state_time_max_{key}"] = df_steady_state[
                    "steady_state_time"
                ].max()
                self.df_dark_current[f"steady_state_value_max_{key}"] = df_steady_state[
                    "steady_state_value"
                ].max()
                self.df_dark_current[f"steady_state_time_min_{key}"] = df_steady_state[
                    "steady_state_time"
                ].min()
                self.df_dark_current[f"steady_state_value_min_{key}"] = df_steady_state[
                    "steady_state_value"
                ].min()


def insert_physical_constant(df):
    """
    Physical constants
    """
    df["e"] = 1.602e-19
    df["R"] = 8.314
    df["k_b"] = 1.381e-23
    df["N_A"] = 6.022e23
    return df


def insert_parameters(df):
    """
    Other than S and Udc_kV, simulation parameters
    """
    # First simulations parameters
    df["P"] = 0.4
    df["Rec"] = 6e-13
    df["mup"] = 4.8e-6
    df["mun"] = 4.8e-6
    df["Dp"] = 1.2e-7
    df["Dn"] = 1.2e-7
    df["er_epoxy"] = 5
    df["k_epoxy"] = 4.2e-17
    df["er_SF6"] = 1.002
    df["k_gas"] = 1e-19
    return df


def process_conical_insulator(run_config, modifier=""):
    """
    Process insulator data and save to parquet
    """
    processed_dir = f"C:/Users/kenji/OneDrive/2024/Research DII006/sindy_conical_ML/database_comsol_ml/Processed{modifier}/"
    dark_currents_path = f"{processed_dir}dark_currents{modifier}.parquet"
    print(f"reading: {dark_currents_path}")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if os.path.exists(dark_currents_path):
        df = pd.read_parquet(dark_currents_path)
        print(f"Initial shape: {df.shape}")
    else:
        df = pd.DataFrame()
    old_save_paths = run_config["save_paths"]
    if modifier != "":
        run_config["save_paths"] = {
            key: f"{old_save_paths[key].split('.parquet')[0]}{modifier}.parquet".replace(
                "Processed", f"Processed{modifier}"
            )
            for key, path in old_save_paths.items()
        }
    old_save_paths = run_config["save_paths"]
    print(f"run_config['save_paths']: {run_config['save_paths']}")
    vValues = [str(15 + i * (500 - 15) // 19) for i in range(20)]
    for voltage in vValues:
        for S in [
            10000,
            40000,
            50000,
            80000,
            100000,
            200000,
        ]:  # Already processed 10000, 40000, 50000, 80000,100000, 200000
            S_scientific = f"{1000 * S:.0e}".replace("+0", "")
            run_config["file_names"] = {
                "dark_current": [
                    f"dark_currents_grd_S{S_scientific}_V{voltage}.txt",
                    4,
                ],  # Computes only ground in v1
                "dark_currents_hv": [
                    f"dark_currents_hv_S{S_scientific}_V{voltage}.txt",
                    4,
                ],  # High voltage
                "dark_currents_sld_hv": [
                    f"dark_currents_sld_hv_S{S_scientific}_V{voltage}.txt",
                    4,
                ],  # solid and HV contact
                "dark_currents_sld_grd": [
                    f"dark_currents_sld_grd_S{S_scientific}_V{voltage}.txt",
                    4,
                ],  # solid and grounded contact
                "volumetric_charge": [
                    f"volumetric_charge_S{S_scientific}_V{voltage}.txt",
                    4,
                ],
                "tds_stats": [f"tds_stats_S{S_scientific}_V{voltage}.txt", 4],
                "E_current_avg": [f"e_current_avg_S{S_scientific}_V{voltage}.txt", 4],
                "E_current_min": [f"e_current_min_S{S_scientific}_V{voltage}.txt", 4],
                "E_current_max": [f"e_current_max_S{S_scientific}_V{voltage}.txt", 4],
                # Don't need this dataframe as the maximum is computed from the surface_Ez_up_min_max and others dataframes
                # "max_electric_field_anode": [
                #     f"maximum_electric_field_anode_S{S_scientific}_V{voltage}.txt",
                #     4,
                # ],
                # "max_electric_field_cathode": [
                #     f"maximum_electric_field_cathode_S{S_scientific}_V{voltage}.txt",
                #     4,
                # ],
                # "max_electric_field_downinterface": [
                #     f"maximum_electric_field_downInterface_S{S_scientific}_V{voltage}.txt",
                #     4,
                # ],
                # "max_electric_field_upinterface": [
                #     f"maximum_electric_field_upInterface_S{S_scientific}_V{voltage}.txt",
                #     4,
                # ],
                # Boundary data
                "surface_charge_density_down": [
                    f"surfaceChargeDown_S{S}_V{voltage}.txt",
                    7,
                ],
                "surface_charge_density_up": [
                    f"surfaceChargeUp_S{S}_V{voltage}.txt",
                    7,
                ],
                "surface_Ez_down": [
                    f"surfaceEzDown_S{S}_V{voltage}.txt",
                    7,
                ],
                "surface_Ez_up": [
                    f"surfaceEzUp_S{S}_V{voltage}.txt",
                    7,
                ],
                "surface_Er_down": [
                    f"surfaceErDown_S{S}_V{voltage}.txt",
                    7,
                ],
                "surface_Er_up": [
                    f"surfaceErUp_S{S}_V{voltage}.txt",
                    7,
                ],
                "HV_Er": [
                    f"HV_Er_S{S}_V{voltage}.txt",
                    7,
                ],
                "Grd_Er": [
                    f"Grd_Er_S{S}_V{voltage}.txt",
                    7,
                ],
                "Grd_Ez": [
                    f"Grd_Ez_S{S}_V{voltage}.txt",
                    7,
                ],
                ## Surface data (at steady state)
                # "concentration_negative_steady": [
                #     f"cn_lastt_S{S}_V{voltage}.txt",
                #     7,
                # ],
                # "concentration_positive_steady": [
                #     f"cp_lastt_S{S}_V{voltage}.txt",
                #     7,
                # ],
                "gas_current_JGr_steady": [
                    f"gas_currentJGr__lastt_S{S}_V{voltage}.txt",
                    7,
                ],
                "gas_current_JGz_steady": [
                    f"gas_currentJGz__lastt_S{S}_V{voltage}.txt",
                    7,
                ],
                "gas_volume_Er_steady": [
                    f"gas_volumeEr__lastt_S{S}_V{voltage}.txt",
                    7,
                ],
                "gas_volume_Ez_steady": [
                    f"gas_volumeEz_lastt_S{S}_V{voltage}.txt",
                    7,
                ],
                # "solid_volume_Er_steady": [
                #     f"solid_volumeEr__lastt_S{S}_V{voltage}.txt",
                #     7,
                # ],
                # "solid_volume_Ez_steady": [
                #     f"solid_volumeEz_lastt_S{S}_V{voltage}.txt",
                #     7,
                # ],
            }
            run_config["save_paths"] = {
                key: f"{old_save_paths[key].split('.parquet')[0]}_{S}S_{voltage}V.parquet"
                for key, path in old_save_paths.items()
            }
            conical_insulator = ConicalInsulator(voltage, S, run_config, modifier)
            conical_insulator.final_dataframes()
            df = pd.concat([df, conical_insulator.df_dark_current])
    df.to_parquet(dark_currents_path)
    print(f"Final shape: {df.shape}")


def process_es_conical_insulator(run_config):
    """
    Append data from electroquasistatic simulations
    """
    dark_currents_path = "C:/Users/kenji/OneDrive/2024/Research DII006/sindy_conical_ML/database_comsol_ml/Processed/dark_currents_es_solution.parquet"
    if os.path.exists(dark_currents_path):
        df = pd.read_parquet(dark_currents_path)
        print(f"Initial shape: {df.shape}")
    else:
        df = pd.DataFrame()
    vValues = [str(15 + i * (500 - 15) // 19) for i in range(20)]
    old_save_paths = run_config["save_paths"]
    for voltage in vValues:
        run_config["file_names"] = {
            "E_current_avg": [f"ES_e_current_avg_V{voltage}.txt", 4],
            "E_current_min": [
                f"ES_e_current_min_V{voltage}.txt",
                4,
            ],
            "E_current_max": [
                f"ES_e_current_max_V{voltage}.txt",
                4,
            ],
            "max_electric_field_anode": [
                f"ES_maximum_electric_field_anode_V{voltage}.txt",
                4,
            ],
            "max_electric_field_cathode": [
                f"ES_maximum_electric_field_cathode_V{voltage}.txt",
                4,
            ],
            "max_electric_field_downinterface": [
                f"ES_maximum_electric_field_downInterface_V{voltage}.txt",
                4,
            ],
            "max_electric_field_upinterface": [
                f"ES_maximum_electric_field_upInterface_V{voltage}.txt",
                4,
            ],
            "surface_Ez_down": [
                f"ES_surfaceEzDown_V{voltage}.txt",
                7,
            ],
            "surface_Ez_up": [
                f"ES_surfaceEzUp_V{voltage}.txt",
                7,
            ],
            "surface_Er_down": [
                f"ES_surfaceErDown_V{voltage}.txt",
                7,
            ],
            "surface_Er_up": [
                f"ES_surfaceErUp_V{voltage}.txt",
                7,
            ],
            "HV_Er": [
                f"ES_HV_Er_V{voltage}.txt",
                7,
            ],
            "Grd_Er": [
                f"ES_Grd_Er_V{voltage}.txt",
                7,
            ],
            "Grd_Ez": [
                f"ES_Grd_Ez_V{voltage}.txt",
                7,
            ],
        }
        run_config["save_paths"] = {
            key: f"{old_save_paths[key].split('.parquet')[0]}_{voltage}V.parquet"
            for key, path in old_save_paths.items()
        }
        conical_insulator = ConicalInsulator(
            voltage, S=0, run_config=copy.deepcopy(run_config)
        )
        conical_insulator.final_dataframes()
        df = pd.concat([df, conical_insulator.df_dark_current])
    df.to_parquet(dark_currents_path)
    print(f"Final shape: {df.shape}")


def load_conical_comsol_parameters():
    """
    Data related to the study [[dark_currents_TDS]], check obsidian for more context
    """
    # Volume of the gas chamber computed on COMSOL m^3
    volume_of_chamber = 0.06195758296701219
    # Volume of the insulator m^3
    volume_of_insulator = 0.006349421463343287
    # Area of the electrodes that the gas ions are integrated m^2
    area_dark_currents = 0.7899908134150739
    # Area of the interface 1 [m^2]
    area_interface_insulator_gas = 0.21054
    # length interface 1 [m]
    length_interface = 0.19877
    # Area of the interface 2 [m^2]
    area_interface_insulator_gas_2 = 0.21410
    # length interface 2 [m]
    length_interface_2 = 0.19882
    # Lenght of high voltage electrode [m] interface with gas
    length_high_voltage_electrode = 0.36987
    # Area of high voltage electrode [m^2] interface with gas
    area_high_voltage_electrode = 0.20916
    # Lenght of grounded electrode [m] interface with gas
    length_grounded_electrode = 0.36987
    # Area of grounded electrode [m^2] interface with gas
    area_grounded_electrode = 0.58084
    # Elementary charge C
    e = 1.602176634e-19

    return {
        "volume_of_chamber": volume_of_chamber,
        "volume_of_insulator": volume_of_insulator,
        "area_dark_currents": area_dark_currents,
        "area_interface_insulator_gas": area_interface_insulator_gas,
        "length_interface": length_interface,
        "area_interface_insulator_gas_2": area_interface_insulator_gas_2,
        "length_interface_2": length_interface_2,
        "length_high_voltage_electrode": length_high_voltage_electrode,
        "area_high_voltage_electrode": area_high_voltage_electrode,
        "length_grounded_electrode": length_grounded_electrode,
        "area_grounded_electrode": area_grounded_electrode,
        "e": e,
    }
