import pandas as pd
from src.utils.load_data import load_train_data
from src.conicalInsulator import insert_parameters, SteadyState
import pickle


def load_all_S_Udc(dataset, modifier="", folder_path="Folder not defined"):
    """
    Load the surface charge density datasets
    """
    list_S = [
        10000,
        40000,
        50000,
        80000,
        100000,
        200000,
    ]
    list_Udc = []
    for i in range(20):
        value = str(15 + i * (500 - 15) // 19)
        list_Udc.append(int(value.split(".")[0]))  # Ensure only integers are passed

    # Initialize an empty list to collect DataFrames
    list_df_all = []
    for S in list_S:
        for Udc_kV in list_Udc:
            df = load_train_data(
                dataset, S=S, voltage=Udc_kV, modifier=modifier, folder_path=folder_path
            )
            df[["S", "Udc_kV"]] = S, Udc_kV
            list_df_all.append(df)

    df_all = pd.concat(list_df_all, ignore_index=True)
    if "Time_s" in df_all.columns:
        df_all.sort_values(by="Time_s", inplace=True)
    return df_all


def steadyState_all_S_Udc(dataset, modifier="", folder_path="Folder not defined"):
    """
    Load the surface charge density datasets
    """
    list_S = [10000, 40000, 50000, 80000, 100000, 200000]  #
    list_Udc = []
    for i in range(20):
        value = str(15 + i * (500 - 15) // 19)
        list_Udc.append(int(value.split(".")[0]))  # Ensure only integers are passed

    # Initialize an empty list to collect DataFrames
    list_df_all = []
    for S in list_S:
        for Udc_kV in list_Udc:
            df = load_train_data(
                dataset, S=S, voltage=Udc_kV, modifier=modifier, folder_path=folder_path
            )
            steady_state = SteadyState(df, "Time_s", dataset, ["Arc"])
            steady_state.get_steady_state_times()
            df_steady_state = pd.DataFrame.from_dict(steady_state.steady_state_bygroup)
            df_steady_state[["S", "Udc_kV"]] = S, Udc_kV
            list_df_all.append(df_steady_state)
            print("finished processing S = ", S, " and Udc = ", Udc_kV)

    df_steady_state_all = pd.concat(list_df_all, ignore_index=True)
    return df_steady_state_all


def steadyState_data_surface(
    interface="up", modifier="", folder_path="Folder not defined"
):
    """
    Concatenate all datasets of varying S and Udc
    """
    if interface == "up":
        df_surface = steadyState_all_S_Udc(
            "surface_charge_density_up", modifier, folder_path=folder_path
        )
    elif interface == "down":
        df_surface = steadyState_all_S_Udc(
            "surface_charge_density_down", modifier, folder_path=folder_path
        )
    df_surface.to_parquet(
        f"{folder_path}Processed{modifier}/df_surface_{interface}{modifier}_steadyState.parquet",
        index=False,
    )


def treat_data_surface(interface="up", modifier="", folder_path="Folder not defined"):
    """
    Concatenate all datasets of varying S and Udc
    """
    if interface == "up":
        df_surface = load_all_S_Udc(
            "surface_charge_density_up", modifier, folder_path=folder_path
        )
        name_dataset = "surface_charge_density_up"
    elif interface == "down":
        df_surface = load_all_S_Udc(
            "surface_charge_density_down", modifier, folder_path=folder_path
        )
        name_dataset = "surface_charge_density_down"
    elif interface == "2D":
        df_surface = load_all_S_Udc("gas_current_JGr_steady", folder_path=folder_path)
        name_dataset = "surface_2D_steady"

    df_surface.to_parquet(
        f"{folder_path}Processed{modifier}/{name_dataset}{modifier}_all.parquet",
        index=False,
    )


def transformation_surface_data(df, dataset):
    """
    Unit transformation
    """
    if dataset == "surface_charge_density_up":
        suffix = "up"
    elif dataset == "surface_charge_density_down":
        suffix = "down"
    # To pico coulomb
    df[f"surface_charge_density_{suffix}"] = (
        df[f"surface_charge_density_{suffix}"] * 1e12
    )
    # Time in hours
    df["Time_s"] = df["Time_s"] / 3600
    # Electric field in kV/cm
    if f"surface_Ez_{suffix}" in df.columns:
        df[f"surface_Ez_{suffix}"] = df[f"surface_Ez_{suffix}"] / 1e3
    if f"surface_Er_{suffix}" in df.columns:
        df[f"surface_Er_{suffix}"] = df[f"surface_Er_{suffix}"] / 1e3
    return df
