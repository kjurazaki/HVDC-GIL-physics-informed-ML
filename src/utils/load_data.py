import pandas as pd
import pickle
import pyarrow.parquet as pq
import numpy as np


# Load train data for SINDy - surface charge density
def load_train_data(
    dataset, S=None, voltage=None, modifier="", folder_path="Folder not defined"
):
    train_data = {
        "dark_currents": f"{folder_path}Processed{modifier}/dark_currents{modifier}.parquet",
        "dark_currents_es_solution": f"{folder_path}Processed{modifier}/dark_currents_es_solution.parquet",
        "surface_charge_density_down": f"{folder_path}Processed{modifier}/surface_charge_density_down{modifier}_{S}S_{voltage}V.parquet",
        "surface_charge_density_up": f"{folder_path}Processed{modifier}/surface_charge_density_up{modifier}_{S}S_{voltage}V.parquet",
        "gas_current_JGr_steady": f"{folder_path}Processed{modifier}/surface_2D_steady{modifier}_{S}S_{voltage}V.parquet",
        "surface_2D_steady_all": f"{folder_path}Processed{modifier}/surface_2D_steady{modifier}_all.parquet",
    }
    print(f"loading from: {train_data[dataset]}")
    df = pd.read_parquet(train_data[dataset])
    return df


def load_data_surface_all(
    interface="up",
    step_size=10,
    columns_to_read=None,
    steadyState=False,
    folder_path="Folder not defined",
    modifier="",
):
    """
    Load the surface charge density concatenated datasets - interface can be "up" or "down"
    and resample the data based on the step_size.
    """
    if steadyState:
        file_path = (
            f"{folder_path}Processed{modifier}/df_surface_up{modifier}_steadyState.parquet"
            if interface == "up"
            else f"{folder_path}Processed{modifier}/df_surface_down{modifier}_steadyState.parquet"
        )
    else:
        file_path = (
            f"{folder_path}Processed{modifier}/surface_charge_density_up{modifier}_all.parquet"
            if interface == "up"
            else f"{folder_path}Processed{modifier}/surface_charge_density_down{modifier}_all.parquet"
        )
    print(f"loading from: {file_path}")
    # Read the unique arc values
    df = pd.read_parquet(file_path, columns=["Arc"])
    unique_arc_values = np.sort(df["Arc"].unique())
    # Resample data in "Arc" dimension - there are 800 data points, step_size divide that
    resampled_arc_values = unique_arc_values[::step_size]

    # Use pyarrow to filter the parquet file and read only specific columns
    if columns_to_read is not None:
        table = pq.read_table(
            file_path,
            filters=[("Arc", "in", resampled_arc_values)],
            columns=columns_to_read,
        )
    else:
        table = pq.read_table(file_path, filters=[("Arc", "in", resampled_arc_values)])
    df_surface = table.to_pandas()

    return df_surface


def load_sindy_data(interface: str, modifier: str = ""):
    """
    Load the sindy data
    """
    # Surface data
    file_name_xdot = f"./output_data/df_surface{modifier}_{interface}_x_dot_list.pkl"
    file_name_x = f"./output_data/df_surface{modifier}_{interface}_x_list.pkl"
    file_name_t = f"./output_data/df_surface{modifier}_{interface}_t_list.pkl"

    print(f"loading xdot from: {file_name_xdot}")
    print(f"loading x from: {file_name_x}")
    print(f"loading t from: {file_name_t}")
    with open(file_name_xdot, "rb") as file:
        xdot = pickle.load(file)

    with open(file_name_x, "rb") as file:
        x = pickle.load(file)

    with open(file_name_t, "rb") as file:
        t = pickle.load(file)

    return xdot, x, t
