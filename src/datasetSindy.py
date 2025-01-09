import pandas as pd
import numpy as np
import pickle
from src.conicalInsulator import insert_physical_constant, insert_parameters
from src.utils.load_data import load_train_data
from pysindy.differentiation import FiniteDifference, SmoothedFiniteDifference
from src.datasetSurface import transformation_surface_data
from src.utils.load_data import load_data_surface_all
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split


def compute_x_dot(x, t, finite_difference=True, smooth_finite_difference=False):
    """
    Differentiate x with respect to timestep
    """
    if finite_difference:
        # Forward difference (order = 1)
        compute_fd = FiniteDifference(order=1)
        x_dot = compute_fd._differentiate(x, t)
    elif smooth_finite_difference:
        compute_sfd = SmoothedFiniteDifference(
            smoother_kws={"window_length": 5}, order=1
        )
        x_dot = compute_sfd._differentiate(x, t)
    return x_dot


def process_iteration(
    name, group, x_column, t_column, features_columns, dot_features_columns
):
    """
    Process one iteration of the data
    Inserts the derivatives of dot_features_columns into the final x list
    """
    x = group[x_column + dot_features_columns].reset_index(drop=True).to_numpy()
    t = group[t_column].reset_index(drop=True).to_numpy()
    x_fd_dot = compute_x_dot(
        x, t, finite_difference=True, smooth_finite_difference=False
    )
    x_sfd_dot = compute_x_dot(
        x, t, finite_difference=False, smooth_finite_difference=True
    )
    return (
        np.hstack(
            (group[features_columns].to_numpy()[1:, :], x_fd_dot[:-1, len(x_column) :])
        ),
        x_fd_dot[1:, : len(x_column)],
        x_sfd_dot[1:, : len(x_column)],
        t[1:],
        name,
    )


def process_sindy_data(
    df,
    x_column,
    t_column,
    features_columns,
    dot_features_columns,
    iterate_columns: list,
) -> dict:
    """
    df: dataframe
    iterate_columns: list of columns to iterate over
    """
    x_list = []
    x_dot_list = []
    t_list = []

    grouped = df.groupby(iterate_columns)
    total_groups = len(grouped)
    print(f"Number of groups: {total_groups}")
    processed_count = 0
    for name, group in grouped:
        x, x_fd_dot, x_sfd_dot, t, var_iteration = process_iteration(
            name, group, x_column, t_column, features_columns, dot_features_columns
        )
        x_list.append(x)
        x_dot_list.append(x_fd_dot)
        t_list.append(t)
        processed_count += 1
        print(f"Processed {processed_count} groups")

    return {
        "x_list": x_list,
        "x_dot_list": x_dot_list,
        "t_list": t_list,
    }


def scale_data_min_max(df, columns_to_scale: list):
    """
    Scale the data using MinMaxScaler from sklearn and save the scalers.
    """
    scalers = {}
    scaled_df = df.copy()

    for column in columns_to_scale:
        scaler = MinMaxScaler()
        scaled_df[column] = scaler.fit_transform(df[[column]])
        scalers[column] = scaler

    # Save the scalers
    for column, scaler in scalers.items():
        with open(f"./output_data/scaler_{column}.pkl", "wb") as f:
            pickle.dump(scaler, f)
    return scaled_df


def transform_final_sindy(
    interface,
    features,
    dot_features,
    step_size: int,
    scaling: bool = False,
    modifier: str = "",
):
    """
    Transform the final sindy data

    parameters:
    - step_size: resamples the data in "Arc" dimension
    """
    df_surface = load_data_surface_all(
        interface=interface, step_size=step_size, modifier=modifier
    )
    df_surface = transformation_surface_data(
        df_surface, f"surface_charge_density_{interface}"
    )
    if scaling:
        df_surface = scale_data_min_max(df_surface, columns_to_scale=features)
    # Process data
    df_surface = process_sindy_data(
        df_surface,
        [f"surface_charge_density_{interface}", "S", "Udc_kV"],
        "Time_s",
        features_columns=features,
        dot_features_columns=dot_features,
        iterate_columns=["Arc", "S", "Udc_kV"],
    )
    # Scale the first column of x_dot_list using the existing scale_data_min_max function
    if scaling:
        x_dot_df = pd.DataFrame(
            np.concatenate(df_surface["x_dot_list"]).reshape(-1, 3),
            columns=["x_dot_1", "x_dot_2", "x_dot_3"],
        )
        x_dot_df = scale_data_min_max(x_dot_df, columns_to_scale=["x_dot_1"])
        # Transform x_dot_df back into a list of arrays
        df_surface["x_dot_list"] = np.split(
            x_dot_df.values, len(df_surface["x_dot_list"])
        )
    for key in df_surface.keys():
        with open(
            f"./output_data/df_surface{modifier}_{interface}_{key}.pkl", "wb"
        ) as f:
            pickle.dump(df_surface[key], f)
    return df_surface


def split_sindy_data(x, xdot, t):
    # Split the lists xdot, x, and t into train and test sets
    xdot, xdot_test, train_indices, test_indices = train_test_split(
        xdot, range(len(xdot)), test_size=0.01, random_state=13
    )
    x, x_test = [x[i] for i in train_indices], [x[i] for i in test_indices]
    t, t_test = [t[i] for i in train_indices], [t[i] for i in test_indices]
    return x, x_test, t, t_test, xdot, xdot_test


def normalize_columns_l2(x, xdot):
    """
    Normalize columns of x and xdot by L2 norm.
    """
    x_normalized = []
    xdot_normalized = []

    for xi, xdot_i in zip(x, xdot):
        xi_norm = np.linalg.norm(xi, 2, axis=0)
        xdot_i_norm = np.linalg.norm(xdot_i, 2, axis=0)

        # Avoid division by zero
        xi_norm[xi_norm == 0] = 1
        xdot_i_norm[xdot_i_norm == 0] = 1

        x_normalized.append(xi / xi_norm)
        xdot_normalized.append(xdot_i / xdot_i_norm)

    return x_normalized, xdot_normalized


def sindy_data(
    interface: str, step_size: int, scaling: bool = False, modifier: str = ""
) -> None:
    """
    Run data treatment for SINDy
    Save final dataframes to parquet file
    """
    features = [
        f"surface_charge_density_{interface}",
        "S",
        "Udc_kV",
        f"surface_Ez_{interface}",
        f"surface_Er_{interface}",
    ]
    dot_features = [
        f"surface_charge_density_{interface}",
        f"surface_Ez_{interface}",
        f"surface_Er_{interface}",
    ]
    transform_final_sindy(
        interface,
        features=features,
        dot_features=dot_features,
        step_size=step_size,
        scaling=scaling,
        modifier=modifier,
    )
