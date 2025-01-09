import numpy as np

import matplotlib.pyplot as plt
from src.utils.load_data import load_sindy_data
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from src.utils.load_data import load_data_surface_all
from src.datasetSurface import transformation_surface_data
from src.datasetSindy import process_iteration


def evaluate_sindy_surface(
    interface: str = "up", predict_function: callable = None
) -> None:
    """
    Compare derivatives found by SINDy with the analytical derivatives
    """
    # load data
    xdot, x, _ = load_sindy_data(interface)
    # Select first arc
    x_first = x[0]
    xdot_predicted = []
    for i in range(len(x_first)):
        xdot_predicted.append(predict_function(x_first[i]))
    print(xdot_predicted[0:5])
    print(xdot[0][0:5, 0])
    plt.scatter(xdot[0][:, 0], xdot_predicted)
    plt.show()


def plot_sindy_prediction(
    X,
    Y,
    units: str = "",
    top_limit=None,
    log_scale=True,
    ax=None,
    plot_equation=None,
    save_path=None,
    suffix_name="",
):
    # Calculate RMSE and R2
    rmse = root_mean_squared_error(X, Y)
    rrmse = rmse / np.mean(np.abs(X))
    mape = mean_absolute_percentage_error(X, Y)

    # Plotting
    # y = x
    ax.plot(
        [X.min(), X.max()],
        [X.min(), X.max()],
        color="black",
        linewidth=1,
        linestyle="dashed",
    )
    ax.scatter(X, Y, color="black", s=1)
    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")
    if top_limit:
        ax.set_ylim(0, top_limit)
        ax.set_xlim(0, top_limit)

    # Rotate x labels by 45 degrees to avoid overlapping
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_ylabel(f"Predicted [{units}]")
    ax.set_xlabel(f"Target [{units}]")

    # Display equation, RMSE, and R2 on the plot
    if plot_equation:
        # Determine the position for the text based on the data points
        x_pos = 0.75
        y_pos = 0.05
        ax.text(
            x_pos,
            y_pos,
            f"RMSE: {rmse:.2f}\nRRMSE: {rrmse:.2f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top" if y_pos == 0.95 else "bottom",
        )
    if save_path is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{save_path}\\{suffix_name}_sindy_prediction_{now}.png"
        plt.savefig(file_name, bbox_inches="tight", dpi=1200)


def insert_derivatives_dataframe(folder_path, step_size=40, modifier: str = ""):
    """
    Insert derivatives in the dataframe
    """
    interface = "up"
    df_surface = load_data_surface_all(
        interface=interface,
        step_size=step_size,
        folder_path=folder_path,
        modifier=modifier,
    )
    df_surface = transformation_surface_data(
        df_surface, f"surface_charge_density_{interface}"
    )

    grouped = df_surface.groupby(["Arc", "S", "Udc_kV"])
    total_groups = len(grouped)
    print(f"Number of groups: {total_groups}")
    processed_count = 0

    # Create a new column for x_fd_dot
    df_surface["x_fd_dot"] = np.nan

    for name, group in grouped:
        x, x_fd_dot, x_sfd_dot, t, var_iteration = process_iteration(
            name,
            group,
            [f"surface_charge_density_{interface}"],
            "Time_s",
            features_columns=["S"],
            dot_features_columns=[f"surface_charge_density_{interface}"],
        )

        # Assign x_fd_dot values back to the corresponding indices in df_surface - First value is left off of df_surface
        df_surface.loc[group.index[1:], "x_fd_dot"] = x_fd_dot[:, 0]

        processed_count += 1
        print(f"Processed {processed_count} groups")

    # Build new feature
    df_surface["S_time_V"] = (
        df_surface["S"] * df_surface["Time_s"] / df_surface["Udc_kV"]
    )
    return df_surface
