from src.utils.load_data import load_train_data, load_data_surface_all
from src.datasetSurface import transformation_surface_data
from datetime import datetime
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd


def load_plot_data_sr(save_path, modifier=""):
    """
    Plot targets data of the learning task of SR
    """
    ## Load data for SR
    df_dark_currents = load_train_data("dark_currents", modifier=modifier)
    df_dark_currents["Current (A) Total"] = (
        df_dark_currents["Current (A)"]
        + df_dark_currents["Current (A) HV"]
        + df_dark_currents["Current (A) SLD HV"]
        + df_dark_currents["Current (A) SLD GRD"]
    )
    df_dark_currents.rename(columns={"Current (A)": "Current (A) GRD"}, inplace=True)
    df_dark_currents["Current (A) TOTAL HV"] = (
        df_dark_currents["Current (A) HV"] + df_dark_currents["Current (A) SLD HV"]
    )
    df_dark_currents["Current (A) TOTAL GRD"] = (
        df_dark_currents["Current (A) GRD"] + df_dark_currents["Current (A) SLD GRD"]
    )
    # df_dark_currents_es = load_train_data("dark_currents_es_solution") Farady C/mol
    df_dark_currents["ion_charge_genegeration"] = (
        df_dark_currents["ion_generation"] * 96485
    )
    # df_dark_currents_es.drop(columns=["Time_s", "S"], inplace=True)
    # df_dark_currents_es = df_dark_currents_es.add_suffix("_es")
    plot_params = {
        "x_column": "Udc_kV",
        "y_column": "Current (A) GRD",
        "style_column": "S",
        "x_label": "Potential [kV]",
        "y_label": "Current [A]",
        "legend_title": "S [$10^6 \\times IP/m^3/s$]",
        "ncol": 3,
        "save_path": save_path,
    }

    if True:
        for current_column in [
            "Current (A) GRD",
            "Current (A) Total",
            "Current (A) HV",
            "Current (A) SLD HV",
            "Current (A) SLD GRD",
            "Current (A) TOTAL HV",
            "Current (A) TOTAL GRD",
            "ion_charge_genegeration",
        ]:
            plot_params["y_column"] = current_column
            # Dark currents
            scatter_plot_steadyState(
                df_dark_currents,
                **plot_params,
            )

    if True:
        for concentration_column in ["negative", "positive"]:
            plot_params["y_column"] = f"concentration_{concentration_column}"
            plot_params["y_label"] = (
                f"Ion concentration [{concentration_column}] [mol/m³]"
            )
            # Ion charge generation
            scatter_plot_steadyState(
                df_dark_currents,
                **plot_params,
            )

    if True:
        # Steady state time in hours
        df_dark_currents["steady_state_time_max_up"] = (
            df_dark_currents["steady_state_time_max_up"] / 3600
        )
        plot_params["y_column"] = "steady_state_time_max_up"
        plot_params["y_label"] = "Time to steady state [h]"
        scatter_plot_steadyState(
            df_dark_currents,
            **plot_params,
            y_scale="linear",
        )


def load_plot_data_surface(save_path: str, modifier=""):
    """
    Load and plot surface data
    """
    interface = "up"
    # Resample data in "Arc" dimension - there are 800 data points, step_size divide that
    step_size = 1
    columns_to_read = ["Arc", "Time_s", "S", "Udc_kV", "surface_charge_density_up"]
    df_surface = load_data_surface_all(
        interface=interface,
        step_size=step_size,
        columns_to_read=columns_to_read,
        modifier=modifier,
    )
    df_surface = transformation_surface_data(
        df_surface, f"surface_charge_density_{interface}"
    )
    df_surface["Arc"] = df_surface["Arc"] / df_surface["Arc"].max()

    df_surface_steadyState = load_data_surface_all(
        interface=interface,
        step_size=step_size,
        columns_to_read=None,
        steadyState=True,
        modifier=modifier,
    )
    df_surface_steadyState["steady_state_time"] = (
        df_surface_steadyState["steady_state_time"] / 3600
    )

    df_surface_steadyState["Arc"] = (
        df_surface_steadyState["Arc"] / df_surface_steadyState["Arc"].max()
    )
    # df_surface[np.abs(df_surface['surface_charge_density_up']) < 1E-1]

    # Sample only the min and max of S and Udc_kV
    min_S, max_S = df_surface_steadyState["S"].min(), df_surface_steadyState["S"].max()
    min_Udc_kV, max_Udc_kV = (
        df_surface_steadyState["Udc_kV"].min(),
        df_surface_steadyState["Udc_kV"].max(),
    )

    # Select potentials 15,  40,  66,  91, 117, 142, 168, 193, 219, 244, 270, 295, 321, 346, 372, 397, 423, 448, 474, 500
    df_sampled = df_surface_steadyState[
        (df_surface_steadyState["S"].isin([min_S, max_S]))
        & (df_surface_steadyState["Udc_kV"].isin([321, max_Udc_kV]))
    ].copy()

    # Plot steady state time vs arc in sampled conditon
    if True:
        plot_surface_time_vs_arc(
            df=df_sampled, save_path=save_path, suffix_name="sampled"
        )

    # Plot steady state value vs arc in sampled condition
    if True:
        plot_surface_time_vs_arc(
            df=df_sampled,
            y_column="steady_state_value",
            y_label="Surface charge density [C/m²]",
            save_path=save_path,
            suffix_name="sampled",
        )

    # Plot steady state time vs arc in specific S
    if True:
        df_sampled = df_surface_steadyState[df_surface_steadyState["S"] == 80000].copy()
        plot_surface_time_vs_arc(
            df=df_sampled,
            save_path=save_path,
            suffix_name="max_S",
            gray_scale=True,
            legend=False,
        )

    # Plot steady state time vs arc in specific Udc_kV
    if True:
        df_sampled = df_surface_steadyState[
            df_surface_steadyState["Udc_kV"] == 40
        ].copy()
        plot_surface_time_vs_arc(
            df=df_sampled,
            save_path=save_path,
            suffix_name="40_V",
            gray_scale=True,
            legend=False,
        )

    # Plot steady state value vs arc in sampled condition
    if True:
        df_sampled = df_surface_steadyState[df_surface_steadyState["S"] == 80000].copy()
        plot_surface_time_vs_arc(
            df=df_sampled,
            y_column="steady_state_value",
            y_label="Surface charge density [C/m²]",
            save_path=save_path,
            suffix_name="sampled",
        )
    if True:
        # Plot distribution of steady state value and time
        for y_column in ["steady_state_time", "steady_state_value"]:
            y_label = (
                "Steady State Time [h]"
                if y_column == "steady_state_time"
                else "Surface charge density [C/m²]"
            )
            plot_surface_distribution(
                df=df_surface_steadyState,
                y_column=y_column,
                y_label=y_label,
                save_path=save_path,
                suffix_name="",
            )
    # Plot specific arc length vs time
    if True:
        df_surface["surface_charge_density_up"] = (
            df_surface["surface_charge_density_up"] / 1e12
        )
        for specific_arc_length in [323, 10, 700, 750, 800]:
            try:
                specific_S = 3  # Up to 6
                specific_Udc_kV = 13  # up to 20

                arcs = list(df_surface["Arc"].sort_values().unique())
                Ss = list(df_surface["S"].sort_values().unique())
                Udc_kVs = list(df_surface["Udc_kV"].sort_values().unique())

                df_sampled = df_surface[
                    (df_surface["Arc"] == arcs[specific_arc_length])
                    & (df_surface["S"] == Ss[specific_S])
                    & (df_surface["Udc_kV"] == Udc_kVs[specific_Udc_kV])
                ].copy()
                plot_surface_time_vs_arc(
                    df=df_sampled,
                    x_column="Time_s",
                    x_label="Time [h]",
                    y_column="surface_charge_density_up",
                    y_label="Surface charge density [C/m²]",
                    save_path=save_path,
                    suffix_name=f"arc_length_{specific_arc_length}",
                    gray_scale=False,
                    legend=True,
                )
            except:
                pass
    # Plot distribution as violin in different arc lengths
    if True:
        arcs = list(df_surface_steadyState["Arc"].sort_values().unique())
        specific_arc_length = [10, 75, 150, 300, 450, 600, 700, 790]
        try:
            df_sampled = df_surface_steadyState[
                df_surface_steadyState["Arc"].isin(
                    [arcs[i] for i in specific_arc_length]
                )
            ].copy()
            plot_surface_violin(
                df=df_sampled,
                y_column="steady_state_time",
                y_label="Steady State Time [h]",
                save_path=save_path,
                fig_size=(12, 9),
                suffix_name="steady_state_time",
            )
            plot_surface_violin(
                df=df_sampled,
                y_column="steady_state_value",
                y_label="Surface charge density [C/m²]",
                save_path=save_path,
            )
        except:
            pass
    print("debug")


def plot_surface_violin(
    df,
    y_column,
    y_label,
    y_scale="linear",
    save_path=None,
    fig_size=(6, 4.5),
    suffix_name="",
):
    # Normalized arc
    df["Arc"] = df["Arc"] / df["Arc"].max()
    plt.figure(figsize=fig_size)
    arc_lengths = lambda x: f"{x:.2f}"
    sns.violinplot(x="Arc", y=y_column, data=df, formatter=arc_lengths)
    plt.xlabel("Arc length normalized", fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.yscale(y_scale)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{save_path}\\violin_plot_steady_state_time_vs_arc_{suffix_name}_{now}.pdf"
        )
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def plot_derivative(df_surface, save_path):
    """
    Plot derivative graphs
    """
    specific_arc_length = 2  # Example arc length selection from 0 to 800
    specific_S = 3  # Up to 6
    specific_Udc_kV = 13  # up to 20
    if True:
        arcs = list(df_surface["Arc"].sort_values().unique())
        Ss = list(df_surface["S"].sort_values().unique())
        Udc_kVs = list(df_surface["Udc_kV"].sort_values().unique())
        df_sampled = df_surface[
            (df_surface["Arc"] == arcs[specific_arc_length])
            & (df_surface["S"] == Ss[specific_S])
            & (df_surface["Udc_kV"] == Udc_kVs[specific_Udc_kV])
        ].copy()
        plot_surface_time_vs_arc(
            df_sampled,
            x_column="Time_s",
            x_label="Time [h]",
            y_column="x_fd_dot",
            y_label="Derivative",
            save_path=save_path,
            suffix_name="derivative",
            legend=False,
            gray_scale=False,
        )
    if True:
        specific_arc_length = 13  # Example arc length selection from 0 to 800
        arcs = list(df_surface["Arc"].sort_values().unique())
        df_sampled = df_surface[(df_surface["Arc"] == arcs[specific_arc_length])].copy()
        # Create a selection for the x-axis
        plot_params = {
            "x_column": "Udc_kV",
            "y_column": "x_fd_dot",
            "style_column": "S",
            "x_label": "Potential [kV]",
            "y_label": "Derivative",
            "legend_title": "S [$10^6 \\times IP/m^3/s$]",
            "ncol": 3,
            "save_path": save_path,
        }
        # Dark currents
        scatter_plot_steadyState(
            df_sampled,
            **plot_params,
        )


def plot_surface_distribution(
    df: pd.DataFrame, y_column: str, y_label: str, save_path: str, suffix_name: str = ""
):
    """
    Plot distribution
    """
    plt.close("all")
    plt.figure(figsize=(10, 6))
    plt.hist(df[y_column], bins=30, edgecolor="k", alpha=0.7)
    plt.xlabel(y_label)
    plt.ylabel("Frequency")
    if save_path is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}\\{y_column}_distribution_{suffix_name}_{now}.pdf"
        plt.savefig(filename, bbox_inches="tight")


def scatter_plot_steadyState(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    style_column: str,
    x_label: str,
    y_label: str,
    legend_title: str,
    ncol: int = 3,
    y_scale: str = "linear",
    save_path: str = None,
) -> None:
    # Dark current vs Udc and stylized by S
    df_plot = df[df["Time_s"] == df["Time_s"].max()].copy()
    df_plot["S"] = df_plot["S"] / 1e3
    df_plot["S"] = df_plot["S"].astype(int)
    plt.close("all")
    plt.figure(figsize=(6, 4.5))
    sns.scatterplot(
        data=df_plot,
        x=x_column,
        y=y_column,
        style=style_column,
        hue=style_column,
        palette=sns.color_palette("husl", df_plot[style_column].nunique()),
    )
    plt.yscale(y_scale)
    plt.tick_params(axis="both", which="major", labelsize=10)
    plt.gca().xaxis.set_major_locator(plt.LinearLocator(10))
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    if (y_column == "Current (A) GRD") or (y_column == "Current (A) Total"):
        plt.legend(
            title=legend_title,
            ncol=ncol,
            fontsize=10,
            title_fontsize=10,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
        )
    else:
        plt.legend(title=legend_title, ncol=ncol, fontsize=10, title_fontsize=10)
    if save_path is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}\\{y_column}_{x_column}_{style_column}_{now}.pdf"
        plt.savefig(filename, bbox_inches="tight")


def plot_surface_time_vs_arc(
    df: pd.DataFrame,
    x_column: str = "Arc",
    x_label: str = "Arc length normalized",
    y_column: str = "steady_state_time",
    y_label: str = "Steady State Time [h]",
    save_path: str = None,
    suffix_name: str = "",
    legend: bool = True,
    gray_scale: bool = False,
):
    """
    Plot steady state time vs arc
    """
    plt.close("all")
    # Plot lines with different hue based on S and Udc_kV
    plt.figure(figsize=(6, 4.5))

    # Style
    df["S&Udc_kv"] = df["S"].astype(str) + ", " + df["Udc_kV"].astype(str)

    if gray_scale:
        sns.lineplot(
            data=df,
            x=x_column,
            y=y_column,
            hue="S&Udc_kv",
            palette="rocket_r",
        )
    else:
        sns.lineplot(
            data=df,
            x=x_column,
            y=y_column,
            hue="S&Udc_kv",
            style="S&Udc_kv",
            palette=(
                sns.color_palette("Dark2", df["S&Udc_kv"].nunique())
                if df["S&Udc_kv"].nunique() > 1
                else ["black"]
            ),
            markers=False,
            dashes=True,
        )
    plt.xlabel(x_label, fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    if legend:
        plt.legend(
            title="S [$\\times 10^3 IP/m^3/s$], Potential [kV]" if legend else None,
            ncol=2,
            fontsize=10,
            title_fontsize=10,
        )
    else:
        plt.legend().set_visible(False)
    if save_path is not None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{save_path}\\{y_column}_{x_column}_S&Udc_kv_{suffix_name}_{now}.pdf"
        )
        plt.savefig(filename, bbox_inches="tight")
