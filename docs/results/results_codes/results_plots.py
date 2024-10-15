from src.utils.plotting import FlexiblePlotter, InteractivePlot
from matplotlib import pyplot as plt
import numpy as np


def plotterly(df, data):
    # ----
    # Understanding mean, median, min and max of surface charge density
    df_plot = df_grouped[(df_grouped["Time_s"] > 0)]

    plotterly = InteractivePlot(df_plot, "Time_s", "min", "Udc_kV", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Relation of dark currents and S and Udc
    df = data.dataframes["dark_current"]
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Udc_kV", "Current (A)", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Surface charge density vs ARC in changing time
    df = data.dataframes["surface_charge_density"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 35]
    plotterly = InteractivePlot(df, "Arc", "surface_charge_density", "Time_s", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Surface charge density vs ARC in changing S
    df = data.dataframes["surface_charge_density"]
    df = df[df["Time_s"] == df["Time_s"].max()]
    plotterly = InteractivePlot(df, "Arc", "surface_charge_density", "S", ["Udc_kV"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # MIN surface charge density vs TIME, Udc_kV selection
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "Time_s", "surface_charge_density_min", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # MIN surface charge density vs Udc_kV, time selection
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "Udc_kV", "surface_charge_density_min", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # MIN surface charge density vs S, time selection
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "S", "surface_charge_density_min", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Max surface charge density vs time
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "Time_s", "surface_charge_density_max", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Min surface charge density vs Current, changing time, fixed Udc_Kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 335]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "surface_charge_density_min", "Current (A)", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Min surface charge density vs Current, changing S, legend Udc_kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "surface_charge_density_min", "Current (A)", "S", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Min surface charge density vs Current, changing S, legend Udc_kv at last time
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] == df["Time_s"].max()]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "surface_charge_density_min", "Current (A)", "S", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Peclet vs time
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] == df["Time_s"].max()]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(df, "peclet_max", "Udc_kV", "S", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Time vs Current, changing S, legend Udc_kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(df, "Time_s", "Current (A)", "S", ["Udc_kV"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Time vs Current, changing S, legend Udc_kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "Time_s", "surface_charge_density_min", "S", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Surface eletric_potential vs Arc in changing time
    df = data.dataframes["surface_electric_potential"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 15]
    plotterly = InteractivePlot(
        df, "Arc", "surface_electric_potential", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Current by time - different Udc_kV and S
    df = data.dataframes["dark_current"]
    df["Time_s"] = df["Time_s"].astype("int")
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "Current (A)", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Dark Current (A) vs Max Surface charge density
    df = data.dataframes["surface_charge_density"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 335]
    plotterly = InteractivePlot(df, "Arc", "surface_charge_density", "Time_s", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "Charge", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Udc_kV", "Charge", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "S", "Charge", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "S", "concentration_positive", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "concentration_positive", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "concentration_negative", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Udc_kV", "concentration_negative", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "S", "concentration_negative", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df, "Time_s", "electric_field_norm_avg", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    plotterly = InteractivePlot(
        df, "Udc_kV", "electric_field_norm_avg", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()


def scatter_plots(df, data, fittings):
    # Saturation points - Saturation in function of ion-pair generation
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["Udc_kV"] == df["Udc_kV"].max()) & (df["Time_s"] == df["Time_s"].max())],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "Current (A)", ["S"])
    plt.show()

    # Slope of exponential increase given S value - first part
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        fittings[(fittings["fitting"] == 1) & (fittings["S"] > 7e7)],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "slope", ["S"])
    plt.show()

    # Slope of exponential increase given S value - second part
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        fittings[(fittings["fitting"] == 2) & (fittings["S"] > 7e7)],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "slope", ["S"])
    plt.show()

    # Beak point analysis on the saturation - End of exponential behaviour for each S
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        fittings[(fittings["fitting"] == 1) & (fittings["S"] > 7e7)],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "break_point", ["S"])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 9e7) & (df["Udc_kV"] == 35) & (df["Time_s"] != 0)], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"])
    plotter.plot_single_y_axis(
        "Time_s", "concentration_negative", ["S", "Udc_kV"], ax=ax2, color="k"
    )
    plotter.plot_single_y_axis(
        "Time_s", "concentration_positive", ["S", "Udc_kV"], ax=ax2, color="red"
    )
    fig.legend(("current", "concentration_negative", "concentration_positive"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 9e7) & (df["Udc_kV"] == 35) & (df["Time_s"] != 0)], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"], color="red")
    plotter.plot_single_y_axis(
        "Time_s", "electric_field_norm_avg", ["S", "Udc_kV"], ax=ax2
    )
    fig.legend(("Current", "Electric Field"))
    plt.show()

    # Analyse ion-concentration and electric field
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 9e7) & (df["Time_s"] == np.max(df["Time_s"]))], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Udc_kV", "concentration_negative", ["S"], color="red")
    plotter.plot_single_y_axis("Udc_kV", "electric_field_norm_avg", ["S"], ax=ax2)
    fig.legend(("concentration", "electric field"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 1.2e8) & (df["Udc_kV"] == 155) & (df["Time_s"] != 0)], ax
    )


from utils.plotting import FlexiblePlotter, InteractivePlot
from matplotlib import pyplot as plt
import numpy as np


def plotterly(df, data):
    # ----

    # Relation of dark currents and S and Udc
    df = data.dataframes["dark_current"]
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Udc_kV", "Current (A)", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Surface charge density vs ARC in changing time
    df = data.dataframes["surface_charge_density"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 35]
    plotterly = InteractivePlot(df, "Arc", "surface_charge_density", "Time_s", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Surface charge density vs ARC in changing S
    df = data.dataframes["surface_charge_density"]
    df = df[df["Time_s"] == df["Time_s"].max()]
    plotterly = InteractivePlot(df, "Arc", "surface_charge_density", "S", ["Udc_kV"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # MIN surface charge density vs TIME, Udc_kV selection
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "Time_s", "surface_charge_density_min", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # MIN surface charge density vs Udc_kV, time selection
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "Udc_kV", "surface_charge_density_min", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # MIN surface charge density vs S, time selection
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "S", "surface_charge_density_min", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Max surface charge density vs time
    df = data.dataframes["surface_charge_density"]
    df.drop_duplicates(subset=["Udc_kV", "S", "Time_s"], inplace=True)
    df = df[df["Time_s"] != 0]
    plotterly = InteractivePlot(
        df, "Time_s", "surface_charge_density_max", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Min surface charge density vs Current, changing time, fixed Udc_Kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 335]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "surface_charge_density_min", "Current (A)", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Min surface charge density vs Current, changing S, legend Udc_kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "surface_charge_density_min", "Current (A)", "S", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Min surface charge density vs Current, changing S, legend Udc_kv at last time
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] == df["Time_s"].max()]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "surface_charge_density_min", "Current (A)", "S", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Peclet vs time
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] == df["Time_s"].max()]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(df, "peclet_max", "Udc_kV", "S", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Time vs Current, changing S, legend Udc_kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(df, "Time_s", "Current (A)", "S", ["Udc_kV"])
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Time vs Current, changing S, legend Udc_kv
    df = data.dataframes["dark_current"]
    df = df[df["Time_s"] != 0]
    df["surface_charge_density_min"] = 1e6 * df["surface_charge_density_min"]
    plotterly = InteractivePlot(
        df, "Time_s", "surface_charge_density_min", "S", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Surface eletric_potential vs Arc in changing time
    df = data.dataframes["surface_electric_potential"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 15]
    plotterly = InteractivePlot(
        df, "Arc", "surface_electric_potential", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Current by time - different Udc_kV and S
    df = data.dataframes["dark_current"]
    df["Time_s"] = df["Time_s"].astype("int")
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "Current (A)", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    # ----

    # Dark Current (A) vs Max Surface charge density
    df = data.dataframes["surface_charge_density"]
    df = df[df["Time_s"] != 0]
    df = df[df["Udc_kV"] == 335]
    plotterly = InteractivePlot(df, "Arc", "surface_charge_density", "Time_s", ["S"])
    fig = plotterly.create_plot()
    fig.show()

    # ----
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "Charge", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Udc_kV", "Charge", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "S", "Charge", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "S", "concentration_positive", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "concentration_positive", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Time_s", "concentration_negative", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "Udc_kV", "concentration_negative", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df[(df["Time_s"] != 0)], "S", "concentration_negative", "Time_s", ["Udc_kV"]
    )
    fig = plotterly.create_plot()
    fig.show()
    plotterly = InteractivePlot(
        df, "Time_s", "electric_field_norm_avg", "Udc_kV", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()

    plotterly = InteractivePlot(
        df, "Udc_kV", "electric_field_norm_avg", "Time_s", ["S"]
    )
    fig = plotterly.create_plot()
    fig.show()


def scatter_plots(df, data, fittings):
    # Saturation points - Saturation in function of ion-pair generation
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["Udc_kV"] == df["Udc_kV"].max()) & (df["Time_s"] == df["Time_s"].max())],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "Current (A)", ["S"])
    plt.show()

    # Slope of exponential increase given S value - first part
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        fittings[(fittings["fitting"] == 1) & (fittings["S"] > 7e7)],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "slope", ["S"])
    plt.show()

    # Slope of exponential increase given S value - second part
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        fittings[(fittings["fitting"] == 2) & (fittings["S"] > 7e7)],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "slope", ["S"])
    plt.show()

    # Beak point analysis on the saturation - End of exponential behaviour for each S
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        fittings[(fittings["fitting"] == 1) & (fittings["S"] > 7e7)],
        ax,
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("S", "break_point", ["S"])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 9e7) & (df["Udc_kV"] == 35) & (df["Time_s"] != 0)], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"])
    plotter.plot_single_y_axis(
        "Time_s", "concentration_negative", ["S", "Udc_kV"], ax=ax2, color="k"
    )
    plotter.plot_single_y_axis(
        "Time_s", "concentration_positive", ["S", "Udc_kV"], ax=ax2, color="red"
    )
    fig.legend(("current", "concentration_negative", "concentration_positive"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 9e7) & (df["Udc_kV"] == 35) & (df["Time_s"] != 0)], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"], color="red")
    plotter.plot_single_y_axis(
        "Time_s", "electric_field_norm_avg", ["S", "Udc_kV"], ax=ax2
    )
    fig.legend(("Current", "Electric Field"))
    plt.show()

    # Analyse ion-concentration and electric field
    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 9e7) & (df["Time_s"] == np.max(df["Time_s"]))], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Udc_kV", "concentration_negative", ["S"], color="red")
    plotter.plot_single_y_axis("Udc_kV", "electric_field_norm_avg", ["S"], ax=ax2)
    fig.legend(("concentration", "electric field"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 1.2e8) & (df["Udc_kV"] == 155) & (df["Time_s"] != 0)], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"])
    plotter.plot_single_y_axis(
        "Time_s", "negative_charge", ["S", "Udc_kV"], ax=ax2, color="k"
    )
    plotter.plot_single_y_axis(
        "Time_s", "positive_charge", ["S", "Udc_kV"], ax=ax2, color="red"
    )
    fig.legend(("current", "concentration_negative", "concentration_positive"))
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    plotter = FlexiblePlotter(
        df[(df["S"] == 1.2e8) & (df["Udc_kV"] == 155) & (df["Time_s"] != 0)], ax
    )
    ax2 = ax.twinx()
    plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"])
    plotter.plot_single_y_axis(
        "Time_s", "concentration_negative", ["S", "Udc_kV"], ax=ax2, color="k"
    )
    plotter.plot_single_y_axis(
        "Time_s", "concentration_positive", ["S", "Udc_kV"], ax=ax2, color="red"
    )
    fig.legend(("current", "concentration_negative", "concentration_positive"))
    plt.show()
