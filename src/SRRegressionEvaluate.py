import numpy as np
import pandas as pd
from src.utils.load_data import load_train_data
from src.conicalInsulator import insert_physical_constant
from src.utils.load_data import load_sindy_data
from src.conicalInsulator import load_conical_comsol_parameters
from matplotlib import pyplot as plt
import numpy as np
from pysr import PySRRegressor
from src.conicalInsulator import insert_physical_constant, insert_parameters
from src.utils.load_data import load_train_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import sympy as sp
import datetime
from src.utils.transform import split_train_val_test
from src.utils.plotting import round_expr
import seaborn as sns
import os


def model_symbolic_regression(X, y):
    """
    Symbolic regression
    tips: https://astroautomata.com/PySR/tuning/#fn:1

    q and then <enter> to quit
    """
    model = PySRRegressor(
        niterations=10000,
        binary_operators=["+", "-", "*", "/", "^"],
        maxsize=20,  # Max complexity of an equation.
        model_selection="accuracy",
        constraints={"^": (-1, 3)},
        unary_operators=["exp", "log"],
        parsimony=0.004,  # Higher more complexity is punished
        batching=False,
        bumper=True,
        random_state=13,
    )
    model.fit(X, y)
    return model


def symbolic_regression(
    features,
    run_symbolic=False,
    var_to_predict="Current (A)",
    units="",
    run_graph=True,
    save_path=None,
    remove_outliers=False,
    model_path=None,
    divide_by_area=False,
    modifier="",
):
    """
    Run symbolic regression for dark currents, time to steady state
    """
    df = symbolic_regression_data(
        features,
        var_to_predict=var_to_predict,
        divide_by_area=divide_by_area,
        remove_outliers=remove_outliers,
        modifier=modifier,
    )
    if run_symbolic and model_path is None:
        datasets = split_train_val_test(
            df,
            test_size=0.05,
            total_validation_size=0.15,
            validation_groups=3,
            mix_validation=True,
            n_groups=5,
        )
        df_train = datasets["train"]
        y = df_train[var_to_predict]
        sr_model = model_symbolic_regression(
            df_train[features],
            y,
        )
        run_plot_sr_equations(
            datasets,
            features,
            var_to_predict,
            units=units,
            sr_model=sr_model,
            save_path=save_path,
        )

    if model_path is not None:
        sr_model = load_and_plot_sr_equations(
            df, features, var_to_predict, units, model_path, save_path
        )
    return sr_model


def symbolic_regression_surface(interface="up", save_path=None):
    """
    Run symbolic regression for surface data
    """
    xdot, x, _ = load_sindy_data(interface)
    df_dot = pd.DataFrame(np.concatenate(xdot)[:, 0], columns=["x_dot"])
    df_x = pd.DataFrame(
        np.concatenate(x),
        columns=[
            "x",
            "S",
            "Udc_kV",
            f"surface_Ez_{interface}",
            f"surface_Er_{interface}",
        ],
    )
    df = pd.concat(
        [df_dot, df_x],
        axis=1,
    )
    # Rename column
    df.rename(columns={"x": f"surface_charge_density_{interface}"}, inplace=True)
    features = [
        f"surface_charge_density_{interface}",
        "ion_pair_rate",
        "Udc_kV",
        f"surface_Ez_{interface}",
        f"surface_Er_{interface}",
    ]
    # Treat data
    df = treat_data_symbolic_regression(df, features)
    # Run symbolic regression
    datasets = split_train_val_test(df)
    df_train = datasets["train"]
    y = df_train["x_dot"]
    sr_model = model_symbolic_regression(
        df_train[features],
        y,
    )
    run_plot_sr_equations(
        datasets,
        features,
        "x_dot",
        sr_model,
        save_path=save_path,
    )


def symbolic_regression_data(
    features, var_to_predict, divide_by_area=False, remove_outliers=False, modifier=""
):
    """
    Symbolic regression for dark currents load and prepare data
    """
    # Load dark_currents
    df = load_train_data("dark_currents", modifier=modifier)
    df_es = load_train_data("dark_currents_es_solution")

    df_es.drop(columns=["Time_s", "S"], inplace=True)
    df_es = df_es.add_suffix("_es")
    # Merge ES features in dark_currents
    df = pd.merge(
        df,
        df_es,
        left_on="Udc_kV",
        right_on="Udc_kV_es",
        how="left",
    )
    # Insert physical constants
    df = insert_physical_constant(df)

    # Load system parameters
    sys_params = load_conical_comsol_parameters()

    # for example, transform Current (A) to picoA
    df = treat_data_symbolic_regression(df, features)

    if divide_by_area:
        for i_current in [
            "Current (A)",
            "Current (A) HV",
            "Current (A) SLD HV",
            "Current (A) SLD GRD",
        ]:
            # Current (A) is the current through the ground electrode
            df[i_current.replace("A", "A_m2")] = (
                df[i_current] / sys_params["area_grounded_electrode"]
            )
            # Var to predict is the current per unit area
            var_to_predict = i_current.replace("A", "A_m2")

    # remove outliers
    if remove_outliers:
        print("Removing outliers")
        if var_to_predict == "steady_state_time_max_up":
            df = df[df["steady_state_time_max_up"] <= 3800]
        elif (var_to_predict == "Current (A)") or (var_to_predict == "Current (A_m2)"):
            df = df[df["Current (A)"] >= 0]

    return df


def symbolic_regression_conductivity(save_path):
    """
    Symbolic regression for conductivity
    """
    df_conductivity = load_train_data("surface_2D_steady_all")
    # remove where Ez or Er is 0
    df_conductivity = df_conductivity[df_conductivity["Ez"] != 0]
    df_conductivity = df_conductivity[df_conductivity["Er"] != 0]
    df_conductivity["sigma_z"] = df_conductivity["JGz"] / df_conductivity["Ez"]
    df_conductivity["sigma_r"] = df_conductivity["JGr"] / df_conductivity["Er"]
    # transform ln
    df_conductivity["ln_sigma_r"] = np.log(df_conductivity["sigma_r"])
    features = [
        "ion_pair_rate",
        "Er",
    ]
    var_to_predict = "ln_sigma_r"
    df_conductivity = treat_data_symbolic_regression(df_conductivity, features)
    # Sample df_conductivity based on the column "sigma_z"
    df_conductivity_sampled = df_conductivity.sample(
        frac=0.0004, weights=np.abs(df_conductivity["sigma_z"])
    )
    print(df_conductivity_sampled.shape)
    datasets = split_train_val_test(
        df_conductivity_sampled,
        test_size=0.05,
        total_validation_size=0.15,
        validation_groups=3,
        mix_validation=True,
        n_groups=5,
    )
    df_train = datasets["train"]
    y = df_train[var_to_predict]
    sr_model = model_symbolic_regression(
        df_train[features],
        y,
    )
    run_plot_sr_equations(
        datasets,
        features,
        var_to_predict,
        sr_model,
        save_path=save_path,
    )
    return sr_model


def load_and_plot_sr_equations_conductivity(
    units, load_path: str = None, save_path: str = None
):
    """
    Load model pickle and plot equations
    """
    df = load_train_data("surface_2D_steady_all")
    # remove where Ez or Er is 0
    df = df[df["Ez"] != 0]
    df = df[df["Er"] != 0]
    df["sigma_z"] = df["JGz"] / df["Ez"]
    df["sigma_r"] = df["JGr"] / df["Er"]
    # transform ln
    df["ln_sigma_r"] = np.log(df["sigma_r"])
    features = [
        "ion_pair_rate",
        "Er",
    ]
    var_to_predict = "ln_sigma_r"
    df = treat_data_symbolic_regression(df, features)
    # Sample df_conductivity based on the column "sigma_z"
    df_conductivity_sampled = df.sample(frac=0.0004, weights=np.abs(df["sigma_z"]))
    print(df_conductivity_sampled.shape)
    datasets = split_train_val_test(
        df_conductivity_sampled,
        test_size=0.05,
        total_validation_size=0.15,
        validation_groups=3,
        mix_validation=True,
        n_groups=5,
    )
    sr_model = PySRRegressor.from_file(load_path)

    df_train, df_test = datasets["train"], datasets["test"]
    y = df_train[var_to_predict]
    plot_sr_equations(
        df_train,
        features,
        var_to_predict,
        units=units,
        sr_model=sr_model,
        plot_equations=True,
        save_path=save_path,
        suffix="train",
    )
    plot_sr_equations(
        df_test,
        features,
        var_to_predict,
        units=units,
        sr_model=sr_model,
        plot_equations=True,
        save_path=save_path,
        suffix="test",
    )

    return sr_model


def load_and_plot_sr_equations(
    df, features, var_to_predict, units, load_path: str = None, save_path: str = None
):
    """
    Load model pickle and plot equations
    """
    sr_model = PySRRegressor.from_file(load_path)
    datasets = split_train_val_test(df)
    df_train, df_test = datasets["train"], datasets["test"]
    y = df_train[var_to_predict]
    plot_sr_equations(
        df_train,
        features,
        var_to_predict,
        units=units,
        sr_model=sr_model,
        plot_equations=True,
        save_path=save_path,
        suffix="train",
    )
    plot_sr_equations(
        df_test,
        features,
        var_to_predict,
        units=units,
        sr_model=sr_model,
        plot_equations=True,
        save_path=save_path,
        suffix="test",
    )
    # Plot complexity
    plot_sr_complexity(sr_model, save_path)
    return sr_model


def run_plot_sr_equations(
    datasets, features, var_to_predict, units, sr_model, save_path: str = None
):
    """
    Symbolic regression
    """
    plot_sr_equations(
        datasets["train"],
        features,
        var_to_predict,
        units=units,
        sr_model=sr_model,
        plot_equations=True,
        save_path=save_path,
        suffix="train",
    )
    plot_sr_equations(
        datasets["test"],
        features,
        var_to_predict,
        units=units,
        sr_model=sr_model,
        plot_equations=True,
        save_path=save_path,
        suffix="test",
    )
    # Plot complexity vs distribution of loss
    plot_sr_validation_loss(
        datasets["validation"],
        features=features,
        var_to_predict=var_to_predict,
        sr_model=sr_model,
        save_path=save_path,
    )


def plot_sr_validation_loss(datasets, features, var_to_predict, sr_model, save_path):
    """
    Plot validation loss for each complexity equation
    """
    # Evaluate the equations
    evaluate_equations = [-1 - x for x in range(min(5, len(sr_model.equations_)))]

    # Plotting each prediction in a subplot
    num_equations = min(len(evaluate_equations), 4)
    num_rows = (num_equations) // 2  # Calculate the number of rows needed

    list_rmse = []
    list_complexity = []
    list_validation_group = []

    # iterates in validation groups
    for i, eq in enumerate(evaluate_equations[:num_equations]):
        for i_validation, df_i_validation in enumerate(datasets):
            predictions = sr_model.predict(df_i_validation[features], index=eq)
            true_values = df_i_validation[var_to_predict]
            rmse = root_mean_squared_error(true_values, predictions)
            list_rmse.append(rmse)
            list_complexity.append(sr_model.equations_.iloc[eq]["complexity"])
            list_validation_group.append(i_validation)

    # To plot
    df_validation_loss = pd.DataFrame(
        {
            "rmse": list_rmse,
            "complexity": list_complexity,
            "validation_group": list_validation_group,
        }
    )

    # Save errors
    equation_file_path = (
        sr_model.equation_file_.replace(".csv", "") + f"_validation.txt"
    )
    data_errors = [
        f"{validation_group}, {error}, {complexity}"
        for validation_group, error, complexity in zip(
            list_validation_group, list_rmse, list_complexity
        )
    ]
    _df = df_validation_loss.groupby("complexity").agg({"rmse": ["mean", "std"]})
    # Data errors
    agg_data_errors = [
        f"{complexity}, {mean}, {std}"
        for complexity, mean, std in _df.reset_index().values.tolist()
    ]
    data_errors.extend(agg_data_errors)
    with open(equation_file_path, "w") as file:
        file.write("\n".join(data_errors))

    # to plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax = sns.scatterplot(x="complexity", y="rmse", data=df_validation_loss, ax=ax)
    ax.set_yscale("log")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("$RMSE_{val}$")

    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}\\{var_to_predict}_validation_{now}.pdf"
        plt.savefig(filename, bbox_inches="tight")
        plt.show()


def plot_sr_complexity(sr_model, save_path: str = None):
    """
    Plot complexity vs loss
    """
    plt.plot(
        sr_model.equations_["complexity"],
        sr_model.equations_["loss"],
        color="black",
    )
    plt.xlabel("Complexity")
    plt.ylabel("Loss")
    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}\\complexity_loss_{now}.pdf"
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def save_each_ax(save_path, var_to_predict, suffix, eq, i, fig, ax):
    # Save each subplot as a separate figure
    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        subplot_filename = os.path.join(
            save_path, f"{var_to_predict}_{suffix}_subplot_eq{eq}_i{i}_{now}.pdf"
        )
        # Save just the portion _inside_ the second axis's boundaries
        extent = ax.get_tightbbox().transformed(fig.dpi_scale_trans.inverted())
        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig(subplot_filename, bbox_inches=extent)


def plot_sr_equations(
    df,
    features,
    var_to_predict,
    sr_model,
    plot_equations=None,
    units: str = "",
    save_path: str = None,
    suffix: str = None,
):
    # Evaluate the equations
    evaluate_equations = [-1 - x for x in range(min(5, len(sr_model.equations_)))]

    # Plotting each prediction in a subplot
    num_equations = min(len(evaluate_equations), 4)
    num_rows = (num_equations) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 2, figsize=(10, 5 * num_rows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    list_errors = []
    list_complexity = []
    for i, eq in enumerate(evaluate_equations[:num_equations]):
        ax = axes[i]
        predictions = sr_model.predict(df[features], index=eq)
        errors = plot_sr_prediction(
            df=df,
            sr_model=predictions,
            var_to_predict=var_to_predict,
            log_scale=False,
            units=units,
            expression=sr_model.get_best(index=eq)["sympy_format"],
            ax=ax,
            plot_equation=plot_equations,
        )
        list_errors.append(errors.values())
        list_complexity.append(sr_model.equations_.iloc[eq]["complexity"])
        save_each_ax(save_path, var_to_predict, suffix, eq, i, fig, ax)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # fig.tight_layout(pad=2)  # Add padding between plots
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    if save_path is not None:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{save_path}\\{var_to_predict}_{suffix}_{now}.pdf"
        plt.savefig(filename)  #  bbox_inches="tight"
        plt.show()

    # Save errors
    equation_file_path = sr_model.equation_file_.replace(".csv", "") + f"_{suffix}.txt"
    errors_data = [
        f"{error}, {complexity}"
        for error, complexity in zip(list_errors, list_complexity)
    ]
    with open(equation_file_path, "w") as file:
        file.write("\n".join(errors_data))


def plot_sr_prediction(
    df,
    top_limit=None,
    sr_model=None,
    var_to_predict=None,
    log_scale=True,
    units="",
    expression=None,
    ax=None,
    plot_equation=None,
):
    df[f"_predicted_{var_to_predict}"] = sr_model
    # Perform linear regression
    X = df[f"_predicted_{var_to_predict}"].values.reshape(-1, 1)
    y = df[var_to_predict].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    # Calculate RMSE and R2
    rmse = root_mean_squared_error(y, X)
    r2 = r2_score(y, X)
    rrmse = rmse / np.mean(np.abs(X))

    # Plotting
    ax.plot(
        [df[var_to_predict].min(), df[var_to_predict].max()],
        [df[var_to_predict].min(), df[var_to_predict].max()],
        color="black",
        linewidth=1,
        linestyle="dashed",
    )
    ax.scatter(df[var_to_predict], df[f"_predicted_{var_to_predict}"], color="black")
    if log_scale:
        ax.set_yscale("log")
        ax.set_xscale("log")
    if top_limit:
        ax.set_ylim(0, top_limit)
        ax.set_xlim(0, top_limit)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Rotate x labels by 45 degrees to avoid overlapping
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.set_ylabel(f"Predicted [${units}$]")
    ax.set_xlabel(f"Target [${units}$]")

    # Display equation, RMSE, and R2 on the plot
    if plot_equation:
        equation = f"y = {reg.coef_[0]:.2f}x + {reg.intercept_:.2f}"
        # Determine the position for the text based on the data points
        x_pos = 0.55
        y_pos = 0.05
        ax.text(
            x_pos,
            y_pos,
            f"RMSE: {rmse:.2e} [${units}$]\nRRMSE: {rrmse:.2e} [${units}$]",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top" if y_pos == 0.95 else "bottom",
        )
        print(
            f"expression: ${sp.latex(sp.simplify(round_expr(expression)), min = -2, max = 2)}$"
        )
    return {"rmse": rmse, "rrmse": rrmse}


def treat_data_symbolic_regression(df, features):
    """
    Treat data for symbolic regression. Return the results from last step of simulation
    Alternative for filtering data:
        # 10 hour condition
        # df = df[df["Time_s"] == 36036]
    """
    df = insert_parameters(df)
    df = insert_physical_constant(df)
    # For some reason the name S cannot be used in pySR
    df["S"] = df["S"] * 1e3
    df = df.rename(columns={"S": "ion_pair_rate"})
    df["Udc_kV"] = df["Udc_kV"].astype(int)
    if "Time_s" in df.columns:
        # Sort and drop duplicates to get the last time step of simulation
        df.sort_values(by="Time_s", inplace=True, ascending=False)
        df = df.drop_duplicates(subset=features)
        df.reset_index(drop=True, inplace=True)
    for i_current in [
        "Current (A)",
        "Current (A) HV",
        "Current (A) SLD HV",
        "Current (A) SLD GRD",
    ]:
        if i_current in df.columns:
            # Target variables
            df[i_current] = df[i_current] * 1e12  # to pico amperes
    if "steady_state_time_max_up" in df.columns:
        df["steady_state_time_max_up"] = (
            df["steady_state_time_max_up"] / 3600
        )  # to hours
    if "sigma_z" in df.columns:
        df["sigma_z"] = df["sigma_z"] * 1e12  # to pico siemens
    if "sigma_r" in df.columns:
        df["sigma_r"] = df["sigma_r"] * 1e12  # to pico siemens
    return df
