import pysindy as ps
import sympy as sp
import numpy as np
import pandas as pd
from pysindy.optimizers.sindy_pi import SINDyPI
import time
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from copy import deepcopy
from src.utils.symbolic_utils import count_nodes

from sklearn.metrics import root_mean_squared_error
from src.utils.load_data import load_sindy_data
from src.datasetSindy import normalize_columns_l2, split_sindy_data
from src.sindyEvaluate import plot_sindy_prediction


# Fit Sindy model
def fit_sindy_model(data_sindy, optimizer, library, save_model=False):
    """
    fit the SINDY model
    """

    def run_fit(x_train, x_dot_train, optimizer, library):
        start_time = time.time()
        print(
            f"Starting fit at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )

        model = ps.SINDy(
            optimizer=optimizer, feature_library=library, discrete_time=False
        )
        model.fit(x=x_train, x_dot=x_dot_train)
        model.print()
        if save_model:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"sindy_model_{now}.pkl", "wb") as f:
                pickle.dump(model, f)
        end_time = time.time()
        print(
            f"Ending fit at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        print(f"Total fit time: {end_time - start_time} seconds")
        return model

    x_train = data_sindy["x_list"]
    x_dot_train = data_sindy["x_dot_list"]
    model = run_fit(x_train, x_dot_train, optimizer, library)
    return model


def fit_sindy_pi_model(data_sindy, save_model=False):
    """
    Fit SINDY PI, rational polynomial model
    """

    def run_fit(x_train, x_dot_train, t, library_sindypi, positions_x0_t):
        start_time = time.time()
        print(
            f"Starting fit at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
        )
        optimizer = SINDyPI(
            reg_weight_lam=0.1,
            tol=1e-5,
            regularizer="l1",
            max_iter=10000,
            model_subset=positions_x0_t,
            verbose_cvxpy=True,
        )
        model = ps.SINDy(
            optimizer=optimizer,
            feature_library=library_sindypi,
            discrete_time=False,
        )
        model.fit(x=x_train, t=t)
        model.print()
        end_time = time.time()
        print(
            f"Ending fit at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
        )
        print(f"Total fit time: {end_time - start_time} seconds")
        if save_model:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            with open(f"sindypi_model_{now}.pkl", "wb") as f:
                pickle.dump(model, f)
        return model

    # Initialize custom SINDy library so that we can have x_dot inside it.
    library_functions = [
        lambda x: x,
        lambda x, y: x * y,
        lambda x: x**2,
        lambda x, y, z: x * y * z,
        lambda x, y: x * y**2,
        lambda x: x**3,
    ]
    library_function_names = [
        lambda x: x,
        lambda x, y: x + y,
        lambda x: x + x,
        lambda x, y, z: x + y + z,
        lambda x, y: x + y + y,
        lambda x: x + x + x,
    ]
    x_train = data_sindy["x_list"]
    x_dot_train = data_sindy["x_dot_list"]
    t = data_sindy["t_list"]
    library_sindypi = ps.PDELibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        derivative_order=1,
        temporal_grid=t,
        implicit_terms=True,
        include_bias=True,
        multiple_trajectories=True,
        fd_scheme=1,
    )
    library_sindypi.fit(x_train)
    list_features = library_sindypi.get_feature_names()
    positions_x0_t = [i for i, feature in enumerate(list_features) if "x0_t" in feature]
    model = run_fit(
        x_train,
        x_dot_train=None,
        t=t,
        library_sindypi=library_sindypi,
        positions_x0_t=positions_x0_t,
    )
    return model


def sindy_surface(
    fit_type: str,
    threshold: float,
    include_bias: bool,
    alpha: float,
    derivative_threshold: float = None,
    max_iter: int = 100,
    polynomial_degree: int = 3,
    interface: str = "up",
    evaluate: bool = True,
    number_features_remove: int = None,
    inverse_relation: bool = False,
    concatenate_inverse: bool = False,
    save_path: str = None,
    model_path: str = None,
    name_of_features: dict = None,
    normalize_columns: bool = False,
    exponential_features: bool = False,
    time_exponent: int = 500,
    save_model: bool = False,
    modifier: str = "",
) -> None:
    """
    Run SINDy models for surface data
    """
    # SINDy surface
    xdot, x, t = load_sindy_data(interface, modifier=modifier)
    # Convert to nano coulomb
    xdot = [xdot_i / 10**3 for xdot_i in xdot]

    # Remove final features (related to the derivatives)
    if number_features_remove is not None:
        x = [
            x[i][
                :, [j for j in range(x[i].shape[-1]) if j not in number_features_remove]
            ]
            for i in range(len(x))
        ]
        if name_of_features is not None:
            keys_to_remove = [
                list(name_of_features.keys())[j] for j in number_features_remove
            ]
            for key in keys_to_remove:
                del name_of_features[key]
            # Rename all keys in name_of_features
            name_of_features = {
                f"x{index}": value
                for index, (key, value) in enumerate(name_of_features.items())
            }
    # Filter xdot arrays based on the derivative_threshold
    if derivative_threshold is not None:
        mask = [abs(xd[:, 0]) >= derivative_threshold for xd in xdot]
        xdot = [xd[m] for xd, m in zip(xdot, mask)]
        x = [xi[m] for xi, m in zip(x, mask)]
        t = [ti[m] for ti, m in zip(t, mask)]

    # Exponential features
    if exponential_features:
        x = [
            np.concatenate([x_i, np.exp(-t_i.reshape(-1, 1) / time_exponent)], axis=1)
            for x_i, t_i in zip(x, t)
        ]
        if name_of_features is not None:
            position = len(name_of_features)
            name_of_features[f"x{position}"] = "exp(-t)"

    # normalize
    if normalize_columns:
        x, xdot = normalize_columns_l2(x, xdot)

    def inverse_function(
        x_i, positions_inverse, positions_S, concatenate_inverse=False
    ):
        """
        !!! attention if something changes in the features order
        """
        if concatenate_inverse:
            for f in positions_inverse:
                x_i = np.concatenate([x_i, 1 / x_i[:, f].reshape(-1, 1)], axis=1)
        else:
            for f in positions_inverse:
                x_i[:, f] = 1 / x_i[:, f]
        # for f in positions_S:
        #     x_i = np.concatenate([x_i, 1 / x_i[:, f].reshape(-1, 1)], axis=1)
        return x_i

    if fit_type == "polynomial":
        # Adding inverses of features
        if name_of_features is not None:
            if inverse_relation == True:
                # Get ordinal position of value "Udc_kV" in the dictionary
                positions_inverse = [
                    list(name_of_features.values()).index(var)
                    for var in ["Udc_kV", "surface_Ez_up", "surface_Er_up"]
                ]
                positions_S = [
                    list(name_of_features.values()).index(var) for var in ["S"]
                ]
                x = [
                    inverse_function(
                        x_i,
                        positions_inverse,
                        positions_S,
                        concatenate_inverse=concatenate_inverse,
                    )
                    for x_i in x
                ]
                # Compute inverse of "Udc_kV", "surface_Ez_up", "surface_Er_up"
                for f in positions_inverse:
                    name_of_features[f"x{f}"] = "1/" + name_of_features[f"x{f}"]
                # for f in positions_S:
                #     name_of_features[f"x{len(name_of_features)}"] = (
                #         "1/" + name_of_features[f"x{f}"]
                #     )
        # Split data
        x, x_test, t, t_test, xdot, xdot_test = split_sindy_data(x, xdot, t)
        # Polynomial SINDy
        optimizer = ps.STLSQ(
            max_iter=max_iter,
            threshold=threshold,
            alpha=alpha,
            normalize_columns=False,
            verbose=True,
        )
        library = ps.PolynomialLibrary(
            degree=polynomial_degree, include_bias=include_bias
        )
        model = fit_sindy_model(
            {"x_list": x, "x_dot_list": xdot}, optimizer, library, save_model=save_model
        )
    if fit_type == "implicit":
        # Remove derivatives
        if number_features_remove is None:  # If something was not already removed
            # Remove derivatives from x
            x = [x_i[:, :-3] for x_i in x]
        # Split data
        x, x_test, t, t_test, xdot, xdot_test = split_sindy_data(x, xdot, t)
        # Implicit SINDy
        model = fit_sindy_pi_model(
            {"x_list": x, "x_dot_list": xdot, "t_list": t}, save_model=save_model
        )
    if fit_type == "load":
        # Split data
        x, x_test, t, t_test, xdot, xdot_test = split_sindy_data(x, xdot, t)
        # Load model
        model = pickle.load(open(model_path, "rb"))

    if evaluate:
        # Predict in the training data
        xdot_predicted = model.predict(x)
        xdot_test_predicted = model.predict(x_test)
        for i_xdot, i_xdot_predicted, suffix_name in [
            (xdot, xdot_predicted, "train"),
            (xdot_test, xdot_test_predicted, "test"),
        ]:
            # Close all plots
            plt.close("all")
            plt.figure(figsize=(6, 4.5))
            plot_sindy_prediction(
                np.concatenate([i_xdot[i][:, 0] for i in range(len(i_xdot))]),
                np.concatenate(
                    [i_xdot_predicted[i][:, 0] for i in range(len(i_xdot_predicted))]
                ),
                units="nC/m$^2$/s",
                plot_equation=True,
                ax=plt.gca(),
                log_scale=False,
                save_path=save_path,
                suffix_name=suffix_name,
            )
    # Print expression with features names
    pretty_equation = model.equations()[0]
    print(pretty_equation)
    for key, name in name_of_features.items():
        pretty_equation = pretty_equation.replace(key, name)
    print(pretty_equation)
    return {
        "data": {"x": x, "x_test": x_test, "xdot": xdot, "xdot_test": xdot_test},
        "model": model,
        "pretty_equation": pretty_equation,
    }


def grid_search_sindy(
    save_path: str, original_features: dict, interface: str, modifier: str = ""
):
    """
    Grid search of threshold
    """
    # Initialize a DataFrame to store the results
    results_df = pd.DataFrame(
        columns=[
            "threshold",
            "alpha",
            "include_bias",
            "time",
            "equation",
            "pretty_equation",
            "rmse_train",
            "rrmse_train",
            "rmse_test",
            "rrmse_test",
            "complexity",
        ]
    )
    for polyn_order in [2, 3]:
        for include_bias in [True, False]:
            for i_threshold in [0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
                for i_alpha in [0.01, 1, 10, 100, 200, 500, 1000]:
                    time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_data = sindy_surface(
                        fit_type="polynomial",
                        threshold=i_threshold,
                        alpha=i_alpha,
                        include_bias=include_bias,
                        derivative_threshold=1e-8,
                        max_iter=4000,
                        polynomial_degree=polyn_order,
                        interface=interface,
                        evaluate=True,
                        number_features_remove=[0, 5, 6, 7],
                        inverse_relation=True,
                        concatenate_inverse=True,
                        save_path=save_path,
                        model_path=None,
                        name_of_features=deepcopy(original_features),
                        normalize_columns=False,
                        exponential_features=True,
                        modifier=modifier,
                    )
                    model = model_data["model"]
                    rmse_train = root_mean_squared_error(
                        np.concatenate(model.predict(model_data["data"]["x"]))[:, 0],
                        np.concatenate(model_data["data"]["xdot"])[:, 0],
                    )
                    rrmse_train = rmse_train / np.mean(
                        np.abs(np.concatenate(model_data["data"]["xdot"])[:, 0])
                    )
                    rmse_test = root_mean_squared_error(
                        np.concatenate(model.predict(model_data["data"]["x_test"]))[
                            :, 0
                        ],
                        np.concatenate(model_data["data"]["xdot_test"])[:, 0],
                    )
                    rrmse_test = rmse_test / np.mean(
                        np.abs(np.concatenate(model_data["data"]["xdot_test"])[:, 0])
                    )

                    # Count the number of operations in the equation
                    equation_sympy = sp.sympify(
                        model.equations()[0].replace(" + ", "+").replace(" ", "*")
                    )
                    # Append the results to the DataFrame
                    results_df = pd.concat(
                        [
                            results_df,
                            pd.DataFrame.from_dict(
                                {
                                    "threshold": [i_threshold],
                                    "alpha": i_alpha,
                                    "include_bias": include_bias,
                                    "time": time,
                                    "polynomial_degree": polyn_order,
                                    "equation": model.equations()[0],
                                    "pretty_equation": model_data["pretty_equation"],
                                    "rmse_train": rmse_train,
                                    "rrmse_train": rrmse_train,
                                    "rmse_test": rmse_test,
                                    "rrmse_test": rrmse_test,
                                    "complexity": count_nodes(equation_sympy),
                                }
                            ),
                        ],
                        ignore_index=True,
                    )
                    print(
                        f"bias: {include_bias}, threshold: {i_threshold}, alpha: {i_alpha}, complexity: {count_nodes(equation_sympy)}, polynomial_degree: {polyn_order}"
                    )
    # Save the results to a CSV file in the specified save_path
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_csv_path = save_path + f"\\{now}_results.csv"
    results_df.to_csv(results_csv_path, index=False)
