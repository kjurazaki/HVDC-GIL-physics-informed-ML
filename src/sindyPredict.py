from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pysindy as ps


def predict_sindy_model(model: ps.SINDy, x, multiple_trajectories=False):
    # TODO: don't use class, just implement function directly (which might be faster)
    x_dot_dt = model.predict(x, multiple_trajectories=multiple_trajectories)
    return x_dot_dt


def compute_trajectory_for_frame(x_frame, t_list, model):
    x_result = [x_frame[0]]
    t_old_frame = t_list[0]
    for i in range(1, len(t_list)):
        t = t_list[i]
        dt = t - t_old_frame
        x_dot_dt = predict_sindy_model(
            model,
            x_result[-1].reshape(1, -1),
            multiple_trajectories=False,
        )
        x_dot_test = dt * x_dot_dt.flatten()
        x_test = x_result[-1][0] + x_dot_test[0]
        features_new = x_frame[i, 1:]
        x_new = np.array([x_test, features_new[0], features_new[1]])
        x_result.append(x_new)
        t_old_frame = t
    return np.array(x_result)


def run_predict_sindy_model(model, data_sindy):
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(compute_trajectory_for_frame, x_frame, t_list, model): idx
            for idx, (x_frame, t_list) in enumerate(
                zip(data_sindy["x_list"], data_sindy["t_list"])
            )
        }
        results = [None] * len(futures)
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Combine results
    x_combined = np.array(results)

    plt.scatter(data_sindy["x_list"][0][:, 0], x_combined[0][:, 0])
    plt.show()
    return x_combined


def predict_dynamic_surface_up(x: list) -> float:
    """
    equation from
    """
    xdot = 0.88972 / (0.9834 ** np.square(x[0]))
    return xdot


def plot_dynamic_data(data_sindy, x_column="surface_charge_density_up"):
    fig, axs = plt.subplots(3, 1)
    axs[0].scatter(
        data_sindy["t_list"][0],
        data_sindy["x_list"][0][x_column],
        label="value",
    )
    axs[1].scatter(data_sindy["t_list"][0], data_sindy["x_dot_list"][0], label="fd")
    axs[2].scatter(
        data_sindy["t_list"][0], data_sindy["x_dot_sfd_list"][0], label="sfd"
    )
    plt.legend()
    plt.show()
