import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from src.utils.plotting import FlexiblePlotter, InteractivePlot


class ComsolComponent:
    """
    Class discontinued as the fitting were better performed in matlab
    """

    def __init__(self, dataframe_dict):
        self.dataframe_dict = dataframe_dict
        self.dataframes = {}
        self.read_files()
        # Merge dataframes
        # self.dataframes["merged_dataframes"] = self.merge_dataframes(
        #     ["r", "z"], ["current", "electricField"]
        # )

    def read_files(self):
        for name, values in self.dataframe_dict.items():
            skip_lines = values[1]
            filepath = values[0]
            df = pd.read_csv(filepath, skiprows=skip_lines, sep="\s+")
            columns_names = df.columns[1:]
            df = df.iloc[:, 0:-1]
            df.columns = columns_names
            self.dataframes[name] = df

    def merge_dataframes(self, key, dataframe_names):
        merged_df = self.dataframes[dataframe_names[0]]
        for name in dataframe_names[1:]:
            merged_df = pd.merge(merged_df, self.dataframes[name], on=key, how="inner")
        return merged_df


class ConductivityFitting(ComsolComponent):
    """
    TODO Class used in the problem of conductivity fitting - not implemented
    """

    def __init__(self, dataframe_dict, skip_lines=0):
        super().__init__(dataframe_dict, skip_lines)

    def correct_names(self):
        # Correct columns of eletric
        self.dataframes["electricField"] = self.dataframes["electricField"].rename(
            columns={"Electric": "ec3.normE"}
        )
        self.dataframes["electricField"].drop(columns=["field", "norm"], inplace=True)

        self.dataframes["current"] = self.dataframes["current"].rename(
            columns={"Current": "ec3.normJ"}
        )
        self.dataframes["current"].drop(columns=["density", "norm"], inplace=True)

    def plot_dataframe(
        self, df_name, x_axis, y_axis, intensity_axis=None, log_scale=None
    ):
        df = self.dataframes[df_name]
        if intensity_axis is not None:
            plt.scatter(df[x_axis], df[y_axis], c=df[intensity_axis], cmap="viridis")
            plt.title(f"2D Plot with {intensity_axis} as Intensity")
        else:
            plt.scatter(df[x_axis], df[y_axis])
            plt.title("2D Plot")

        if log_scale is not None:
            if "x" in log_scale:
                plt.xscale("log")
            if "y" in log_scale:
                plt.yscale("log")

        plt.colorbar()
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

    # Smooth data method
    def smooth_data(self, df_name, keys, sample_per_group=10):
        """
        Smooth data from dataframe df_name, using groups of sample_per_group samples.
        Data is sorted by the first key passed.
        The samples are grouped by sample_per_group samples.
        Means is computed as the mean off the 'key' parameter by each grouped.
        : parameters
        :: keys: list of keys to be smoothed.
        :: sample_per_group: number of samples per group.
        :: df_name: name of the dataframe to be smoothed.
        """
        self.dataframes[df_name].sort_values(by=keys[0], inplace=True)
        N_groups = len(self.dataframes[df_name]) // sample_per_group
        means = np.zeros([N_groups, len(keys)])
        for k in range(len(keys)):
            for n in range(N_groups):
                means[n, k] = np.mean(
                    self.dataframes[df_name][
                        sample_per_group * n : sample_per_group * (n + 1)
                    ][keys[k]]
                )
        return means

    def fit_data(self, x_data, y_data, E2):
        def func(x, k, d):
            return k * x * (1 + (x / E2) ** d)

        p0 = [1, 1]  # initial guess for parameters
        params, _ = curve_fit(func, x_data, y_data, p0=p0, xtol=1e-30, gtol=1e-30)
        return params, _

    def plot_fit(self, x_data, y_data, params, E2):
        def func(x, k, d):
            return k * x * (1 + (x / E2) ** d)

        plt.scatter(x_data, y_data, label="Original Data")
        y_predicted = func(x_data, *params)
        plt.plot(x_data, y_predicted, label="Fitted Curve", c="r")
        plt.legend()


def fit_params():
    """
    Function discontinued as the fitting were better performed in matlab
    This were used to fit the data from the comsol simulation to the regression model of conductivity
    """
    files_folder = "./databases/"
    files_names = {
        "conductivity": "Conductivity_ec3_smoothed.txt",
        "current": "Current_ec3_smoothed.txt",
        "electricField": "EletricField_ec3_smoothed.txt",
    }
    files_path = {key: files_folder + value for key, value in files_names.items()}
    results = ComsolComponent(files_path, skip_lines=7)

    means = results.smooth_data("merged_dataframes", ["ec3.normE", "ec3.normJ"])

    # smoothed data
    results.dataframes["smoothed_dataframes"] = pd.DataFrame.from_dict(
        {"ec3.normE_smoothed": means[:, 0], "ec3.normJ_smoothed": means[:, 1]}
    )

    results.plot_dataframe(
        "merged_dataframes", "ec3.normE", "ec3.normJ", log_scale=["y"]
    )
    plt.scatter(means[:, 0], means[:, 1])
    plt.yscale("log")
    plt.show()

    params, _ = results.fit_data(
        x_data=results.dataframes["smoothed_dataframes"]["ec3.normE_smoothed"],
        y_data=results.dataframes["smoothed_dataframes"]["ec3.normJ_smoothed"],
        E2=1e7,
    )

    results.plot_fit(
        x_data=results.dataframes["smoothed_dataframes"]["ec3.normE_smoothed"],
        y_data=results.dataframes["smoothed_dataframes"]["ec3.normJ_smoothed"],
        E2=1e7,
        params=params,
    )
    plt.yscale("log")
    plt.show()
    print(params)


def build_dataframe_alliterations(files_folder, iterations):
    """
    Buid a dataframe with all iterations from the comsol simulation.
    """
    results = pd.DataFrame()
    for i_ionpair in list(iterations.keys()):
        for i_iteration in iterations[i_ionpair]:
            files_names = {
                "current": f"current_norm_{i_ionpair}_iter{i_iteration}.txt",
                "electricField": f"eletric_field_norm_{i_ionpair}_iter{i_iteration}.txt",
            }
            files_path = {
                key: files_folder + value for key, value in files_names.items()
            }
            df = ComsolComponent(files_path, skip_lines=7).dataframes[
                "merged_dataframes"
            ]
            df["ionpair_S"] = i_ionpair
            df["iter"] = i_iteration
            results = pd.concat([results, df])
    return results
