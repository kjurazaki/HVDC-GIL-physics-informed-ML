import numpy as np
import matplotlib.pyplot as plt
import pwlf
from sklearn.linear_model import LinearRegression
import pandas as pd
from src.utils.plotting import FlexiblePlotter, InteractivePlot


class LinearFit:
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def fit_model(self):
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.x.reshape(-1, 1), self.y.reshape(-1, 1))
        print(self._get_fit())

    def _get_fit(self):
        return {
            "slope": self.linear_model.coef_[0],
            "intercept": self.linear_model.intercept_,
        }

    def graph_fit(self):
        # Predict y values based on the fitted model
        x_fit = np.linspace(min(self.x), max(self.x), 100)
        y_fit = self.linear_model.predict(x_fit.reshape(-1, 1))

        # Plot the original data and the piecewise linear fit
        plt.scatter(self.x, self.y, color="green", label="Data Points")
        plt.plot(
            x_fit,
            y_fit,
            color="red",
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()


class PieceWiseFit:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit_model(self, number_of_segments):
        # Initialize the piecewise linear fit object
        self.number_of_segments = number_of_segments
        self.pwlf_model = pwlf.PiecewiseLinFit(self.x, self.y)
        self.pwlf_model.fit(number_of_segments)

    def _get_fit(self):
        return {
            "slopes": self.pwlf_model.slopes,
            "intercepts": self.pwlf_model.intercepts,
            "break_points": self.pwlf_model.fit_breaks,
        }

    def graph_fit(self):
        # Predict y values based on the fitted model
        x_fit = np.linspace(min(self.x), max(self.x), 100)
        y_fit = self.pwlf_model.predict(x_fit)

        # Plot the original data and the piecewise linear fit
        plt.scatter(self.x, self.y, color="green", label="Data Points")
        plt.plot(
            x_fit,
            y_fit,
            color="red",
            label=f"{self.number_of_segments} Segment Piecewise Fit",
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

    def graph_fit_with_equation(self):
        # Breakpoints
        breakpoints = self.pwlf_model.fit_breaks
        for i in range(self.number_of_segments):
            # Get the slope and intercept for each segment
            slope = self.pwlf_model.slopes[i]
            intercept = self.pwlf_model.intercepts[i]

            # Define the x-range for this segment
            x_start = breakpoints[i]
            x_end = breakpoints[i + 1]

            # Position where to place the equation in the plot (middle of the segment)
            x_pos = (x_start + x_end) / 2
            y_pos = self.pwlf_model.predict([x_pos])[0]  # Corresponding y position

            # Create the equation text as a string
            equation = f"y = {slope:.2e}x + {intercept:.2e}"

            # Add the text annotation for the equation on the plot
            plt.text(x_pos, y_pos, equation, fontsize=10, color="blue", ha="center")

        self.graph_fit()


def _wrapper_compute_fit(df):
    S = df["S"].iloc[0]
    fit_params = piecewise_fit_currents(df, S, 2)
    results = pd.DataFrame([])
    for j, (slope, intercept) in enumerate(
        zip(fit_params["slopes"], fit_params["intercepts"])
    ):
        # Append the result to the list
        results = (
            pd.concat(
                [
                    results,
                    pd.DataFrame.from_dict(
                        {
                            "S": [S],
                            "fitting": j + 1,  # Segment index (1-based)
                            "slope": slope,
                            "intercept": intercept,
                            "break_point": fit_params["break_points"][1],
                        }
                    ),
                ],
                axis=0,
            ),
        )[0]
    return results


def piecewise_fit_currents(df, S, number_of_segments, plot=False):
    # Get the linear piecewise fite for the current vs voltage

    df = df[df["Time_s"] == df["Time_s"].max()]
    df = df[df["S"] == S]
    x = df["Udc_kV"]
    y = df["Current (A)_log"]

    linear_model = PieceWiseFit(x, y)
    linear_model.fit_model(number_of_segments)
    fit_params = linear_model._get_fit()
    if plot:
        linear_model.graph_fit_with_equation()
    return fit_params
