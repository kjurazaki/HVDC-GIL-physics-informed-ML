from scipy import stats
import numpy as np


class SteadyState:
    def __init__(
        self,
        data,
        time_column,
        value_column,
        group_columns,
    ):
        self.data = data
        self.time_column = time_column
        self.value_column = value_column
        self.group_columns = group_columns

    def get_steady_state_times(self):
        list_name = []
        id_steady = []
        steady_state_value = []
        steady_state_time = []
        for name, group in self.data.groupby(self.group_columns):
            # Convert inputs to numpy arrays if they aren't already
            time = np.array(group[self.time_column])
            values = np.array(group[self.value_column])
            self.detect_steady_state(time=time, values=values)
            list_name.append(name[0])
            id_steady.append(self.steady_state_idx)
            steady_state_value.append(self.metrics["steady_state_value"])
            steady_state_time.append(self.steady_state_time)

        self.steady_state_bygroup = {
            f"{self.group_columns[0]}": list_name,
            "id_steady": id_steady,
            "steady_state_value": steady_state_value,
            "steady_state_time": steady_state_time,
        }

    def detect_steady_state(
        self, time, values, window_size=20, threshold=1e-3, min_consecutive=3
    ):
        """
        Detect steady state in time series data.

        Parameters:
        -----------
        time : array-like
            Time points of the series
        values : array-like
            Values of the time series
        window_size : int
            Size of the rolling window to analyze (default: 1000)
        threshold : float
            Maximum allowed slope to consider steady state (default: 1e-12)
        min_consecutive : int
            Minimum number of consecutive windows that must meet criteria (default: 3)

        Returns:
        --------
        steady_state_time : float
            Time point where steady state begins
        is_steady : bool
            Whether steady state was detected
        metrics : dict
            Additional metrics about the steady state detection
        """
        if len(time) < window_size:
            return None, False, {"message": "Time series too short for analysis"}

        # Initialize variables
        steady_windows = 0
        steady_state_idx = None
        std_dev_ts = np.std(values)
        for i in range(len(values) - window_size):
            # Get current window
            window_values = values[i : i + window_size]

            # Calculate metrics for this window
            # slope, _, _, _, _ = stats.linregress(window_time, window_values)
            std_dev = np.std(window_values)
            rel_std_dev = std_dev / std_dev_ts

            # Check if window meets steady state criteria
            if rel_std_dev < threshold:
                steady_windows += 1
                if steady_windows >= min_consecutive and steady_state_idx is None:
                    steady_state_idx = i
            else:
                steady_windows = 0
                steady_state_idx = None

        # If we found steady state, return the details
        if steady_state_idx is not None:
            self.metrics = {
                "steady_state_value": np.mean(values[steady_state_idx:]),
                "standard_deviation": np.std(values[steady_state_idx:]),
                "confidence": steady_windows / min_consecutive,
            }
            self.steady_state_idx = steady_state_idx
            self.steady_state_time = time[steady_state_idx]

        return None, False, {"message": "No steady state detected"}

    def naive_find_steady_state(self, threshold=0.1):
        """Find the time to steady-state based on threshold
        threshold: float. Percentage of the total variation of the variable to consider the system as steady-state
        Don't work, only for monotonic data
        """
        steady_state_times = {}
        for name, group in self.data.groupby(self.group_columns):
            group = group.sort_values(by=self.time_column)
            initial_value = group[self.value_column].iloc[0]
            final_value = group[self.value_column].iloc[-1]
            steady_state_value = (
                initial_value + (final_value - initial_value) * threshold
            )

            # Interpolation to find the approximate time when steady state is reached
            above_steady_state = group[group[self.value_column] >= steady_state_value]
            if not above_steady_state.empty:
                idx = above_steady_state.index[0]
                if idx > 0:
                    x0, y0 = group.loc[idx - 1, [self.time_column, self.value_column]]
                    x1, y1 = group.loc[idx, [self.time_column, self.value_column]]
                    steady_state_time = x0 + (steady_state_value - y0) * (x1 - x0) / (
                        y1 - y0
                    )
                else:
                    steady_state_time = group.loc[idx, self.time_column]
            else:
                steady_state_time = group[self.time_column].max()

            steady_state_times[name] = steady_state_time
        return steady_state_times

    def check_steady_state(self, steady_state_tolerance=1e-5):
        """
        Check if the system is in steady-state by comparing if the derivative of the final timestep is close to 0
        """
        steady_state_status = {}
        for name, group in self.data.groupby(self.group_columns):
            group = group.sort_values(by=self.time_column)
            if len(group) < 2:
                steady_state_status[name] = False
                continue

            # Calculate the derivative (difference) of the final two timesteps
            final_derivative = (
                group[self.value_column].iloc[-1] - group[self.value_column].iloc[-2]
            ) / (group[self.time_column].iloc[-1] - group[self.time_column].iloc[-2])

            # Check if the derivative is close to 0
            steady_state_status[name] = abs(final_derivative) < steady_state_tolerance

        return steady_state_status
        pass
