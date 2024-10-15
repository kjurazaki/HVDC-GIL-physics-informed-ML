from scipy.interpolate import interp1d
import numpy as np


def normalize_variables(df, category_cols, variable_cols):
    """
    Normalize the variables in the dataframe
    """

    def normalize(group):
        return (group - group.min()) / (group.max() - group.min())

    for variable in variable_cols:
        df[f"{variable}_norm"] = df.groupby(category_cols)[variable].transform(
            normalize
        )
    return df


def compute_t90(df, time_col, value_col, group_cols=[]):
    """
    Compute the transition time t90 as the Winter paper
    """

    def t90_group(group):
        # Ensure group is sorted by time
        group_sorted = group.sort_values(by=time_col)
        initial = group_sorted.iloc[0][value_col]
        final = group_sorted.iloc[-1][value_col]
        delta = final - initial
        target = initial + 0.9 * delta
        if delta == 0:
            return np.nan
        # Find the two points surrounding the target
        below = group_sorted[group_sorted[value_col] < target].iloc[-1]
        above = group_sorted[group_sorted[value_col] >= target].iloc[0]
        if below is None or above is None:
            return np.nan
        # Linear interpolation between below and above
        t90 = below[time_col] + (target - below[value_col]) * (
            above[time_col] - below[time_col]
        ) / (above[value_col] - below[value_col])
        return t90, target

    return df.groupby(group_cols).apply(t90_group).reset_index(name="t_90")
