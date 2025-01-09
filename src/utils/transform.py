from scipy.interpolate import interp1d
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def split_train_val_test(
    df,
    test_size=0.05,
    total_validation_size=0.15,
    validation_groups=3,
    mix_validation=True,
    n_groups=5,
):
    """
    Split data into training and testing sets
    """
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=13)
    # Split train into train and validation
    df_train, df_validation = train_test_split(
        df_train, test_size=total_validation_size, random_state=13
    )

    # Split validation data into validation groups
    validation_splits = split_validation_groups(df_validation, validation_groups)

    # Mix validation groups - it is valid as none of these samples are in training
    if mix_validation:
        validation_splits = mix_validation_groups(validation_splits, n_groups=n_groups)
    return {"train": df_train, "validation": validation_splits, "test": df_test}


def mix_validation_groups(validation_splits, n_groups):
    """
    Create new n_groups groups mixing between validation groups
    """
    # Mix validation groups to generate more validation group samples with fixed size
    mixed_validation_splits = []
    original_size = len(
        validation_splits[0]
    )  # Assuming all original splits have the same size
    for i in range(n_groups):
        mixed_group = (
            pd.concat(validation_splits).sample(n=original_size).reset_index(drop=True)
        )
        mixed_validation_splits.append(mixed_group)
    validation_splits.extend(mixed_validation_splits)
    return validation_splits


def split_validation_groups(df, validation_groups=3):
    """
    Split validation data into validation groups
    """
    # Initialize list for validation groups
    validation_splits = []

    # Sequentially split validation_data into groups
    remaining_validation = df
    for _ in range(validation_groups - 1):
        group, remaining_validation = train_test_split(
            remaining_validation,
            test_size=1 / (validation_groups - len(validation_splits)),
            random_state=13,
        )
        validation_splits.append(group)
    validation_splits.append(remaining_validation)  # Append the last remaining part
    return validation_splits


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
