import numpy as np
import pandas as pd

from utils.plotting import plot_all_ionpairs, plot_distribution
from src.comsolComponent import build_dataframe_alliterations


def conductivity_fitting():
    """
    Conductivity fitting simulations
    """
    # Remember to copy from vm_share to this folder
    folder_path = (
        "C:/Users/kenji/OneDrive/2024/Research DII006/experimentFitConductivity"
    )
    files_folder = f"{folder_path}/databases/"
    iterations = {
        "S11": np.arange(1, 26),
        "S22": np.arange(1, 26),
        "S33": np.arange(1, 31),
        "S44": np.arange(1, 31),
    }
    results = build_dataframe_alliterations(files_folder, iterations)

    # Load case 1
    df_fitted = pd.read_excel(f"{folder_path}/fitted_6_try")
    df_fitted = df_fitted[df_fitted["ID"] != "S00"]
    # Remove 'iter' that is not integer (experimental iterations) and negatives
    df_fitted = df_fitted[
        df_fitted["iter"].astype(str).apply(lambda a: a[-1] == str(0))
    ]
    df_fitted = df_fitted[df_fitted["iter"] > 0]

    # Load case 2
    df_fitted_case_two = pd.read_excel(
        f"{folder_path}/fitted_6_try", sheet_name="fitted_7_try"
    )
    df_fitted_case_two = df_fitted_case_two[
        df_fitted_case_two.apply(
            lambda row: row.astype(str).str.contains("\.7").any(), axis=1
        )
    ]
    df_fitted_case_two["'iter"] = df_fitted_case_two["iter"].apply(
        lambda a: str(a).split(".")[0]
    )
    df_fitted_case_two["iter"] = df_fitted_case_two["'iter"].astype(int)

    # Filter the results
    select_ionpair_S = "S33"
    filtered_results = results[results["ionpair_S"] == select_ionpair_S]

    plot_distribution(
        filtered_results,
        x="iter",
        y="ec3.normJ",
        xlabel="Iterations",
        ylabel="ec3.normJ",
        title=f"Distribution of ec3.normJ for ionpair_S = {select_ionpair_S}",
    )

    plot_distribution(
        filtered_results,
        x="iter",
        y="ec3.normE",
        xlabel="Iterations",
        ylabel="ec3.normE",
        title=f"Distribution of ec3.normE for ionpair_S = {select_ionpair_S}",
    )

    plot_distribution(
        results,
        x="ionpair_S",
        y="ec3.normE",
        xlabel="ionpair",
        ylabel="ec3.normE",
        title="Distribution of ec3.normE for ionpair generations",
    )

    plot_distribution(
        results,
        x="ionpair_S",
        y="ec3.normJ",
        xlabel="ionpair",
        ylabel="ec3.normJ",
        title="Distribution of ec3.normJ for ionpair generations",
    )

    # Distribution for each ion-pair generation
    plot_distribution(
        df_fitted,
        x="ID",
        y="k (S/m)",
        xlabel="ionpair",
        ylabel="k (S/m)",
        title="Distribution of k (S/m) for ionpair generations",
    )

    # Diff evolution for each iteration
    select_ionpair_S = "S44"
    filtered_fitted = df_fitted[df_fitted["ID"] == select_ionpair_S]
    plot_distribution(
        filtered_fitted,
        x="iter",
        y="DIFF",
        xlabel="ionpair",
        ylabel="DIFF",
        title="Distribution of k (S/m) for ionpair generations",
    )

    # k (S/m) for each ion-pair generation for each iteration
    plot_all_ionpairs(
        df_fitted,
        "iter",
        "k (S/m)",
        "Iterations",
        "k (S/m)",
        "Distribution of k (S/m) for each ionpair generation - case 1",
        ln_axis=[False, True],
    )

    # diff for each ion-pair generation for each iteration
    plot_all_ionpairs(
        df_fitted,
        "iter",
        "DIFF",
        "Iterations",
        "Difference (S/m)",
        "Difference of k (S/m) for each ionpair generation - case 1",
        ln_axis=[False, True],
    )

    # case 2: k (S/m) for each ion-pair generation for each iteration
    plot_all_ionpairs(
        df_fitted_case_two,
        "iter",
        "k (S/m)",
        "Iterations",
        "k (S/m)",
        "Distribution of k (S/m) for each ionpair generation - case 2",
        ln_axis=[False, True],
    )

    # case 2: diff for each ion-pair generation for each iteration
    plot_all_ionpairs(
        df_fitted_case_two,
        "iter",
        "DIFF",
        "Iterations",
        "Difference (S/m)",
        "Difference of k (S/m) for each ionpair generation - case 2",
        ln_axis=[False, True],
    )

    # case 2: k (S/m) vs k fitted for each ion-pair generation for each iteration
    plot_all_ionpairs(
        df_fitted_case_two,
        "k (S/m)",
        "DIFF",
        "Iterations",
        "k (S/m)",
        "Distribution of k (S/m) for each ionpair generation - case 2",
    )
