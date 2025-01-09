import seaborn as sns
from copy import deepcopy
import json
import argparse
from matplotlib import rc

from src.task_parse import (
    argparse_datatree,
    argparse_plottree,
    argparse_symbolicRegressiontree,
    argparse_sindy,
)

with open("sns_config.json", "r") as file:
    sns_config = json.load(file)
# Enable LaTeX-style text rendering
rc("font", family="serif")
sns.set_theme()
sns.set_style("darkgrid")
sns.set_theme(font="Helvetica")
sns.set_theme(
    rc=sns_config["set_theme"],
)
sns.set_context(
    "notebook",
    rc=sns_config["set_context"],
)

with open("run_config_es.json", "r") as file:
    run_config_es = json.load(file)

with open("run_config.json", "r") as file:
    run_config = json.load(file)

with open("sr_regression_config.json", "r") as file:
    sr_config = json.load(file)

with open("sindy_config.json", "r") as file:
    sindy_config = json.load(file)

with open("config.json", "r") as file:
    config = json.load(file)


def main():
    parser = argparse.ArgumentParser(description="Run different tasks for the project.")
    parser.add_argument("--task", type=str, required=True, help="Task to perform")
    parser.add_argument(
        "--modifier", type=str, default="", help="Modifier for the task"
    )
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(task=None, modifier=None)

    if not args.task:
        args.task = input("Please enter the task to perform: ")
    if not args.modifier:
        args.modifier = input("Please enter the modifier for the task (optional): ")

    #### DATA ####
    argparse_datatree(args, config, run_config, run_config_es)

    #### PLOTS ####
    argparse_plottree(args, config, run_config, run_config_es)

    #### SYMBOLIC REGRESSION ####
    argparse_symbolicRegressiontree(args, config, sr_config, run_config, run_config_es)

    #### SINDY ####
    argparse_sindy(args, config, sr_config, sindy_config, run_config, run_config_es)


if __name__ == "__main__":
    main()
