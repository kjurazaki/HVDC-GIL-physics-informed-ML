import argparse
from src.conicalInsulator import process_conical_insulator, process_es_conical_insulator
from src.datasetSurface import treat_data_surface, steadyState_data_surface
from src.plotting_Conical import load_plot_data_sr, load_plot_data_surface
from src.plotting_Conical import plot_derivative
from src.sindyEvaluate import insert_derivatives_dataframe
from src.SRRegressionEvaluate import (
    symbolic_regression,
    symbolic_regression_conductivity,
    load_and_plot_sr_equations_conductivity,
)
from src.datasetSindy import sindy_data
from src.sindyRegression import sindy_surface, grid_search_sindy
from pysr import PySRRegressor


def argparse_datatree(args, config, run_config, run_config_es):
    ## Load and treat data for df dark_currents - TDS and EC simulations. Modifier "_no_tail" removing final part of arc
    if args.task == "process_conical_insulator":
        process_conical_insulator(run_config, modifier=args.modifier)

    ## Load and treat data for df dark_currents - Electroquasistatic simulations
    elif args.task == "process_es_conical_insulator":
        process_es_conical_insulator(run_config_es)

    ## INTERFACE Get all surface data and merge in one dataframe, adding parameters and computing steady state time
    elif args.task == "treat_data_surface":
        treat_data_surface("up", modifier=args.modifier)
        steadyState_data_surface(interface="up", modifier="_no_tail")

    # 2D GAS Get all surface 2D data and merge in one dataframe, adding parameters and computing steady state time
    elif args.task == "treat_data_surface_2D":
        treat_data_surface("2D")
    ## Prepare data for SINDy
    elif args.task == "sindy_data":
        sindy_data(interface="up", step_size=8, scaling=False, modifier="_no_tail")
    else:
        print(f"Unknown task: {args.task}")


def argparse_plottree(args, config, run_config, run_config_es):
    ## Load and plot data for SR
    if args.task == "load_plot_data_sr":
        load_plot_data_sr(config["save_path"], modifier="_no_tail")

    ## Load and plot data of surface - ["Arc"] <= 0.95
    if args.task == "load_plot_data_surface":
        load_plot_data_surface(config["save_path"], modifier="_no_tail")

    ## Insert derivatives in the dataframe and plot
    if args.task == "insert_derivatives_dataframe":
        df_surface = insert_derivatives_dataframe(
            folder_path=config["folder_path"],
            step_size=40,
            modifier="_no_tail",
        )
        plot_derivative(df_surface, config["save_path"])


def argparse_symbolicRegressiontree(args, config, sr_config, run_config, run_config_es):
    ## FIT the symbolic regression. "Current (A)" or "Current (A) SLD GRD" or "steady_state_time_max_up"
    if args.task == "run_symbolic_regression":
        sr_model = symbolic_regression(
            features=sr_config["features"],
            run_symbolic=True,
            var_to_predict=sr_config["target"],
            units="h",
            run_graph=False,
            save_path=config["save_path"],
            remove_outliers=True,
            divide_by_area=False,
            modifier="_no_tail",
        )
        equation_file_path = sr_model.equation_file_.replace(".csv", "") + "_table.txt"
        with open(equation_file_path, "w") as file:
            file.write(
                sr_model.latex_table()
                .replace("Udc_{kV}", "V")
                .replace("ion_{pair rate}", "S")
            )
        print("debug")
    ## LOAD symbolic regression Model of dark currents LOAD
    if args.task == "load_sr_darkCurrents":
        models = {
            "dark_currents_gas": [
                "Current (A)",
                ".\data\pull_production\SR of dark currents (gas - grd electrode)\hall_of_fame_2024-12-19_164126.986.pkl",
                "pA",
            ],
            "dark_currents_sld": [
                "Current (A) SLD GRD",
                ".\data\pull_production\SR of dark currents (solid - grd electrode)\hall_of_fame_2024-12-19_172557.203.pkl",
                "pA",
            ],
            "steady_state_time_max_up": [
                "steady_state_time_max_up",
                ".\data\pull_production\SR of steady state time surface charging interface 1\hall_of_fame_2024-12-22_165041.182.pkl",
                "h",
            ],
        }
        for model_name, model_info in models.items():
            var_to_predict, model_path, unit = model_info
            sr_model = symbolic_regression(
                features=sr_config["features"],
                run_symbolic=False,
                var_to_predict=var_to_predict,
                units=unit,
                run_graph=False,
                save_path=config["save_path"],
                remove_outliers=False,
                model_path=model_path,
                modifier="_no_tail",
            )
        print(
            sr_model.latex_table()
            .replace("Udc_{kV}", "V")
            .replace("ion_{pair rate}", "S")
        )
    ## Run symbolic regression for conductivity
    if args.task == "run_sr_conductivity":
        model_conductivity = {
            "conductivity": [
                "conductivity",
                ".\data\pull_production\SR of conductivity\hall_of_fame_2024-12-20_102353.110.pkl",
                "S/m",
            ]
        }
        load_and_plot_sr_equations_conductivity(
            units="log(S/m)",
            load_path=model_conductivity["conductivity"][1],
            save_path=config["save_path"],
        )
        sr_model = symbolic_regression_conductivity(config["save_path"])
        equation_file_path = sr_model.equation_file_.replace(".csv", "") + "_table.txt"
        with open(equation_file_path, "w") as file:
            file.write(
                sr_model.latex_table()
                .replace("Udc_{kV}", "V")
                .replace("ion_{pair rate}", "S")
            )


def argparse_sindy(args, config, sr_config, sindy_config, run_config, run_config_es):
    ## Run sindy implicit/polynomial
    # print(sindy_config)
    if args.task == "run_sindy_conductivity":
        # ## Run SINDy
        model_data = sindy_surface(
            fit_type="polynomial",
            threshold=sindy_config["model_config"]["threshold"],
            include_bias=sindy_config["model_config"]["include_bias"],
            alpha=sindy_config["model_config"]["alpha"],
            derivative_threshold=sindy_config["model_config"]["derivative_threshold"],
            max_iter=sindy_config["model_config"]["max_iter"],
            polynomial_degree=sindy_config["model_config"]["polynomial_degree"],
            interface=sindy_config["interface"],
            evaluate=True,
            number_features_remove=[0, 5, 6, 7],
            save_path=config["save_path"] + "\\sindy_results",
            model_path=None,
            name_of_features=sindy_config[
                f"original_features_{sindy_config['interface']}"
            ],
            inverse_relation=sindy_config["model_config"]["inverse_relation"],
            normalize_columns=sindy_config["model_config"]["normalize_columns"],
            exponential_features=sindy_config["model_config"]["exponential_features"],
            time_exponent=sindy_config["model_config"]["time_exponent"],
            save_model=True,
            modifier="_no_tail",
        )
    ## Run Grid search SINDy
    if args.task == "run_sindy_gridSearch":
        grid_search_sindy(
            config["save_path"] + "\\sindy_results",
            sindy_config[f"original_features_{sindy_config['interface']}"],
            sindy_config["interface"],
            modifier="_no_tail",
        )
