from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import random
from itertools import cycle
from sympy import Number, sympify


def plot_distribution(df, x, y, xlabel, ylabel, title):
    """
    Plot the distribution of y for each x
    """
    grouped_results = df.groupby(x)[y]
    plt.boxplot(
        [grouped_results.get_group(iteration) for iteration in grouped_results.groups],
        tick_labels=grouped_results.groups,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_all_ionpairs(df, x, y, xlabel, ylabel, title, ln_axis=[False, False]):
    plt.figure()
    for ionpair in df["ID"].unique():
        filtered_fitted = df[df["ID"] == ionpair]
        plt.plot(filtered_fitted[x], filtered_fitted[y], label=f"ionpair = {ionpair}")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if ln_axis[0]:
        plt.xscale("log")
    if ln_axis[1]:
        plt.yscale("log")
    plt.legend()
    plt.show()


class FlexiblePlotter:
    """
    Just some scatterp plots
    Usage example:
        plotter = FlexiblePlotter(df)
        plotter.plot_single_y_axis("Time_s", "Current (A)", ["S", "Udc_kV"])
        plotter.plot_mean_std("Time_s", "Current (A)", ["S", "Udc_kV"])
        plotter.last_step("Time_s", ["S", "Udc_kV"])
        plotter.plot_single_y_axis("Udc_kV", "Current (A)", ["S"], "log")
    """

    def __init__(self, dataframe, ax):
        self.df = dataframe
        self.ax = ax

    def last_step(self, time_col, category_cols):
        # Group the dataframe by the category columns
        grouped = self.df.groupby(category_cols)

        # For each group, find the row with the maximum time_col value
        self.df = grouped.apply(lambda x: x.loc[x[time_col].idxmax()]).reset_index(
            drop=True
        )

    def plot_single_y_axis(
        self,
        x_col,
        y_col,
        category_cols,
        y_scale="linear",
        color="b",
        ax=None,
        label=None,
    ):
        if ax is not None:
            self.ax = ax
        if category_cols:
            sns.scatterplot(
                data=self.df,
                x=x_col,
                y=y_col,
                hue=category_cols[0],
                style=category_cols[1] if len(category_cols) > 1 else None,
                ax=self.ax,
            )
        else:
            sns.scatterplot(
                data=self.df,
                x=x_col,
                y=y_col,
                color=color,
                ax=self.ax,
            )
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_label(label)
        self.ax.set_yscale(y_scale)

    def plot_single_y_axis_with_categories(
        self, x_col, y_col, category_cols, y_scale="linear", ax=None
    ):
        if ax is not None:
            self.ax = ax
        if category_cols:
            category_1 = (
                category_cols[1] if len(category_cols) > 1 else category_cols[0]
            )
            sns.scatterplot(
                data=self.df,
                x=x_col,
                y=y_col,
                hue=category_cols[0],
                style=category_1,
                ax=self.ax,
            )
        else:
            sns.scatterplot(
                data=self.df,
                x=x_col,
                y=y_col,
                ax=self.ax,
            )
        self.ax.set_xlabel(x_col)
        self.ax.set_ylabel(y_col)
        self.ax.set_yscale(y_scale)

    def plot_mean_std(self, time_col, y_col, category_cols):
        group_cols = category_cols + [time_col]
        plt.figure(figsize=(10, 6))
        grouped = (
            self.df.groupby(group_cols).agg({y_col: ["mean", "std"]}).reset_index()
        )
        for name, group in grouped.groupby(category_cols):
            mean = group[(y_col, "mean")]
            std = group[(y_col, "std")]
            time = group[time_col]
            plt.errorbar(time, mean, yerr=std, label=f"{name}", fmt="o")
        plt.xlabel(time_col)
        plt.ylabel(f"{y_col} (mean ± std)")
        plt.title(f"Mean and Standard Deviation of {y_col} vs {time_col}")
        plt.legend(title=f"{category_cols}")
        plt.show()


class InteractivePlot:
    def __init__(self, df, x_col, y_col, time_col, category_cols, y_scale="linear"):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.time_col = time_col
        self.category_cols = category_cols
        self.y_scale = y_scale
        self.style_map = {}
        self.color_map = {}

    def assign_random_styles(self, categories):
        import random

        symbols = [
            "circle",
            "square",
            "diamond",
            "cross",
            "x",
            "triangle-up",
            "triangle-down",
            "star",
            "hexagram",
        ]
        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "pink",
            "brown",
            "gray",
            "olive",
        ]

        symbol_cycle = cycle(symbols)
        color_cycle = cycle(colors)

        for cat in categories:
            self.style_map[cat] = next(symbol_cycle)
            self.color_map[cat] = next(color_cycle)

    def create_plot(self):
        # Sort dataframe by time_col to ensure proper slider functionality
        self.df = self.df.sort_values(by=self.time_col)

        # Create a list of unique time points for the slider
        unique_times = self.df[self.time_col].unique()
        initial_time = unique_times[0]

        if self.category_cols:
            # Determine the second category column for styling, if provided
            category_1 = (
                self.category_cols[1]
                if len(self.category_cols) > 1
                else self.category_cols[0]
            )

            # Create tuples of the combined categories for random style assignment
            combined_categories = list(
                self.df.sort_values(by=list(set([self.category_cols[0], category_1])))[
                    [self.category_cols[0], category_1]
                ]
                .drop_duplicates()
                .itertuples(index=False, name=None)
            )

            # Assign random styles to the unique category combinations
            self.assign_random_styles(combined_categories)
        else:
            combined_categories = [(None, None)]

        # Initialize figure
        fig = go.Figure()

        # Create a trace for each category combination with initial data
        for cat_comb in combined_categories:
            if self.category_cols:
                cat1, cat2 = cat_comb
                df_filtered = self.df[
                    (self.df[self.time_col] == initial_time)
                    & (self.df[self.category_cols[0]] == cat1)
                    & (self.df[category_1] == cat2)
                ]
                marker_style = dict(
                    symbol=self.style_map[cat_comb], color=self.color_map[cat_comb]
                )
                name = f"{cat1}, {cat2}"
            else:
                df_filtered = self.df[self.df[self.time_col] == initial_time]
                marker_style = dict()
                name = "Data"

            trace = go.Scatter(
                x=df_filtered[self.x_col],
                y=df_filtered[self.y_col],
                mode="markers",
                marker=marker_style,
                name=name,
            )
            fig.add_trace(trace)

        # Create slider
        steps = []
        for i, time_point in enumerate(unique_times):
            step = dict(
                method="update",
                args=[
                    {"x": [], "y": []},
                    {
                        "title": f"{self.y_col} vs {self.x_col} at {time_point} {self.time_col}"
                    },
                ],
                label=str(time_point),
            )
            for j, cat_comb in enumerate(combined_categories):
                if self.category_cols:
                    cat1, cat2 = cat_comb
                    df_filtered = self.df[
                        (self.df[self.time_col] == time_point)
                        & (self.df[self.category_cols[0]] == cat1)
                        & (self.df[category_1] == cat2)
                    ]
                else:
                    df_filtered = self.df[self.df[self.time_col] == time_point]

                step["args"][0]["x"].append(df_filtered[self.x_col])
                step["args"][0]["y"].append(df_filtered[self.y_col])
            steps.append(step)

        sliders = [
            dict(
                active=0,
                currentvalue={"prefix": f"{self.time_col}: "},
                pad={"t": 50},
                steps=steps,
            )
        ]

        # Update layout
        fig.update_layout(
            sliders=sliders,
            yaxis_type=self.y_scale,
            xaxis_title=self.x_col,
            yaxis_title=self.y_col,
            title=f"{self.y_col} vs {self.x_col} over {self.time_col}",
            legend_title=(
                f"{self.category_cols[0]} & {category_1}"
                if self.category_cols
                else "Data"
            ),
            # Format x-axis to show whole numbers
            xaxis=dict(tickformat=",d", title=self.x_col),  # Format as integer
            # Format y-axis to use scientific notation
            yaxis=dict(
                tickformat=".2e",  # Scientific notation with 2 decimal places
                title=self.y_col,
            ),
        )

        # Add buttons for changing y-axis scale
        updatemenus = [
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[{"yaxis.type": "linear"}],
                        label="Linear Scale",
                        method="relayout",
                    ),
                    dict(
                        args=[{"yaxis.type": "log"}],
                        label="Log Scale",
                        method="relayout",
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top",
            ),
        ]
        fig.update_layout(updatemenus=updatemenus)

        return fig

    def create_parallel_plot(self, parallel_columns, short_labels=None):
        # Ensure that the columns passed in `parallel_columns` are in the dataframe
        for col in parallel_columns:
            if col not in self.df.columns:
                raise ValueError(f"Column {col} not found in the dataframe.")

        # Create shorter labels for the plot
        if short_labels is None:
            short_labels = {
                col: col.replace("_", " ").title()[:15] for col in parallel_columns
            }  # Truncate to 15 characters

        # Create the parallel coordinates plot
        fig = px.parallel_coordinates(
            self.df,
            dimensions=parallel_columns,
            labels=short_labels,  # Use the shortened labels
        )

        # Update layout for better visualization and fix formatting issues
        fig.update_layout(
            title="Parallel Coordinates Plot",
            font=dict(size=10),  # Slightly smaller font
            margin=dict(l=80, r=80, t=80, b=80),  # More margin for labels
            paper_bgcolor="white",
            plot_bgcolor="white",
            dragmode="pan",
            yaxis=dict(
                tickformat=".2e",  # Scientific notation with 2 decimal places
            ),
        )

        # Apply rotation and formatting to axis labels
        fig.update_xaxes(
            tickangle=45,
            tickmode="array",
            tickvals=[i for i in range(len(parallel_columns))],
            ticktext=[
                short_labels[col] for col in parallel_columns
            ],  # Use short labels
        )

        # Clean up y-axis and remove overlapping values
        fig.update_yaxes(
            tickangle=0,
            automargin=True,
            nticks=5,  # Reduce number of ticks to avoid clutter
        )

        return fig


def parallel_plot_darkcurrents(df, plotterly):
    parallel_cols = [
        col
        for col in df.columns
        if col
        not in [
            "Time_s",
            "Current (A)_norm",
            "Charge_norm",
            "current_density_norm_avg",
            "current_density_norm_min",
            "current_density_norm_max",
            "electric_field_norm_min",  # Always 0
        ]
    ]
    for move_element in [3, 2]:
        element = parallel_cols.pop(move_element)
        parallel_cols.append(element)

    def replace_strings(column):
        column = column.replace("gas_current_density_norm", "GC")
        column = column.replace("electric_field_norm", "E")
        column = column.replace("concentration", "C")
        return column

    short_labels = {col: replace_strings(col) for col in parallel_cols}

    fig = plotterly.create_parallel_plot(
        parallel_columns=parallel_cols, short_labels=short_labels
    )
    fig.show()


def round_expr(expr):
    """
    Round coefficients in the expression to a certain number of digits
    """
    return expr.xreplace(
        {
            n: sympify("{:.2e}".format(float(n)) if abs(n) > 10 else n)
            for n in expr.atoms(Number)
        }
    )
