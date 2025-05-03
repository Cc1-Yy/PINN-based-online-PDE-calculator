from dash.dependencies import Input, Output
from dash import callback_context
from pinn_app.figures import *


def register_result_graph(app):
    """
    Register a callback on the Dash app to update the result graph, subtitle, and tab selections.

    :param app: The Dash application instance to attach the callback to.
    :return: None. This function registers the callback and does not return a value.
    """
    @app.callback(
        [
            Output("result-graph", "figure"),
            Output("result-subtitle", "children"),
            Output("result-tabs-row1", "value"),
            Output("result-tabs-row2", "value"),
        ],
        [
            Input("result-tabs-row1", "value"),
            Input("result-tabs-row2", "value"),
            Input("fig-interval", "n_intervals"),
        ],
    )
    def update_result_graph(val1, val2, _n):
        """
        Update the displayed result figure and subtitle based on which tab was triggered.

        :param val1: Selected tab value in the first row of result tabs.
        :param val2: Selected tab value in the second row of result tabs.
        :param _n: Interval counter from a dcc.Interval component for periodic refresh.
        :return: A tuple containing:
            - figure: The Plotly Figure to display in "result-graph".
            - subtitle: A string for "result-subtitle", derived from the title_map.
            - new_val1: Updated value for the first row tabs (or None if unchanged).
            - new_val2: Updated value for the second row tabs (or None if unchanged).
        """
        title_map = {
            "fig1": "Collocation Points (Set 1)",
            "fig2": "Solution & Residual (Set 1)",
            "fig3": "Error Distribution (Set 1)",
            "fig4": "Training Loss Curves (Set 1)",
            "fig5": "Boundary Loss (Set 1)",
            "fig6": "2D Frequency Spectrum",
            "fig7": "Collocation Points (Set 2)",
            "fig8": "Solution & Residual (Set 2)",
            "fig9": "Error Distribution (Set 2)",
            "fig10": "Training Loss Curves (Set 2)",
            "fig11": "Boundary Loss (Set 2)",
        }
        fig_func_map = {
            "fig1": lambda: make_colloc_fig(FIG_PATHS["fig1"]),
            "fig2": lambda: make_solution_residual_fig(FIG_PATHS["fig2"]),
            "fig3": lambda: make_error_fig(FIG_PATHS["fig3"]),
            "fig4": lambda: make_loss_fig(FIG_PATHS["fig4"]),
            "fig5": lambda: make_boundary_loss_fig(FIG_PATHS["fig5"]),
            "fig6": lambda: make_spectrum_fig(FIG_PATHS["fig6"]),
            "fig7": lambda: make_colloc_fig(FIG_PATHS["fig7"]),
            "fig8": lambda: make_solution_residual_fig(FIG_PATHS["fig8"]),
            "fig9": lambda: make_error_fig(FIG_PATHS["fig9"]),
            "fig10": lambda: make_loss_fig(FIG_PATHS["fig10"]),
            "fig11": lambda: make_boundary_loss_fig(FIG_PATHS["fig11"]),
        }

        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        if triggered == "result-tabs-row1":
            key = val1 or "fig1"
            new_val1, new_val2 = key, None
        elif triggered == "result-tabs-row2":
            key = val2 or "fig7"
            new_val1, new_val2 = None, key
        else:
            if val1:
                key, new_val1, new_val2 = val1, val1, None
            elif val2:
                key, new_val1, new_val2 = val2, None, val2
            else:
                key, new_val1, new_val2 = "fig1", "fig1", None

        loader = fig_func_map.get(key, lambda: go.Figure)
        fig = get_fig(key, loader)

        subtitle = title_map.get(key, "")
        if fig.layout.annotations and fig.layout.annotations[0].text == "The result has not yet been generated...":
            subtitle = ""

        return fig, subtitle, new_val1, new_val2
