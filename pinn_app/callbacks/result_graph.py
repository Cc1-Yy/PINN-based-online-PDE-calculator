import os
from dash.dependencies import Input, Output, State
from dash import callback_context
from dash.exceptions import PreventUpdate
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
        [
            State("session-id", "data"),
        ]
    )
    def update_result_graph(val1, val2, _n, session_id):
        """
        Update the displayed result figure and subtitle based on which tab was triggered.

        :param val1: Selected tab value in the first row of result tabs.
        :param val2: Selected tab value in the second row of result tabs.
        :param _n: Interval counter from a dcc.Interval component for periodic refresh.
        :param session_id:
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
        if session_id is None:
            raise PreventUpdate
        base = os.path.join(os.getcwd(), "data", session_id)
        fig_files = {
            "fig1": os.path.join(base, "collocation_point_1.npz"),
            "fig2": os.path.join(base, "solution_residual_1.npz"),
            "fig3": os.path.join(base, "error_1.npz"),
            "fig4": os.path.join(base, "loss_1.npz"),
            "fig5": os.path.join(base, "boundary_loss_1.npz"),
            "fig6": os.path.join(base, "frequency_spectrum.npz"),
            "fig7": os.path.join(base, "collocation_point_2.npz"),
            "fig8": os.path.join(base, "solution_residual_2.npz"),
            "fig9": os.path.join(base, "error_2.npz"),
            "fig10": os.path.join(base, "loss_2.npz"),
            "fig11": os.path.join(base, "boundary_loss_2.npz"),
        }
        loader_map = {
            "fig1": lambda: make_colloc_fig(fig_files["fig1"]),
            "fig2": lambda: make_solution_residual_fig(fig_files["fig2"]),
            "fig3": lambda: make_error_fig(fig_files["fig3"]),
            "fig4": lambda: make_loss_fig(fig_files["fig4"]),
            "fig5": lambda: make_boundary_loss_fig(fig_files["fig5"]),
            "fig6": lambda: make_spectrum_fig(fig_files["fig6"]),
            "fig7": lambda: make_colloc_fig(fig_files["fig7"]),
            "fig8": lambda: make_solution_residual_fig(fig_files["fig8"]),
            "fig9": lambda: make_error_fig(fig_files["fig9"]),
            "fig10": lambda: make_loss_fig(fig_files["fig10"]),
            "fig11": lambda: make_boundary_loss_fig(fig_files["fig11"]),
        }
        # loader = lambda key: {
        #     "fig1": lambda: make_colloc_fig(fig_files["fig1"]),
        #     "fig2": lambda: make_solution_residual_fig(fig_files["fig2"]),
        #     "fig3": lambda: make_error_fig(fig_files["fig3"]),
        #     "fig4": lambda: make_loss_fig(fig_files["fig4"]),
        #     "fig5": lambda: make_boundary_loss_fig(fig_files["fig5"]),
        #     "fig6": lambda: make_spectrum_fig(fig_files["fig6"]),
        #     "fig7": lambda: make_colloc_fig(fig_files["fig7"]),
        #     "fig8": lambda: make_solution_residual_fig(fig_files["fig8"]),
        #     "fig9": lambda: make_error_fig(fig_files["fig9"]),
        #     "fig10": lambda: make_loss_fig(fig_files["fig10"]),
        #     "fig11": lambda: make_boundary_loss_fig(fig_files["fig11"]),
        # }[key]()

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

        loader = loader_map.get(key, lambda: make_missing_fig())
        fig = get_fig(key, loader)

        subtitle = title_map.get(key, "")
        if fig.layout.annotations and fig.layout.annotations[0].text == "The result has not yet been generated...":
            subtitle = ""

        return fig, subtitle, new_val1, new_val2
