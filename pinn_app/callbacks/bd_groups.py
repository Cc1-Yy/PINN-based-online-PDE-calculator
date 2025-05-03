from dash.dependencies import Input, Output, State
from dash import callback_context
from pinn_app.layout import make_bd_group


def register_bd_groups(app):
    @app.callback(
        Output("bd-groups", "children"),
        [
            Input("btn-add-bd", "n_clicks"),
            Input("btn-remove-bd", "n_clicks"),
        ],
        State("bd-groups", "children"),
        prevent_initial_call=True
    )
    def update_bd_groups(n_add, n_remove, current):
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]
        count = len(current)
        if triggered == "btn-add-bd":
            new_count = count + 1
        elif triggered == "btn-remove-bd" and count > 1:
            new_count = count - 1
        else:
            new_count = count

        return [make_bd_group(i) for i in range(1, new_count + 1)]
