import threading
from dash.dependencies import Input, Output, State, ALL
from dash import callback_context, no_update
from pinn_app.constants import FIG_CACHE, LOG_BUFFER
from pinn_app.software import run_pinn_training


def register_training_callbacks(app):
    """
    Register callbacks to handle starting PINN training and updating the training log display,
    as well as toggling UI component states based on training status.

    :param app: The Dash application instance to attach the callbacks to.
    :return: None. The callbacks are registered directly on the app.
    """

    @app.callback(
        Output("training-log", "children"),
        [
            Input("btn-start-training", "n_clicks"),
            Input('log-interval', 'n_intervals'),
        ],
        [
            State("input-equation", "value"),

            State({"type": "bd", "field": "x-min", "idx": ALL}, "value"),
            State({"type": "bd", "field": "x-max", "idx": ALL}, "value"),
            State({"type": "bd", "field": "y-min", "idx": ALL}, "value"),
            State({"type": "bd", "field": "y-max", "idx": ALL}, "value"),
            State({"type": "bd", "field": "u", "idx": ALL}, "value"),

            State("input-x-min", "value"),
            State("input-x-max", "value"),
            State("input-y-min", "value"),
            State("input-y-max", "value"),

            State("input-scl", "value"),
            State("input-epsil", "value"),

            State("input-n-col", "value"),
            State("input-n-bd", "value"),
            State("input-n-add", "value"),

            State("input-depth", "value"),
            State("input-width", "value"),

            State("input-testing-x", "value"),
            State("input-testing-y", "value"),

            State("input-epoch-adam", "value"),
            State("input-epoch-lbfgs", "value"),

            State("input-weight-f", "value"),
            State("input-weight-df", "value"),
        ],
        prevent_initial_call=True,
    )
    def start_training(
            n_clicks, n_intervals,
            equation,
            bd_x_min, bd_x_max, bd_y_min, bd_y_max, bd_u,
            x_min, x_max, y_min, y_max,
            scl, epsil,
            n_col, n_bd, n_add,
            depth, width,
            testing_x, testing_y,
            epoch_adam, epoch_lbfgs,
            weight_f, weight_df,
    ):
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "btn-start-training":
            def _train():
                """
                Worker function to run PINN training without blocking the UI thread.
                """
                try:
                    boundary_list = {}
                    for i, (xmin, xmax, ymin, ymax, u) in enumerate(
                            zip(bd_x_min, bd_x_max, bd_y_min, bd_y_max, bd_u),
                            start=1
                    ):
                        boundary_list[f"bd_x{i}_min"] = xmin
                        boundary_list[f"bd_x{i}_max"] = xmax
                        boundary_list[f"bd_y{i}_min"] = ymin
                        boundary_list[f"bd_y{i}_max"] = ymax
                        boundary_list[f"bd_u{i}"] = u
                    run_pinn_training(
                        equation=equation,
                        boundary=boundary_list,
                        domain={"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max},
                        scl=scl,
                        epsil=epsil,
                        sample_points={"n_col": n_col, "n_bd": n_bd, "n_add": n_add},
                        network_size={"depth": depth, "width": width},
                        testing_size={"x": testing_x, "y": testing_y},
                        epochs={"adam": epoch_adam, "lbfgs": epoch_lbfgs},
                        equation_weight={"f": weight_f, "df": weight_df},
                        log_path="../data/training.log",
                    )
                finally:
                    pass

            FIG_CACHE.clear()
            LOG_BUFFER.clear()
            threading.Thread(target=_train, daemon=True).start()
            return ""

        if trigger == "log-interval":
            if not n_clicks:
                return no_update
            return '\n'.join(LOG_BUFFER)

        return ""

    @app.callback(
        # —— Outputs —— #
        Output("btn-start-training", "disabled"),
        Output("btn-start-training", "style"),
        Output("btn-add-bd", "disabled"),
        Output("btn-remove-bd", "disabled"),

        # —— pattern outputs —— #
        Output({"type": "bd", "field": "x-min", "idx": ALL}, "disabled"),
        Output({"type": "bd", "field": "x-max", "idx": ALL}, "disabled"),
        Output({"type": "bd", "field": "y-min", "idx": ALL}, "disabled"),
        Output({"type": "bd", "field": "y-max", "idx": ALL}, "disabled"),
        Output({"type": "bd", "field": "u", "idx": ALL}, "disabled"),

        # —— static inputs —— #
        Output("input-equation", "disabled"),
        Output("input-x-min", "disabled"),
        Output("input-x-max", "disabled"),
        Output("input-y-min", "disabled"),
        Output("input-y-max", "disabled"),
        Output("input-scl", "disabled"),
        Output("input-epsil", "disabled"),
        Output("input-n-col", "disabled"),
        Output("input-n-bd", "disabled"),
        Output("input-n-add", "disabled"),
        Output("input-depth", "disabled"),
        Output("input-width", "disabled"),
        Output("input-testing-x", "disabled"),
        Output("input-testing-y", "disabled"),
        Output("input-epoch-adam", "disabled"),
        Output("input-epoch-lbfgs", "disabled"),
        Output("input-weight-f", "disabled"),
        Output("input-weight-df", "disabled"),

        # —— Inputs —— #
        Input("btn-start-training", "n_clicks"),
        Input("input-equation", "value"),
        Input("input-equation", "invalid"),
        Input("input-x-min", "value"),
        Input("input-x-max", "value"),
        Input("input-y-min", "value"),
        Input("input-y-max", "value"),
        Input("input-scl", "value"),
        Input("input-epsil", "value"),
        Input("input-n-col", "value"),
        Input("input-n-bd", "value"),
        Input("input-n-add", "value"),
        Input("input-depth", "value"),
        Input("input-width", "value"),
        Input("input-testing-x", "value"),
        Input("input-testing-y", "value"),
        Input("input-epoch-adam", "value"),
        Input("input-epoch-lbfgs", "value"),
        Input("input-weight-f", "value"),
        Input("input-weight-df", "value"),

        # —— pattern States —— #
        State({"type": "bd", "field": "x-min", "idx": ALL}, "value"),
        State({"type": "bd", "field": "x-max", "idx": ALL}, "value"),
        State({"type": "bd", "field": "y-min", "idx": ALL}, "value"),
        State({"type": "bd", "field": "y-max", "idx": ALL}, "value"),
        State({"type": "bd", "field": "u", "idx": ALL}, "value"),

        prevent_initial_call=True,
    )
    def toggle_all(
            n_clicks,
            eq_val, eq_invalid,
            x_min_val, x_max_val,
            y_min_val, y_max_val,
            scl_val, epsil_val,
            n_col_val, n_bd_val, n_add_val,
            depth_val, width_val,
            testing_x_val, testing_y_val,
            epoch_adam_val, epoch_lbfgs_val,
            weight_f_val, weight_df_val,
            x_mins, x_maxs, y_mins, y_maxs, us,
    ):
        enabled_style = {
            "backgroundColor": "#125aff",
            "pointerEvents": "auto",
            "cursor": "pointer",
        }
        disabled_style = {
            "backgroundColor": "#737373",
            "cursor": "not-allowed",
            "pointerEvents": "none",
            "color": "#ffffff",
        }
        n_bd_groups = len(x_mins)
        eq_ok = (eq_val is not None and not eq_invalid and str(eq_val).strip() != "")
        static_vals = [
            x_min_val, x_max_val,
            y_min_val, y_max_val,
            scl_val, epsil_val,
            n_col_val, n_bd_val, n_add_val,
            depth_val, width_val,
            testing_x_val, testing_y_val,
            epoch_adam_val, epoch_lbfgs_val,
            weight_f_val, weight_df_val,
        ]

        if n_clicks and n_clicks > 0:
            btn_disabled = True
            btn_style = disabled_style

            add_bd_disabled = True
            remove_bd_disabled = True

            x_min_disabled = [True] * n_bd_groups
            x_max_disabled = [True] * n_bd_groups
            y_min_disabled = [True] * n_bd_groups
            y_max_disabled = [True] * n_bd_groups
            u_disabled = [True] * n_bd_groups
            static_disabled = [True] * (1 + len(static_vals))

        else:
            all_inputs = static_vals + x_mins + x_maxs + y_mins + y_maxs + us
            all_ok = (
                    eq_ok and
                    all(v is not None and (not isinstance(v, str) or v.strip() != "")
                        for v in all_inputs)
            )
            btn_disabled = not all_ok
            btn_style = enabled_style if all_ok else disabled_style

            add_bd_disabled = False
            remove_bd_disabled = False

            x_min_disabled = [False] * n_bd_groups
            x_max_disabled = [False] * n_bd_groups
            y_min_disabled = [False] * n_bd_groups
            y_max_disabled = [False] * n_bd_groups
            u_disabled = [False] * n_bd_groups
            static_disabled = [False] * (1 + len(static_vals))

        return [
            btn_disabled,
            btn_style,
            add_bd_disabled,
            remove_bd_disabled,
            x_min_disabled,
            x_max_disabled,
            y_min_disabled,
            y_max_disabled,
            u_disabled
        ] + static_disabled
