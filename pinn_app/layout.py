from dash.dependencies import Input, Output
import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html


def make_bd_group(idx: int):
    """
    Create a boundary/initial condition input group.

    :param idx: Unique index for this condition group (used in component IDs)
    :return: A Div containing numeric inputs for x-min, x-max, y-min, y-max, and U.
    """
    return html.Div(
        [
            html.Span("X:", className="me-1 d-flex align-items-center"),
            dbc.Input(
                type="number",
                id={"type": "bd", "field": "x-min", "idx": idx},
                placeholder="Min",
                step="any",
                className="no-spinner",
                required=True,
            ),
            html.Span("−", className="mx-1 d-flex align-items-center"),
            dbc.Input(
                type="number",
                id={"type": "bd", "field": "x-max", "idx": idx},
                placeholder="Max",
                step="any",
                className="no-spinner",
                required=True,
            ),
            html.Span("Y:", className="ms-3 me-1 d-flex align-items-center"),
            dbc.Input(
                type="number",
                id={"type": "bd", "field": "y-min", "idx": idx},
                placeholder="Min",
                step="any",
                className="no-spinner",
                required=True,
            ),
            html.Span("−", className="mx-1 d-flex align-items-center"),
            dbc.Input(
                type="number",
                id={"type": "bd", "field": "y-max", "idx": idx},
                placeholder="Max",
                step="any",
                className="no-spinner",
                required=True,
            ),
            html.Span("U:", className="ms-3 me-1 d-flex align-items-center"),
            dbc.Input(
                type="number",
                id={"type": "bd", "field": "u", "idx": idx},
                placeholder="U",
                step="any",
                className="no-spinner",
                required=True,
            ),
        ],
        className="d-flex align-items-stretch flex-nowrap mb-2",
        style={"gap": "0.42rem", "width": "100%"},
    )


def create_layout() -> Dash:
    """
    Construct the Dash application, including layout and clientside behavior.

    :return: A Dash app instance with configured layout, stylesheets, and callbacks.
    """
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.title = "PINN Training UI"
    app.layout = dbc.Container(
        [
            dcc.Store(id='log-scroll-store'),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(
                                dbc.Card(
                                    [
                                        html.Div("Problem Setup", className="settings-title"),
                                        html.Div(
                                            [
                                                # Equation
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Label("Equation:", html_for="input-equation",
                                                                          className="me-2"),
                                                                html.Span(
                                                                    "?",
                                                                    id="eq-help-icon",
                                                                    style={
                                                                        "display": "inline-block",
                                                                        "width": "18px",
                                                                        "height": "18px",
                                                                        "lineHeight": "18px",
                                                                        "textAlign": "center",
                                                                        "borderRadius": "50%",
                                                                        "backgroundColor": "#6c757d",
                                                                        "color": "white",
                                                                        "fontSize": "12px",
                                                                        "cursor": "pointer",
                                                                        "marginBottom": "4px",
                                                                    },
                                                                ),
                                                                dbc.Tooltip(
                                                                    """In the form of:
                                                                    A*u ± B*u_x ± C*u_y ± D*u_xx ± E*u_xy ± F*u_yy + G
                                                                    e.g., for Uₓₓ + 3Uᵧᵧ = 5,
                                                                    input u_xx + 3*u_yy - 5""",
                                                                    target="eq-help-icon",
                                                                    placement="right",
                                                                ),
                                                            ],
                                                            className="d-flex align-items-center mb-1",
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Span("f =",
                                                                          className="me-2 d-flex align-items-center"),
                                                                dbc.Input(
                                                                    type="text",
                                                                    id="input-equation",
                                                                    placeholder="Enter equation",
                                                                    required=True,
                                                                    debounce=True,
                                                                    invalid=False,
                                                                    style={
                                                                        "flex": "1 1 auto",
                                                                        "minWidth": "200px",
                                                                    },
                                                                ),
                                                            ],
                                                            className="mb-3",
                                                            style={
                                                                "display": "inline-flex",
                                                                "flexWrap": "nowrap",
                                                                "alignItems": "center",
                                                                "whiteSpace": "nowrap",
                                                                "width": "100%",
                                                            },
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dbc.Label("Boundary/Initial Condition:"),
                                                                dbc.Button("+",
                                                                           id="btn-add-bd",
                                                                           n_clicks=0,
                                                                           className="me-1 icon-btn",
                                                                           style={
                                                                               "margin": "0 0 4px 8px"
                                                                           }
                                                                           ),
                                                                dbc.Button("−",
                                                                           id="btn-remove-bd",
                                                                           n_clicks=0,
                                                                           className="icon-btn"
                                                                           ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            id="bd-groups",
                                                            children=[make_bd_group(1), make_bd_group(2)]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Domain Boundary
                                                html.Div(
                                                    [
                                                        dbc.Label("Domain Boundary:"),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-x-min",
                                                                        placeholder="X Min",
                                                                        step="any",
                                                                        className="no-spinner",
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-x-max",
                                                                        placeholder="X Max",
                                                                        step="any",
                                                                        className="no-spinner",
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                            ],
                                                            className="mb-2",
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-y-min",
                                                                        placeholder="Y Min",
                                                                        step="any",
                                                                        className="no-spinner",
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-y-max",
                                                                        placeholder="Y Max",
                                                                        step="any",
                                                                        className="no-spinner",
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # SCL (Frequency)
                                                html.Div(
                                                    [
                                                        dbc.Label("SCL (Frequency):"),
                                                        dbc.Input(
                                                            type="number",
                                                            id="input-scl",
                                                            placeholder="Enter scl",
                                                            min=1e-9,
                                                            step="any",
                                                            className="no-spinner",
                                                            required=True,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Epsilon (Range)
                                                html.Div(
                                                    [
                                                        dbc.Label("Epsilon (Range):"),
                                                        dbc.Input(
                                                            type="number",
                                                            id="input-epsil",
                                                            placeholder="Enter epsilon",
                                                            min=1e-9,
                                                            step="any",
                                                            className="no-spinner",
                                                            required=True,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                            ]
                                        ),
                                        html.Hr(),
                                        # Training Settings
                                        html.Div("Training Settings", className="settings-title"),
                                        html.Div(
                                            [
                                                # Sample Points
                                                html.Div(
                                                    [
                                                        dbc.Label("Sample Points:"),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-n-col",
                                                                        placeholder="N-col",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-n-bd",
                                                                        placeholder="N-bd",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=4,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-n-add",
                                                                        placeholder="N-add",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=4,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Network Size
                                                html.Div(
                                                    [
                                                        dbc.Label("Network Size:"),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-depth",
                                                                        placeholder="Depth",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-width",
                                                                        placeholder="Width",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Testing Size
                                                html.Div(
                                                    [
                                                        dbc.Label("Testing Size:"),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-testing-x",
                                                                        placeholder="X",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-testing-y",
                                                                        placeholder="Y",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Training Epoch
                                                html.Div(
                                                    [
                                                        dbc.Label("Training Epoch:"),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-epoch-adam",
                                                                        placeholder="Adam",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-epoch-lbfgs",
                                                                        placeholder="L-BFGS",
                                                                        min=1,
                                                                        step=1,
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Equation Weight
                                                html.Div(
                                                    [
                                                        dbc.Label("Equation Weight:"),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-weight-f",
                                                                        placeholder="f",
                                                                        min=1e-9,
                                                                        step="any",
                                                                        className="no-spinner",
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                                dbc.Col(
                                                                    dbc.Input(
                                                                        type="number",
                                                                        id="input-weight-df",
                                                                        placeholder="df",
                                                                        min=0,
                                                                        step="any",
                                                                        className="no-spinner",
                                                                        required=True,
                                                                    ),
                                                                    width=6,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),

                                                # Start Training
                                                dbc.Button(
                                                    "Start Training",
                                                    id="btn-start-training",
                                                    className="btn-start-training",
                                                    n_clicks=0,
                                                ),
                                            ]
                                        ),
                                    ],
                                    className="settings-card",
                                ),
                                style={"flex": "1"},
                            ),
                        ],
                        width=4,
                        className="left-col d-flex flex-column h-100",
                    ),
                    dbc.Col(
                        [
                            # Result
                            html.Div(
                                dbc.Card(
                                    [
                                        html.Div("Result", className="result-title"),
                                        html.Div(
                                            [
                                                dcc.Tabs(
                                                    id="result-tabs-row1",
                                                    value=None,
                                                    children=[
                                                        dcc.Tab(label="Collocation Point - 1", value="fig1"),
                                                        dcc.Tab(label="Solution & Residual - 1", value="fig2"),
                                                        dcc.Tab(label="Error - 1", value="fig3"),
                                                        dcc.Tab(label="Loss - 1", value="fig4"),
                                                        dcc.Tab(label="Boundary Loss - 1", value="fig5"),
                                                        dcc.Tab(label="Frequency Spectrum", value="fig6"),
                                                    ],
                                                    className="result-tabs-row",
                                                ),
                                                dcc.Tabs(
                                                    id="result-tabs-row2",
                                                    value=None,
                                                    children=[
                                                        dcc.Tab(label="Collocation Point - 2", value="fig7"),
                                                        dcc.Tab(label="Solution & Residual - 2", value="fig8"),
                                                        dcc.Tab(label="Error - 2", value="fig9"),
                                                        dcc.Tab(label="Loss - 2", value="fig10"),
                                                        dcc.Tab(label="Boundary Loss - 2", value="fig11"),
                                                    ],
                                                    className="result-tabs-row",
                                                ),
                                                html.Div(
                                                    id="result-subtitle",
                                                    className="result-subtitle",
                                                    style={"margin": "0.5rem 0", "fontWeight": "bold"},
                                                ),
                                                dcc.Graph(
                                                    id="result-graph",
                                                    style={
                                                        "flex": 1,
                                                        "display": "flex",
                                                        "height": "100%",
                                                        "width": "100%"
                                                    },
                                                    config={"responsive": True},
                                                ),
                                            ],
                                            className="d-flex flex-column flex-grow-1", style={"height": "600px"}
                                        ),
                                    ],
                                    className="result-card",
                                ),
                                style={"flex": 2, "display": "flex", "flexDirection": "column"},
                            ),

                            # Training Log
                            html.Div(
                                html.Div(
                                    [
                                        html.Div("Training Log", className="log-title"),
                                        html.Pre(
                                            id="training-log",
                                            children="Training logs will be displayed here...",
                                        ),
                                        dcc.Interval(id="log-interval", interval=1000, n_intervals=0),
                                    ],
                                    className="log-card",
                                ),
                                style={"flex": 1, "display": "flex", "flexDirection": "column"},
                            ),
                        ],
                        width=8,
                        className="right-col d-flex flex-column h-100",
                    ),
                ],
                className="h-100",
                align="stretch",
            ),
            dcc.Interval(id="fig-interval", interval=1000, n_intervals=0),
        ],
        fluid=True,
        className="app-container",
    )
    app.clientside_callback(
        """
        function(children) {
          const el = document.getElementById('training-log');
          if (el) {
            el.scrollTop = el.scrollHeight;
          }
          return '';
        }
        """,
        Output('log-scroll-store', 'data'),
        Input('training-log', 'children')
    )
    return app
