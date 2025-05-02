import glob
import os
import sys
import numpy as np
import threading
import logging
from collections import deque
from typing import Any, Dict, Callable
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import State, ALL
from dash import Dash, no_update, dcc, html
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import zoom
from scripts.software import run_pinn_training

LOG_BUFFER: deque[str] = deque(maxlen=1000)
FIG_CACHE: Dict[str, go.Figure] = {}
FIG_PATHS = {
    "fig1": "../data/collocation_point_1.npz",
    "fig2": "../data/solution_residual_1.npz",
    "fig3": "../data/error_1.npz",
    "fig4": "../data/loss_1.npz",
    "fig5": "../data/boundary_loss_1.npz",
    "fig6": "../data/frequency_spectrum.npz",
    "fig7": "../data/collocation_point_2.npz",
    "fig8": "../data/solution_residual_2.npz",
    "fig9": "../data/error_2.npz",
    "fig10": "../data/loss_2.npz",
    "fig11": "../data/boundary_loss_2.npz",
}


#  ——— Class部分 ———
class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        LOG_BUFFER.append(msg)


class Tee:
    def __init__(self, orig_stream: Any, logger: logging.Logger) -> None:
        self.orig = orig_stream
        self.logger = logger

    def write(self, data: str) -> None:
        self.orig.write(data)
        for line in data.rstrip().splitlines():
            self.logger.info(line)

    def flush(self) -> None:
        self.orig.flush()


#  ——— Setting部分 ———
def make_bd_group(idx: int):
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


#  ——— 日志函数部分 ———

def init_logger(name: str = 'pinn_logger') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter('%(message)s')
    buff = BufferHandler()
    buff.setFormatter(fmt)
    logger.addHandler(buff)

    return logger


def redirect_std_streams(logger: logging.Logger) -> None:
    sys.stdout = Tee(sys.__stdout__, logger)
    sys.stderr = Tee(sys.__stderr__, logger)


#  ——— 画图函数部分 ———

def get_fig(name: str, loader: Callable[[], go.Figure]) -> go.Figure:
    if name in FIG_CACHE:
        return FIG_CACHE[name]

    path = FIG_PATHS.get(name, "")
    if not path or not os.path.exists(path):
        return make_missing_fig()

    try:
        fig = loader()
        FIG_CACHE[name] = fig
        return fig
    except Exception:
        return make_missing_fig()


def make_missing_fig():
    fig = go.Figure()
    fig.add_annotation(
        text="The result has not yet been generated...",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=20, color="grey")
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        autosize=True,
        margin=dict(l=0, r=20, t=20, b=20),
    )
    return fig


def make_colloc_fig(path):
    data = np.load(path)
    U = data['U']
    X_col = data['X_col']
    limit = data['limit']
    x1min, x1max, x2min, x2max = limit
    x = np.linspace(x1min, x1max, U.shape[1])
    y = np.linspace(x2min, x2max, U.shape[0])
    fig = go.Figure([
        go.Heatmap(x=x, y=y, z=U, colorscale='Rainbow', colorbar=dict()),
        go.Scatter(x=X_col[:, 0], y=X_col[:, 1],
                   mode='markers', marker=dict(symbol='x', color='black', size=6),
                   name='Collocation Points')
    ])
    fig.update_layout(xaxis_title='t', yaxis_title='h', autosize=True,
                      margin=dict(l=20, r=20, t=20, b=20))
    return fig


def make_solution_residual_fig(path):
    upsample_factor = 10
    data = np.load(path)
    U = data['U']
    F = data['F']
    U_fine = zoom(U, (upsample_factor, upsample_factor))
    F_fine = zoom(F, (upsample_factor, upsample_factor))

    ny, nx = U_fine.shape
    r = np.linspace(0.1, 1.0, nx)
    theta = np.linspace(0, 2 * np.pi, ny)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['u', 'f'],
        shared_yaxes=True,
        column_widths=[0.48, 0.48],
        horizontal_spacing=0.04
    )

    fig.add_trace(
        go.Heatmap(
            x=r, y=theta, z=U_fine,
            colorscale='Jet', showscale=True,
            colorbar=dict(
                thickness=15, len=0.9,
                yanchor='middle', y=0.5,
                x=0.45, xanchor='left'
            )
        ), row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(
            x=r, y=theta, z=F_fine,
            colorscale='Jet', showscale=True,
            colorbar=dict(
                thickness=15, len=0.9,
                yanchor='middle', y=0.5,
                x=1.0, xanchor='left'
            )
        ), row=1, col=2
    )

    fig.update_xaxes(domain=[0, 0.45], row=1, col=1)
    fig.update_xaxes(domain=[0.55, 1], row=1, col=2)

    for col in [1, 2]:
        fig.update_xaxes(
            tickmode='linear', tick0=0.1, dtick=0.1,
            title_text='r', row=1, col=col
        )
        fig.update_yaxes(title_text='θ', row=1, col=col)

    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    for trace in fig.data:
        if hasattr(trace, 'colorbar'):
            trace.colorbar.lenmode = 'fraction'
            trace.colorbar.len = 1

    return fig


def make_error_fig(path):
    zoom_factor = 10
    data = np.load(path)
    r, t, Error = data['r'], data['t'], data['Error']

    Error_fine = zoom(Error, zoom_factor, order=3)
    r_fine = np.linspace(r.min(), r.max(), Error_fine.shape[1])
    t_fine = np.linspace(t.min(), t.max(), Error_fine.shape[0])

    fig = px.imshow(
        Error_fine,
        x=r_fine, y=t_fine,
        origin='lower',
        labels={'x': 'r', 'y': 'θ', 'color': ''},
        aspect='auto',
    )
    fig.update_traces(
        selector=dict(type='heatmap'),
        colorbar=dict(title='')
    )

    fig.update_layout(
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    fig.update_traces(
        hovertemplate='r=%{x:.3f}<br>θ=%{y:.3f}<br>Error=%{z:.4e}'
    )
    return fig


def make_loss_fig(path):
    data = np.load(path)
    loss = data['loss']
    iters = np.arange(loss.shape[0])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iters, y=loss[:, 0], mode='lines', name='Total Loss',
                             hovertemplate='Iter=%{x}<br>Loss=%{y:.4e}'))
    fig.add_trace(go.Scatter(x=iters, y=loss[:, 1], mode='lines', name='Data Loss',
                             hovertemplate='Iter=%{x}<br>Data Loss=%{y:.4e}'))
    fig.add_trace(go.Scatter(x=iters, y=loss[:, 2], mode='lines', name='Eqn Loss',
                             hovertemplate='Iter=%{x}<br>Eqn Loss=%{y:.4e}'))
    fig.update_yaxes(type='log')
    fig.update_layout(hovermode='x unified', autosize=True, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def make_boundary_loss_fig(path):
    data = np.load(path)
    l = data['loss_xy_l']
    r_ = data['loss_xy_r']
    iters = np.arange(l.shape[0])
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=['Boundary Loss - xy_l', 'Boundary Loss - xy_r'])
    fig.add_trace(go.Scatter(x=iters, y=l, mode='lines', name='xy_l',
                             hovertemplate='Iter=%{x}<br>Loss_xy_l=%{y:.4e}'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=iters, y=r_, mode='lines', name='xy_r',
                             hovertemplate='Iter=%{x}<br>Loss_xy_r=%{y:.4e}'),
                  row=1, col=2)
    fig.update_yaxes(type='log')
    fig.update_layout(hovermode='x unified', autosize=True, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def make_spectrum_fig(path):
    data = np.load(path)
    fx = data['freq_x']
    ft = data['freq_t']
    mz = data['log_mag']

    fig = go.Figure(
        go.Contour(
            x=fx,
            y=ft,
            z=mz,
            colorscale='Jet',
            contours=dict(
                start=mz.min(),
                end=mz.max(),
                size=(mz.max() - mz.min()) / 100,
                coloring='heatmap',
                showlines=False
            ),
            line_width=0,
            colorbar=dict(
                lenmode='fraction',
                len=1
            )
        )
    )

    fig.update_layout(
        xaxis_title='r',
        yaxis_title='t',
        xaxis=dict(range=[0, 5]),
        yaxis=dict(range=[0, 5]),
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig


#  ——— dash部分 ———


def create_dash_app() -> Dash:
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
                                                                    Au ± Buₓ ± Cuₜ ± Duₓₓ ± Euₓₜ ± Fuₜₜ + G
                                                                    e.g., for Uₓₓ + 3Uᵧᵧ = 5,
                                                                    input Uₓₓ + 3Uᵧᵧ - 5""",
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
                                                                        min=1e-9,
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
                                style={"flex": 3, "display": "flex", "flexDirection": "column"},
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

    import re
    from dash.dependencies import Input, Output
    from dash import callback_context

    @app.callback(
        Output("input-equation", "invalid"),
        Input("input-equation", "value"),
        prevent_initial_call=True,
    )
    def on_equation_change(expr: str):
        allowed_vars = [
            "u_xx", "u_xy", "u_yx", "u_yy",
            "u_x", "u_y",
            "x", "y", "u",
        ]
        num_re = r"(?:\d+\.\d*|\.\d+|\d+)"
        var_re = r"(?:%s)" % "|".join(map(re.escape, allowed_vars))
        op_re = r"(?:\+|\-|\*{1,2}|\/)"
        token_re = f"(?:{num_re}|{var_re})"
        full_re = re.compile(rf"^{token_re}(?:{op_re}{token_re})*$")

        if expr is None or expr == "":
            return False

        s = re.sub(r"\s+", "", expr)
        allow_chars = "".join(allowed_vars) + "0123456789+-*/."
        if re.search(rf"[^{re.escape(allow_chars)}]", s):
            return True

        if re.match(rf"^{op_re}|{op_re}$", s):
            return True

        if not full_re.match(s):
            return True

        return False

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
                try:
                    # params1 = {
                    #     "equation": "test equation",
                    #     "boundary": {
                    #         "bd_x1_min": 0.1,
                    #         "bd_x1_max": 0.1,
                    #         "bd_y1_min": 0,
                    #         "bd_y1_max": 6.28,
                    #         "bd_u1": 1,
                    #         "bd_x2_min": 1,
                    #         "bd_x2_max": 1,
                    #         "bd_y2_min": 0,
                    #         "bd_y2_max": 6.28,
                    #         "bd_u2": 0,
                    #     },
                    #     "domain": {
                    #         "x_min": 0.1,
                    #         "x_max": 1,
                    #         "y_min": 0,
                    #         "y_max": 6.28
                    #     },
                    #     "scl": 1,
                    #     "epsil": 1,
                    # }
                    # params2 = {
                    #     "sample_points": {
                    #         "n_col": 3000,
                    #         "n_bd": 1000,
                    #         "n_add": 1000
                    #     },
                    #     "network_size": {
                    #         "depth": 60,
                    #         "width": 6
                    #     },
                    #     "testing_size": {
                    #         "x": 111,
                    #         "y": 111
                    #     },
                    #     "training_epoch": {
                    #         "adam": 8000,
                    #         "lbfgs": 5000
                    #     },
                    #     "equation_weight": {
                    #         "f": 0.05,
                    #         "df": 0
                    #     },
                    # }
                    # run_pinn_training(
                    #     equation=params1["equation"],
                    #     boundary=params1["boundary"],
                    #     domain=params1["domain"],
                    #     scl=params1["scl"],
                    #     epsil=params1["epsil"],
                    #     sample_points=params2["sample_points"],
                    #     network_size=params2["network_size"],
                    #     testing_size=params2["testing_size"],
                    #     epochs=params2["training_epoch"],
                    #     equation_weight=params2["equation_weight"],
                    # )
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

    @app.callback(
        # —— Outputs —— #
        Output("btn-start-training", "disabled"),
        Output("btn-start-training", "style"),

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

            x_min_disabled = [False] * n_bd_groups
            x_max_disabled = [False] * n_bd_groups
            y_min_disabled = [False] * n_bd_groups
            y_max_disabled = [False] * n_bd_groups
            u_disabled = [False] * n_bd_groups
            static_disabled = [False] * (1 + len(static_vals))

        return [
            btn_disabled,
            btn_style,
            x_min_disabled,
            x_max_disabled,
            y_min_disabled,
            y_max_disabled,
            u_disabled
        ] + static_disabled

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


def start_training_thread(params: Dict[str, Any]) -> threading.Thread:
    def _train() -> None:
        run_pinn_training(**params)

    thread = threading.Thread(target=_train, daemon=True)
    thread.start()
    return thread


if __name__ == "__main__":
    logger = init_logger()
    redirect_std_streams(logger)
    for f in glob.glob("../data/*.npz"):
        try:
            os.remove(f)
        except OSError:
            pass
    app = create_dash_app()
    app.run(debug=True, use_reloader=False)
