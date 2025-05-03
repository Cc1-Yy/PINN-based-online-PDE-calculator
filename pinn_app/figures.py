import os
import numpy as np
from typing import Callable
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.ndimage import zoom

from pinn_app.constants import FIG_CACHE, FIG_PATHS


def get_fig(name: str, loader: Callable[[], go.Figure]) -> go.Figure:
    """
    Retrieve a cached figure or load it on demand.

    :param name: Key identifying the figure in FIG_CACHE.
    :param loader: Function that returns a plotly Figure when called.
    :return: The requested plotly Figure, or a placeholder if loading fails.
    """
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
    """
    Create a placeholder figure when the real result is unavailable.

    :return: A plotly Figure with a 'missing' annotation.
    """
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
    """
    Build a collocation points heatmap with markers.

    :param path: File path to a .npz containing 'U', 'X_col', and 'limit'.
    :return: A heatmap Figure with collocation point overlay.
    """
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
    """
    Generate side-by-side heatmaps of solution 'u' and residual 'f'.

    :param path: File path to a .npz containing 'U' and 'F'.
    :return: A subplot Figure with upsampled heatmaps for 'u' and 'f'.
    """
    upsample_factor = 10
    data = np.load(path)
    U = data['U']
    F = data['F']
    U_fine = np.asarray(zoom(U, (upsample_factor, upsample_factor)))
    F_fine = np.asarray(zoom(F, (upsample_factor, upsample_factor)))
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
    """
    Create an error heatmap from upsampled data.

    :param path: File path to a .npz containing 'r', 't', and 'Error'.
    :return: A plotly Figure displaying the interpolated error field.
    """
    zoom_factor = 10
    data = np.load(path)
    r, t, Error = data['r'], data['t'], data['Error']

    Error_fine = np.asarray(zoom(Error, zoom_factor, order=3))
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
    """
    Plot total, data, and equation losses over iterations.

    :param path: File path to a .npz containing 'loss'.
    :return: A Figure with three loss traces on a log y-axis.
    """
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
    """
    Generate side-by-side plots of left and right boundary losses.

    :param path: File path to a .npz containing 'loss_xy_l' and 'loss_xy_r'.
    :return: A subplot Figure showing boundary loss traces.
    """
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
    """
    Create a contour plot of log-magnitude spectrum.

    :param path: File path to a .npz containing 'freq_x', 'freq_t', and 'log_mag'.
    :return: A Figure with a filled contour of the spectrum.
    """
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
            line=dict(width=0),
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
